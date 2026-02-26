from datetime import date, datetime

import streamlit as st
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_uuid: str
    claim_uuid: str
    publication_number: str
    job_created_at: datetime
    claim_number: int
    claim_text: str | None
    essentiality_likelihood: str | None
    essentiality_reasoning: str | None
    implementation_likelihood: float | None
    implementation_reasoning: str | None


def _get_conn(ttl: int | None = None):
    """Return the shared Streamlit SQL connection (reads .streamlit/secrets.toml)."""
    return st.connection("postgresql", type="sql", ttl=ttl)


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def get_jobs_for_project(
    project_name: str,
    start_date: date,
    end_date: date,
    valid_claim_ids: list[str],
    batch_id: str | None = None,
    standard: str | None = None,
    filter_is_independent: bool = False,
    filter_claim_number_1: bool = False,
    limit_jobs: int = 0,
) -> list[JobRecord]:
    """
    Retrieves jobs from Supabase based on project name, creation date range,
    and checks if the claim is within the valid benchmark claim IDs.
    Deduplicates jobs by keeping only the most recently created one per claim.
    """
    clauses: list[str] = []
    params: dict = {
        "project_name": project_name,
        "start_date": start_date,
        "end_date": end_date,
    }

    base = """
    SELECT DISTINCT ON (c.id)
        j.id::text AS job_uuid,
        c.id::text AS claim_uuid,
        pat.publication_number,
        j.created_at AS job_created_at,
        c.number_in_patent AS claim_number,
        c.text AS claim_text,
        c.is_independent,
        pr.essentiality_likelihood,
        pr.essentiality_reasoning,
        pr.implementation_likelihood::float,
        pr.implementation_reasoning
    FROM projects p
    JOIN project_patents pp ON p.id = pp.project_id
    JOIN patents pat ON pp.patent_id = pat.id
    JOIN patent_claims c ON pat.id = c.patent_id
    JOIN jobs j ON j.claim_id = c.id
    LEFT JOIN prism_results pr ON pr.job_id = j.id
    WHERE p.name = :project_name
      AND p.organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
      AND j.created_at >= :start_date
      AND j.created_at <= :end_date
      AND c.id::text IN :claim_ids
    """

    # Build the tuple for the IN clause
    params["claim_ids"] = tuple(valid_claim_ids)

    if batch_id:
        clauses.append("AND j.batch_id = :batch_id")
        params["batch_id"] = batch_id

    if standard:
        clauses.append("AND pr.standard::text = :standard")
        params["standard"] = standard

    if filter_is_independent:
        clauses.append("AND c.is_independent = True")

    if filter_claim_number_1:
        clauses.append("AND c.number_in_patent = 1")

    order = "ORDER BY c.id, j.created_at DESC"

    limit_clause = ""
    if limit_jobs > 0:
        limit_clause = "LIMIT :limit_jobs"
        params["limit_jobs"] = limit_jobs

    query = "\n".join([base, *clauses, order, limit_clause])

    conn = _get_conn()
    with conn.session as session:
        result = session.execute(text(query), params)
        rows = result.mappings().all()

    if not rows:
        return []

    return [JobRecord.model_validate(dict(r)) for r in rows]


def get_recent_projects_list(limit: int = 10) -> list[str]:
    """Retrieves the names of the most recently created projects."""
    query = """
    SELECT name
    FROM projects
    WHERE organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
    ORDER BY created_at DESC
    LIMIT :limit
    """
    conn = _get_conn()
    with conn.session as session:
        result = session.execute(text(query), {"limit": limit})
        rows = result.mappings().all()

    return [r["name"] for r in rows if r["name"]]


def get_available_standards() -> list[str]:
    """
    Retrieves the possible values of the custom enum type used in the 'standard'
    column of the 'prism_benchmarks' table.
    """
    query = """
    SELECT e.enumlabel
    FROM pg_enum e
    JOIN pg_type t ON e.enumtypid = t.oid
    JOIN pg_attribute a ON a.atttypid = t.oid
    JOIN pg_class c ON a.attrelid = c.oid
    WHERE c.relname = 'prism_benchmarks' AND a.attname = 'standard'
    ORDER BY e.enumsortorder
    """
    conn = _get_conn()
    with conn.session as session:
        result = session.execute(text(query))
        rows = result.mappings().all()

    return [r["enumlabel"] for r in rows]


def get_benchmark_names(standard: str) -> list[str]:
    """Retrieves the unique benchmark names for a given standard."""
    query = """
    SELECT DISTINCT name
    FROM prism_benchmarks
    WHERE standard::text = :standard AND name IS NOT NULL
    ORDER BY name
    """
    conn = _get_conn(ttl=60*60)
    with conn.session as session:
        result = session.execute(text(query), {"standard": standard})
        rows = result.mappings().all()

    return [r["name"] for r in rows]


def get_standard_truth_labels(
    standard: str,
    benchmark_name: str,
) -> tuple[list[str], list[str]]:
    """
    Retrieves claim IDs for a given standard and benchmark name from the prism_benchmarks table,
    segregated into true positives (1) and true negatives (0).
    """
    query = """
    SELECT claim_id::text, expected_essentiality
    FROM prism_benchmarks
    WHERE standard::text = :standard AND name = :benchmark_name
    """
    conn = _get_conn()
    with conn.session as session:
        result = session.execute(
            text(query), {"standard": standard, "benchmark_name": benchmark_name}
        )
        rows = result.mappings().all()

    tp_ids = [r["claim_id"] for r in rows if r["expected_essentiality"] == 1]
    tn_ids = [r["claim_id"] for r in rows if r["expected_essentiality"] == 0]
    return tp_ids, tn_ids


def get_job_stats_for_project(
    project_name: str,
    start_date: date,
    end_date: date,
    batch_id: str | None = None,
    standard: str | None = None,
    filter_is_independent: bool = False,
    filter_claim_number_1: bool = False,
    limit_jobs: int = 0,
) -> tuple[int, int]:
    """Returns (total_jobs, unique_patents) for a project + filters."""
    clauses: list[str] = []
    params: dict = {
        "project_name": project_name,
        "start_date": start_date,
        "end_date": end_date,
    }

    base = """
    WITH filtered_claims AS (
        SELECT
            c.id AS claim_id,
            c.patent_id AS patent_id
        FROM projects p
        JOIN project_patents pp ON p.id = pp.project_id
        JOIN patent_claims c ON pp.patent_id = c.patent_id
        JOIN jobs j ON j.claim_id = c.id
        LEFT JOIN prism_results pr ON pr.job_id = j.id
        WHERE p.name = :project_name
          AND p.organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
          AND j.created_at >= :start_date
          AND j.created_at <= :end_date
    """

    if batch_id:
        clauses.append("AND j.batch_id = :batch_id")
        params["batch_id"] = batch_id

    if filter_is_independent:
        clauses.append("AND c.is_independent = True")

    if filter_claim_number_1:
        clauses.append("AND c.number_in_patent = 1")

    if standard:
        clauses.append("AND pr.standard::text = :standard")
        params["standard"] = standard

    group_and_limit = "GROUP BY c.id, c.patent_id"
    if limit_jobs > 0:
        group_and_limit += "\nLIMIT :limit_jobs"
        params["limit_jobs"] = limit_jobs

    tail = """
    )
    SELECT
        COUNT(claim_id) as total_jobs,
        COUNT(DISTINCT patent_id) as unique_patents
    FROM filtered_claims;
    """

    query = "\n".join([base, *clauses, group_and_limit, tail])

    conn = _get_conn()
    with conn.session as session:
        result = session.execute(text(query), params)
        row = result.mappings().first()

    if row is None:
        return 0, 0
    return row["total_jobs"] or 0, row["unique_patents"] or 0
