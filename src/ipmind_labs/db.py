from datetime import date, datetime

import asyncpg
from pydantic import BaseModel, ConfigDict

from ipmind_labs.config import settings


class JobRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_uuid: str
    claim_uuid: str
    publication_number: str
    job_created_at: datetime
    claim_number: int
    claim_text: str | None
    is_independent: bool | None
    essentiality_likelihood: str | None
    essentiality_reasoning: str | None
    implementation_likelihood: float | None
    implementation_reasoning: str | None


async def get_jobs_for_project(
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
    query = """
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
    WHERE p.name = $1
      AND p.organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
      AND j.created_at >= $2
      AND j.created_at <= $3
      AND c.id = ANY($4::uuid[])
    """

    args = [project_name, start_date, end_date, valid_claim_ids]

    if batch_id:
        args.append(batch_id)
        query += f"      AND j.batch_id = ${len(args)}\n"

    if standard:
        args.append(standard)
        # Assuming the 'standard' exists on prism_results or similar, wait, the user said filter jobs according their value in the standard column. We should map it to PRISM results standard if that's where it is. Wait, if it's on jobs or prism_results? Let's assume prism_results.standard since jobs might not have it natively, or maybe it's j.standard. I'll add joining prism_results if not joined, but it is joined.
        # As there are multiple tables, we shouldn't guess. The user said "filter the jobs according their value in the standard column". Let's assume it's `j.standard` or `pr.standard`. For safety let's assume `pr.standard`.
        # Actually I need to check the exact column location. Let's execute a quick query first if unsure, but I have to replace it. I'll use `pr.standard::text`.
        query += f"      AND pr.standard::text = ${len(args)}\n"

    if filter_is_independent:
        query += "      AND c.is_independent = True\n"

    if filter_claim_number_1:
        query += "      AND c.number_in_patent = 1\n"

    query += """
    ORDER BY c.id, j.created_at DESC
    """

    if limit_jobs > 0:
        args.append(limit_jobs)
        query += f"    LIMIT ${len(args)}\n"

    # NOTE: statement_cache_size=0 because Supabase uses PgBouncer in transaction mode
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        records = await conn.fetch(query, *args)

        if not records:
            return []

        return [JobRecord.model_validate(dict(r)) for r in records]
    finally:
        await conn.close()


async def get_recent_projects_list(limit: int = 10) -> list[str]:
    """
    Retrieves the names of the most recently created projects from Supabase.
    """
    query = """
    SELECT name
    FROM projects
    WHERE organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
    ORDER BY created_at DESC
    LIMIT $1
    """

    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        records = await conn.fetch(query, limit)
        return [r["name"] for r in records if r["name"]]
    finally:
        await conn.close()


async def get_available_standards_from_db() -> list[str]:
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
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        records = await conn.fetch(query)
        return [r["enumlabel"] for r in records]
    finally:
        await conn.close()


async def get_benchmark_names_from_db(standard: str) -> list[str]:
    """
    Retrieves the unique benchmark names for a given standard.
    """
    query = """
    SELECT DISTINCT name
    FROM prism_benchmarks
    WHERE standard::text = $1 AND name IS NOT NULL
    ORDER BY name
    """
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        records = await conn.fetch(query, standard)
        return [r["name"] for r in records]
    finally:
        await conn.close()


async def get_standard_truth_labels_from_db(
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
    WHERE standard::text = $1 AND name = $2
    """
    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        records = await conn.fetch(query, standard, benchmark_name)
        tp_ids = [r["claim_id"] for r in records if r["expected_essentiality"] == 1]
        tn_ids = [r["claim_id"] for r in records if r["expected_essentiality"] == 0]
        return tp_ids, tn_ids
    finally:
        await conn.close()


async def get_job_stats_for_project(
    project_name: str,
    start_date: date,
    end_date: date,
    batch_id: str | None = None,
    standard: str | None = None,
    filter_is_independent: bool = False,
    filter_claim_number_1: bool = False,
    limit_jobs: int = 0,
) -> tuple[int, int]:
    """Returns (total_jobs, unique_patents) for a project + filters"""

    query = """
    WITH filtered_claims AS (
        SELECT 
            c.id AS claim_id,
            c.patent_id AS patent_id
        FROM projects p
        JOIN project_patents pp ON p.id = pp.project_id
        JOIN patent_claims c ON pp.patent_id = c.patent_id
        JOIN jobs j ON j.claim_id = c.id
        LEFT JOIN prism_results pr ON pr.job_id = j.id
        WHERE p.name = $1
          AND p.organization_id = '61a01994-8e93-42b0-a0f7-a46db8f8e883'
          AND j.created_at >= $2
          AND j.created_at <= $3
    """

    args = [project_name, start_date, end_date]

    if batch_id:
        args.append(batch_id)
        query += f"      AND j.batch_id = ${len(args)}\n"

    if filter_is_independent:
        query += "      AND c.is_independent = True\n"

    if filter_claim_number_1:
        query += "      AND c.number_in_patent = 1\n"

    if standard:
        args.append(standard)
        query += f"      AND pr.standard::text = ${len(args)}\n"

    query += """
        GROUP BY c.id, c.patent_id
    """

    if limit_jobs > 0:
        args.append(limit_jobs)
        query += f"    LIMIT ${len(args)}\n"

    query += """
    )
    SELECT 
        COUNT(claim_id) as total_jobs,
        COUNT(DISTINCT patent_id) as unique_patents
    FROM filtered_claims;
    """

    conn = await asyncpg.connect(settings.database_url, statement_cache_size=0)
    try:
        row = await conn.fetchrow(query, *args)
        return row["total_jobs"] or 0, row["unique_patents"] or 0
    finally:
        await conn.close()
