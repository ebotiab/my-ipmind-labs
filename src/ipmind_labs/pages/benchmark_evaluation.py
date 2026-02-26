import asyncio
import datetime
import random

import polars as pl
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ipmind_labs.agents.metrics_explainer_agent import ModelMetrics, get_metrics_summary
from ipmind_labs.db import (
    get_available_standards,
    get_benchmark_names,
    get_job_stats_for_project,
    get_jobs_for_project,
    get_recent_projects_list,
    get_standard_truth_labels,
)


def display_metrics(
    df: pl.DataFrame, tp_ids: list[str], tn_ids: list[str]
) -> pl.DataFrame | None:
    if "claim_uuid" not in df.columns:
        st.info(
            "Evaluation features require 'claim_uuid' to be fetched from the database. Please update the query."
        )
        return None

    valid_ids = tp_ids + tn_ids
    df_eval = df.filter(
        pl.col("claim_uuid").cast(pl.Utf8).is_in(valid_ids)
    ).with_columns(
        label=pl.when(pl.col("claim_uuid").cast(pl.Utf8).is_in(tp_ids))
        .then(1)
        .otherwise(0),
        prediction=pl.when(
            pl.col("essentiality_likelihood")
            .cast(pl.Float64, strict=False)
            .fill_null(0.0)
            > 0.7
        )
        .then(1)
        .otherwise(0),
    )

    st.markdown("---")
    st.markdown("### Evaluation Metrics")

    if len(df_eval) == 0:
        return None

    y_true = df_eval["label"].to_list()
    y_pred = df_eval["prediction"].to_list()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    accuracy = accuracy_score(y_true, y_pred)

    st.markdown(f"Evaluated **{len(df_eval)}** valid jobs")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Precision", f"{float(precision):.2%}")
    m2.metric("Recall", f"{float(recall):.2%}")
    m3.metric("F1 Score", f"{float(f1):.2%}")
    m4.metric("Accuracy", f"{float(accuracy):.2%}")

    st.markdown("#### Confusion Matrix")

    total_val = tp + fp + fn + tn

    def fmt(val: int):
        pct = (val / total_val) * 100 if total_val > 0 else 0
        return f"{val} ({pct:.1f}%)"

    cm_df = pl.DataFrame(
        {
            "Actual \\ Predicted": ["Positive", "Negative", "Total"],
            "Positive": [fmt(tp), fmt(fp), fmt(tp + fp)],
            "Negative": [fmt(fn), fmt(tn), fmt(fn + tn)],
            "Total": [fmt(tp + fn), fmt(fp + tn), fmt(total_val)],
        }
    )

    def style_cm(row):
        return [
            "font-weight: bold" if row.name == 2 or col == "Total" else ""
            for col in row.index
        ]

    styled_cm = cm_df.to_pandas().style.apply(style_cm, axis=1)
    st.dataframe(styled_cm, width=400, hide_index=True)

    if st.button("🤖 Explain Metrics"):
        with st.spinner("Analyzing metrics with AI..."):
            metrics = ModelMetrics(
                total_records=len(df_eval),
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                accuracy=float(accuracy),
                tp=int(tp),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
            )
            try:
                explanation = asyncio.run(get_metrics_summary(metrics))
                st.session_state["metrics_explanation"] = explanation
            except Exception as e:
                st.error(f"Explanation failed: {e}")

    if "metrics_explanation" in st.session_state:
        st.info(st.session_state["metrics_explanation"])

    st.markdown("---")

    return df_eval


def display_reasoning_inspection(df_eval: pl.DataFrame):
    st.markdown("### Inspect Predictions")

    fp_df = df_eval.filter((pl.col("label") == 0) & (pl.col("prediction") == 1))
    fn_df = df_eval.filter((pl.col("label") == 1) & (pl.col("prediction") == 0))
    tp_df = df_eval.filter((pl.col("label") == 1) & (pl.col("prediction") == 1))
    tn_df = df_eval.filter((pl.col("label") == 0) & (pl.col("prediction") == 0))

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            f"False Positives ({len(fp_df)})",
            f"False Negatives ({len(fn_df)})",
            f"True Positives ({len(tp_df)})",
            f"True Negatives ({len(tn_df)})",
        ]
    )

    def render_inspection_pane(df: pl.DataFrame, case_type: str):
        if len(df) == 0:
            st.success(f"No {case_type} cases found!")
            return

        state_key = f"inspection_idx_{case_type}"
        if state_key not in st.session_state:
            st.session_state[state_key] = 0

        # Ensure within bounds
        if st.session_state[state_key] >= len(df):
            st.session_state[state_key] = 0

        idx = st.session_state[state_key]
        row = df.slice(idx, 1).to_dicts()[0]

        st.markdown(f"**Showing {case_type} {idx + 1} of {len(df)}**")

        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**Job ID:** `{row.get('job_uuid', 'N/A')}`")
            st.markdown(f"**Claim ID:** `{row.get('claim_uuid', 'N/A')}`")
            st.markdown(f"**Claim Number:** {row.get('claim_number', 'N/A')}")
            st.markdown(f"**Publication:** {row.get('publication_number', 'N/A')}")
            st.markdown(f"**Is Independent:** {row.get('is_independent', 'N/A')}")
            st.markdown(
                f"**Likelihood Score:** {row.get('essentiality_likelihood', 'N/A')}"
            )
            st.markdown(
                f"**Impl. Likelihood Score:** {row.get('implementation_likelihood', 'N/A')}"
            )

        with col2:
            st.markdown("**Claim Text:**")
            st.info(row.get("claim_text", "N/A"))

        st.markdown("**Essentiality Reasoning:**")
        reasoning = row.get("essentiality_reasoning")
        if not reasoning:
            st.warning("No essentiality reasoning available")
        else:
            with st.container(border=True):
                st.markdown(reasoning)

        st.markdown("**Implementation Reasoning:**")
        impl_reasoning = row.get("implementation_reasoning")
        if not impl_reasoning:
            st.warning("No implementation reasoning available")
        else:
            with st.container(border=True):
                st.markdown(impl_reasoning)

        def go_prev(sk: str = state_key):
            st.session_state[sk] -= 1

        def go_next(sk: str = state_key):
            st.session_state[sk] += 1

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c2:
            st.button(
                "⬅️ Previous",
                key=f"prev_{case_type}",
                on_click=go_prev,
                disabled=(idx == 0),
                use_container_width=True,
            )
        with c3:
            st.button(
                "Next ➡️",
                key=f"next_{case_type}",
                on_click=go_next,
                disabled=(idx == (len(df) - 1)),
                use_container_width=True,
            )

    with tab1:
        render_inspection_pane(fp_df, "FP")

    with tab2:
        render_inspection_pane(fn_df, "FN")

    with tab3:
        render_inspection_pane(tp_df, "TP")

    with tab4:
        render_inspection_pane(tn_df, "TN")


def main():
    st.title("IP Mind - PRISM Benchmark Evaluator")

    st.markdown(
        "Evaluate PRISM Claim Analysis performance for patent jobs belonging to a specific Project."
    )

    available_standards = get_available_standards()

    with st.container(border=True):
        st.subheader("Job Filters")

        recent_projects = get_recent_projects_list(30)
        recent_projects = [p for p in recent_projects if "test" not in p]

        def populate_project_name():
            selection = st.session_state.get("project_pills")
            if selection:
                st.session_state["project_name_input"] = selection

        if recent_projects:
            st.pills(
                "Recent Projects",
                recent_projects,
                selection_mode="single",
                key="project_pills",
                on_change=populate_project_name,
            )

        col9, col10 = st.columns(2)
        with col9:
            project_name = st.text_input("Project Name", key="project_name_input")
        with col10:
            standard = st.selectbox(
                "Standard",
                options=available_standards,
                help="Select the standard to filter jobs by.",
            )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date (Job Creation)", value=datetime.date(2026, 1, 1)
            )
        with col2:
            end_date = st.date_input(
                "End Date (Job Creation)", value=datetime.date.today()
            )

        col3, col4 = st.columns(2)
        with col3:
            batch_id = st.text_input(
                "Batch ID (Optional)",
                help="Filter jobs matching a specific batch UUID.",
                value="",
            )
        with col4:
            limit_jobs = st.number_input(
                "Limit Jobs",
                min_value=0,
                max_value=100000,
                value=0,
                help="Maximum number of jobs to fetch. Set to 0 to disable.",
            )

        col5, col6 = st.columns(2)
        with col5:
            filter_is_independent = st.checkbox(
                "Must be Independent Claim",
                value=False,
                help="Only retrieve jobs associated with independent claims.",
            )
        with col6:
            filter_claim_number_1 = st.checkbox(
                "Must be Claim #1",
                value=False,
                help="Only retrieve jobs for the first claim of a patent.",
            )

        st.markdown("---")
        st.subheader("Benchmark Filters")

        benchmark_names = get_benchmark_names(standard) if standard else []
        benchmark_name = st.selectbox(
            "Benchmark Name",
            options=benchmark_names,
            help="Select the specific benchmark name to evaluate the jobs.",
        )

        col7, col8 = st.columns(2)
        with col7:
            limit_claims = st.number_input(
                "Limit Benchmark Claims",
                min_value=0,
                max_value=10000,
                value=0,
                help="Maximum number of claim IDs to extract from the benchmark for evaluation. Set to 0 to disable.",
            )
        with col8:
            balance_essentiality = st.checkbox(
                "Balance essentiality",
                value=False,
                help="Sample an equal number of Positive and Negative claims from the benchmark.",
            )

        submitted = st.button("Run Evaluation", type="primary")

    if submitted:
        if "metrics_explanation" in st.session_state:
            del st.session_state["metrics_explanation"]

        if not project_name:
            st.warning("Please enter a Project Name.")
        elif not standard:
            st.warning("Please select a Standard.")
        elif not benchmark_name:
            st.warning("Please select a Benchmark Name.")
        else:
            tp_ids, tn_ids = get_standard_truth_labels(standard, benchmark_name)

            if balance_essentiality:
                min_len = min(len(tp_ids), len(tn_ids))
                if len(tp_ids) > min_len:
                    tp_ids = random.sample(tp_ids, min_len)
                elif len(tn_ids) > min_len:
                    tn_ids = random.sample(tn_ids, min_len)

            valid_ids = tp_ids + tn_ids

            if limit_claims > 0:
                random.shuffle(valid_ids)
                valid_ids = valid_ids[:limit_claims]

            if not valid_ids:
                st.error(
                    "No valid claim IDs found for the selected standard benchmark."
                )
                return

            with st.spinner("Fetching job statistics..."):
                try:
                    total_jobs, unique_patents = get_job_stats_for_project(
                        project_name,
                        start_date,
                        end_date,
                        batch_id if batch_id else None,
                        standard,
                        filter_is_independent,
                        filter_claim_number_1,
                        limit_jobs,
                    )
                    st.session_state["job_stats"] = {
                        "total_jobs": total_jobs,
                        "unique_patents": unique_patents,
                    }
                except Exception as e:
                    st.error(f"Error fetching job stats: {e}")
                    return

            with st.spinner("Fetching jobs from Supabase..."):
                try:
                    jobs = get_jobs_for_project(
                        project_name,
                        start_date,
                        end_date,
                        valid_ids,
                        batch_id if batch_id else None,
                        standard,
                        filter_is_independent,
                        filter_claim_number_1,
                        limit_jobs,
                    )

                    if not jobs:
                        df = pl.DataFrame(
                            schema={
                                "job_uuid": pl.Utf8,
                                "claim_uuid": pl.Utf8,
                                "publication_number": pl.Utf8,
                                "job_created_at": pl.Datetime,
                                "claim_number": pl.Int64,
                                "claim_text": pl.Utf8,
                                "is_independent": pl.Boolean,
                                "essentiality_likelihood": pl.Utf8,
                                "essentiality_reasoning": pl.Utf8,
                                "implementation_likelihood": pl.Float64,
                                "implementation_reasoning": pl.Utf8,
                            }
                        )
                    else:
                        df = pl.DataFrame([j.model_dump() for j in jobs])

                    st.session_state["fetched_df"] = df
                    st.session_state["tp_ids"] = tp_ids
                    st.session_state["tn_ids"] = tn_ids
                    st.session_state["eval_project_name"] = project_name

                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    if "fetched_df" in st.session_state:
        df = st.session_state["fetched_df"]
        eval_project_name = st.session_state["eval_project_name"]
        tp_ids = st.session_state["tp_ids"]
        tn_ids = st.session_state["tn_ids"]

        if "job_stats" in st.session_state:
            stats = st.session_state["job_stats"]
            st.success(
                f"**Total jobs requested:** {stats['total_jobs']} jobs from {stats['unique_patents']} unique patents."
            )
        else:
            st.success(f"Found {len(df)} jobs for project '{eval_project_name}'.")
        st.info(
            f"**Benchmark Match:** {len(df)} jobs matched out of {len(tp_ids + tn_ids)} benchmark claims."
        )

        df_eval = None
        if tp_ids or tn_ids:
            df_eval = display_metrics(df, tp_ids, tn_ids)

        if df_eval is not None:
            st.dataframe(
                df_eval.rename(
                    mapping={
                        "prediction": "is_predicted_ess_above_0.7",
                        "label": "expected_essentiality",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            display_reasoning_inspection(df_eval)


main()
