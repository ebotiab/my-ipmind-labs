import streamlit as st

st.set_page_config(
    page_title="IP Mind Labs",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Evaluations": [
        st.Page(
            "pages/benchmark_evaluation.py", title="Benchmark Evaluation", icon="📊"
        ),
        st.Page("pages/benchmark_execution.py", title="Benchmark Execution", icon="🏃"),
    ]
}

pg = st.navigation(pages)
pg.run()
