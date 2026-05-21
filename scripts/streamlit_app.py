import sys, json, os
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, str(Path(__file__).parent))

from config import EVAL_DIR, RESULTS_DIR
from retrieval import STRATEGIES, retrieve, load_data

import streamlit as st
st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG — Multi-Strategy Retrieval System")
st.caption("MS MARCO 10K passages | BGE-base-en-v1.5 (GPU) | BGE Reranker | Qdrant | RAGAS Evaluation")

passages, _ = load_data()

tab1, tab2, tab3 = st.tabs(["Search", "Results", "API"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Query", placeholder="Enter your question...")
    with col2:
        strategy = st.selectbox("Strategy", list(STRATEGIES.keys()), index=4)
        k = st.slider("Top-K", 1, 20, 10)

    if st.button("Search", use_container_width=True) and query:
        with st.spinner("Retrieving..."):
            ids = retrieve(strategy, query, passages, top_k=k)
        st.subheader(f"Results ({len(ids)} documents)")
        for i, pid in enumerate(ids):
            with st.container():
                st.markdown(f"**#{i+1}** `doc_{pid}`")
                st.text(passages[pid]["text"][:600])
                st.divider()

with tab2:
    st.subheader("Strategy Comparison")
    p = EVAL_DIR / "summary.json"
    if p.exists():
        with open(p) as f:
            summary = json.load(f)
        cols = st.columns(3)
        for i, (name, scores) in enumerate(summary.items()):
            with cols[i % 3]:
                cp = scores.get("context_precision", 0) or 0
                cr = scores.get("context_recall", 0) or 0
                st.metric(name, f"P:{cp:.3f}", f"R:{cr:.3f}")
    else:
        st.warning("No evaluation data yet")

    st.subheader("Charts")
    radar_p = RESULTS_DIR / "radar_chart.png"
    bar_p = RESULTS_DIR / "bar_chart.png"
    if radar_p.exists() and bar_p.exists():
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(radar_p), caption="Radar Chart")
        with c2:
            st.image(str(bar_p), caption="Bar Chart")

    st.subheader("CSV Export")
    csv_p = RESULTS_DIR / "comparison.csv"
    if csv_p.exists():
        with open(csv_p) as f:
            st.download_button("Download CSV", f.read(), "comparison.csv", "text/csv")

with tab3:
    st.subheader("FastAPI Endpoints")
    st.code("""
GET /health              Server health
GET /strategies          List all strategies
GET /search?q=&strategy=&k=     Search with strategy
GET /search_all?q=&k=   Compare all strategies
GET /summary             Evaluation summary JSON
GET /results/csv         Download CSV
GET /results/chart/{name}  Radar/bar chart PNG
    """)
    st.caption("FastAPI docs at http://localhost:8000/docs")
