import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, EVAL_DIR, RESULTS_DIR, TOP_K_RETRIEVAL
from retrieval import STRATEGIES, retrieve, load_data

# ═══════════════════════════════════════════════════════════
#  FastAPI Backend
# ═══════════════════════════════════════════════════════════
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import uvicorn

api = FastAPI(title="Agentic RAG API", version="2.0", docs_url="/docs")
_passages, _qas = None, None

def _ensure_data():
    global _passages, _qas
    if _passages is None:
        _passages, _qas = load_data()

@api.get("/health")
def health():
    return {"status": "ok", "strategies": len(STRATEGIES)}

@api.get("/strategies")
def list_strategies():
    return {"strategies": [{"id": k, "label": k} for k in STRATEGIES]}

@api.get("/search")
def search(q: str, strategy: str = "dense", k: int = TOP_K_RETRIEVAL):
    _ensure_data()
    ids = retrieve(strategy, q, _passages, top_k=k)
    results = [{"id": pid, "text": _passages[pid]["text"][:500]} for pid in ids]
    return {"query": q, "strategy": strategy, "results": results}

@api.get("/search_all")
def search_all(q: str, k: int = 5):
    _ensure_data()
    out = {}
    for name in STRATEGIES:
        ids = retrieve(name, q, _passages, top_k=k)
        out[name] = [{"id": pid, "text": _passages[pid]["text"][:300]} for pid in ids]
    return {"query": q, "results": out}

@api.get("/summary")
def summary():
    p = EVAL_DIR / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return JSONResponse({"error": "No summary"}, status_code=404)

@api.get("/results/csv")
def results_csv():
    p = RESULTS_DIR / "comparison.csv"
    return FileResponse(p, media_type="text/csv") if p.exists() else JSONResponse({"error": "No CSV"}, status_code=404)

@api.get("/results/chart/{name}")
def results_chart(name: str):
    p = RESULTS_DIR / f"{name}.png"
    return FileResponse(p, media_type="image/png") if p.exists() else JSONResponse({"error": "No chart"}, status_code=404)


# ═══════════════════════════════════════════════════════════
#  Streamlit Frontend
# ═══════════════════════════════════════════════════════════
_STREAMLIT_CODE = r'''
import sys, json, os
PROJECT_ROOT = os.environ.get("RAG_PROJECT_ROOT", os.getcwd())
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from config import EVAL_DIR, RESULTS_DIR
from retrieval import STRATEGIES, retrieve, load_data

import streamlit as st
st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG — Multi-Strategy Retrieval System")
st.caption("MS MARCO 10K passages | BGE-base-en-v1.5 (GPU) | BGE Reranker | Qdrant | RAGAS Evaluation")

passages, _ = load_data()

tab1, tab2, tab3 = st.tabs(["Search", "Results", "API"])

# ── Tab 1: Search ──
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

# ── Tab 2: Results ──
with tab2:
    st.subheader("Strategy Comparison")
    p = EVAL_DIR / "summary.json"
    if p.exists():
        with open(p) as f:
            s = json.load(f)
        cols = st.columns(3)
        for i, (name, scores) in enumerate(s.items()):
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
    else:
        st.info("Run scripts/generate_results.py to create charts")

    st.subheader("CSV Export")
    csv_p = RESULTS_DIR / "comparison.csv"
    if csv_p.exists():
        with open(csv_p) as f:
            st.download_button("Download CSV", f.read(), "comparison.csv", "text/csv")

# ── Tab 3: API ──
with tab3:
    st.subheader("FastAPI Endpoints")
    st.code("""
GET /health             Server health
GET /strategies         List all strategies
GET /search?q=&strategy=&k=     Search with strategy
GET /search_all?q=&k=  Compare all strategies
GET /summary            Evaluation summary JSON
GET /results/csv        Download CSV
GET /results/chart/{name}  Radar/bar chart PNG
    """)
    st.caption("FastAPI docs at http://localhost:8000/docs")
'''


def run_streamlit():
    import subprocess
    app_path = str(Path(__file__).parent / "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path,
                    "--server.port", "8501", "--server.headless", "true"])


if __name__ == "__main__":
    import argparse, threading
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["api", "ui", "both"], default="both")
    args = parser.parse_args()
    if args.mode == "api":
        uvicorn.run(api, host="0.0.0.0", port=8000)
    elif args.mode == "ui":
        run_streamlit()
    else:
        t = threading.Thread(target=uvicorn.run, args=(api,), kwargs={"host": "0.0.0.0", "port": 8000}, daemon=True)
        t.start()
        run_streamlit()
