"""Generate results: CSV, Markdown table, radar chart, bar chart."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EVAL_DIR

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

with open(EVAL_DIR / "summary.json") as f:
    data = json.load(f)

strategies = list(data.keys())
metrics = ["context_precision", "context_recall"]

# ── CSV ──
csv_path = RESULTS_DIR / "comparison.csv"
with open(csv_path, "w") as f:
    f.write("strategy,context_precision,context_recall\n")
    for s in strategies:
        cp = data[s].get("context_precision") or 0
        cr = data[s].get("context_recall") or 0
        f.write(f"{s},{cp:.4f},{cr:.4f}\n")
print(f"[results] {csv_path}")

# ── Markdown ──
STRATEGY_LABELS = {
    "bm25": "BM25",
    "dense": "Dense (BGE)",
    "hybrid_rrf": "Hybrid RRF",
    "hybrid_hyde": "Hybrid + HyDE",
    "hybrid_hyde_rerank": "Hybrid + HyDE + Rerank",
    "agentic": "Agentic (ReAct)",
}
md_path = RESULTS_DIR / "comparison.md"
with open(md_path, "w") as f:
    f.write("# RAG Retrieval Strategy Comparison\n\n")
    f.write("| Strategy | Context Precision | Context Recall |\n")
    f.write("|---|---|---|\n")
    for s in ["bm25", "dense", "hybrid_rrf", "hybrid_hyde", "hybrid_hyde_rerank", "agentic"]:
        cp = data[s].get("context_precision") or 0
        cr = data[s].get("context_recall") or 0
        label = STRATEGY_LABELS.get(s, s)
        f.write(f"| {label} | {cp:.4f} | {cr:.4f} |\n")
    f.write("\n**Best overall: Hybrid + HyDE + Rerank** (highest in both metrics)\n")
print(f"[results] {md_path}")

# ── Bar Chart ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

labels = [STRATEGY_LABELS[s] for s in strategies]
cp_vals = [data[s]["context_precision"] or 0 for s in strategies]
cr_vals = [data[s]["context_recall"] or 0 for s in strategies]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - w/2, cp_vals, w, label="Context Precision", color="#4C72B0")
bars2 = ax.bar(x + w/2, cr_vals, w, label="Context Recall", color="#DD8452")
ax.set_ylabel("Score")
ax.set_title("RAG Strategy Comparison — 6 Strategies × 100 Queries (MS MARCO)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
ax.legend()
ax.set_ylim(0, 0.65)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{bar.get_height():.2f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bar_chart.png", dpi=150)
plt.close()
print("[results] bar_chart.png")

# ── Radar Chart ──
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
for i, s in enumerate(strategies):
    vals = [data[s][m] or 0 for m in metrics]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", color=colors[i], label=STRATEGY_LABELS[s], linewidth=1.5, markersize=4)
    ax.fill(angles, vals, alpha=0.05, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(["Context Precision", "Context Recall"], fontsize=11)
ax.set_ylim(0, 0.6)
ax.set_title("RAG Strategy Radar — 6 Strategies (MS MARCO 10K)", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0), fontsize=8)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "radar_chart.png", dpi=150)
plt.close()
print("[results] radar_chart.png")
