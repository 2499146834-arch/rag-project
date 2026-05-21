import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, EMBED_DIR, QDRANT_URL, COLLECTION_DENSE,
    EMBED_MODEL, RERANKER_MODEL, TOP_K_RETRIEVAL, TOP_K_RERANK,
    MODEL_SONNET, MODEL_HAIKU, RAG_API_KEY, RAG_BASE_URL,
)

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


def load_data():
    with open(DATA_DIR / "passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)
    with open(DATA_DIR / "qas.json", "r", encoding="utf-8") as f:
        qas = json.load(f)
    return passages, qas


def get_qdrant():
    return QdrantClient(url=QDRANT_URL, timeout=60)


# ── Strategy 1: BM25 ──
def retrieve_bm25(query, passages, top_k=TOP_K_RETRIEVAL):
    with open(EMBED_DIR / "bm25_index.pkl", "rb") as f:
        bm25 = pickle.load(f)
    results = []
    for q in query if isinstance(query, list) else [query]:
        results.append(bm25.get_top_k(q, top_k))
    return results[0] if not isinstance(query, list) else results


# ── Strategy 2: Dense ──
class DenseRetriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL, device="cuda")
        self.client = get_qdrant()
        self.passages = None

    def retrieve(self, query, top_k=TOP_K_RETRIEVAL):
        vec = self.model.encode(query, normalize_embeddings=True).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION_DENSE, query=vec, limit=top_k,
        )
        return [hit.id for hit in result.points]


_dense_retriever = None


def _get_dense():
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = DenseRetriever()
    return _dense_retriever


def retrieve_dense(query, top_k=TOP_K_RETRIEVAL):
    return _get_dense().retrieve(query, top_k)


# ── Strategy 3: Hybrid RRF ──
def _rrf_fusion(results_a, results_b, k=60, top_k=TOP_K_RETRIEVAL):
    scores = {}
    for rank, doc_id in enumerate(results_a):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(results_b):
        scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
    sorted_ids = sorted(scores, key=scores.get, reverse=True)
    return sorted_ids[:top_k]


def retrieve_hybrid_rrf(query, passages, top_k=TOP_K_RETRIEVAL):
    bm25_results = retrieve_bm25(query, passages, top_k=top_k * 2)
    dense_results = retrieve_dense(query, top_k=top_k * 2)
    return _rrf_fusion(bm25_results, dense_results, top_k=top_k)


# ── Strategy 4: Hybrid + HyDE ──
def _generate_hyde(query):
    from anthropic import Anthropic

    client = Anthropic(api_key=RAG_API_KEY, base_url=RAG_BASE_URL)
    resp = client.messages.create(
        model=MODEL_SONNET,
        max_tokens=256,
        system="Write a short scientific passage that answers the user's question. Output ONLY the passage, no extra text.",
        messages=[{"role": "user", "content": query}],
    )
    return resp.content[0].text


def retrieve_hybrid_hyde(query, passages, top_k=TOP_K_RETRIEVAL):
    hypo_doc = _generate_hyde(query)
    dense_results = retrieve_dense(hypo_doc, top_k=top_k * 2)
    bm25_full = retrieve_bm25(query, passages, top_k=top_k * 2)
    return _rrf_fusion(bm25_full, dense_results, top_k=top_k)


# ── Strategy 5: Hybrid + HyDE + Rerank ──
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from FlagEmbedding import FlagReranker
        import torch

        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=True)
    return _reranker


def retrieve_hybrid_hyde_rerank(query, passages, top_k=TOP_K_RETRIEVAL):
    candidates = retrieve_hybrid_hyde(query, passages, top_k=TOP_K_RETRIEVAL * 2)
    reranker = _get_reranker()
    pairs = [[query, passages[pid]["text"]] for pid in candidates]
    scores = reranker.compute_score(pairs)
    if not isinstance(scores, list):
        scores = scores.tolist()
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:top_k]]


# ── Strategy 6: Agentic (ReAct + Reflection) ──
def retrieve_agentic(query, passages, top_k=TOP_K_RETRIEVAL, max_rounds=3):
    from anthropic import Anthropic

    client = Anthropic(api_key=RAG_API_KEY, base_url=RAG_BASE_URL)
    context_pool = {}
    reflection_notes = ""

    system_prompt = (
        "You are a ReAct retrieval agent. Your goal is to find the best passages "
        "to answer a question. You have access to these tools:\n\n"
        "- search_bm25(query: str) -> list of (passage_id, passage_text)\n"
        "- search_dense(query: str) -> list of (passage_id, passage_text)\n"
        "- think(thought: str) -> log your reasoning\n\n"
        "On each turn output exactly one action in this format:\n"
        "THOUGHT: <reasoning>\n"
        "ACTION: <tool_name>(<args>)\n\n"
        "Or when you have enough results:\n"
        "THOUGHT: <final reasoning>\n"
        "FINAL: <comma-separated passage_ids in priority order>\n\n"
        f"You may search up to {max_rounds} times. After each search you will see the results."
    )

    def tool_search_bm25(q):
        ids = retrieve_bm25(q, passages, top_k=top_k)
        results = [(pid, passages[pid]["text"][:300]) for pid in ids]
        context_pool.update({pid: passages[pid]["text"] for pid in ids})
        return results

    def tool_search_dense(q):
        ids = retrieve_dense(q, top_k=top_k)
        results = [(pid, passages[pid]["text"][:300]) for pid in ids]
        context_pool.update({pid: passages[pid]["text"] for pid in ids})
        return results

    messages = [{"role": "user", "content": f"Question: {query}"}]

    for turn in range(max_rounds):
        resp = client.messages.create(
            model=MODEL_SONNET, max_tokens=512,
            system=system_prompt,
            messages=messages,
        )
        content = resp.content[0].text.strip()
        messages.append({"role": "assistant", "content": content})

        if "FINAL:" in content:
            final_line = content.split("FINAL:")[-1].strip()
            ids = [int(x.strip()) for x in final_line.split(",") if x.strip().isdigit()]
            break

        # Parse action
        action_line = None
        for line in content.split("\n"):
            if line.strip().startswith("ACTION:"):
                action_line = line.split("ACTION:", 1)[-1].strip()
                break

        if not action_line:
            observed = "Error: no ACTION found. Respond with THOUGHT + ACTION or FINAL."
        elif "search_bm25" in action_line:
            arg = action_line.split("(", 1)[-1].rstrip(")")
            arg = arg.strip().strip("\"'")
            results = tool_search_bm25(arg)
            observed = f"BM25 results for '{arg}':\n" + "\n".join(
                f"  [{pid}] {txt[:200]}" for pid, txt in results
            )
        elif "search_dense" in action_line:
            arg = action_line.split("(", 1)[-1].rstrip(")")
            arg = arg.strip().strip("\"'")
            results = tool_search_dense(arg)
            observed = f"Dense results for '{arg}':\n" + "\n".join(
                f"  [{pid}] {txt[:200]}" for pid, txt in results
            )
        elif "think" in action_line.lower():
            observed = "Thought logged."
        else:
            observed = f"Unknown action: {action_line}"

        # Reflection
        reflect_resp = client.messages.create(
            model=MODEL_HAIKU, max_tokens=256,
            system="You are a reflection agent. Review the observation and decide if the retrieved passages are relevant and sufficient. Be concise.",
            messages=[{"role": "user", "content": f"Question: {query}\nObservation: {observed}\nContext pool size: {len(context_pool)}\n\nVerdict: sufficient or need more? One sentence."}],
        )
        reflection_notes += f"[Turn {turn+1}] {reflect_resp.content[0].text}\n"
        messages.append({"role": "user", "content": f"OBSERVATION: {observed}\nREFLECTION: {reflection_notes.split(chr(10))[-1]}"})

    else:
        ids = list(context_pool.keys())[:top_k]

    return ids[:top_k]


# ── Dispatcher ──
STRATEGIES = {
    "bm25": retrieve_bm25,
    "dense": retrieve_dense,
    "hybrid_rrf": retrieve_hybrid_rrf,
    "hybrid_hyde": retrieve_hybrid_hyde,
    "hybrid_hyde_rerank": retrieve_hybrid_hyde_rerank,
    "agentic": retrieve_agentic,
}


def retrieve(strategy_name, query, passages, top_k=TOP_K_RETRIEVAL):
    fn = STRATEGIES[strategy_name]
    if strategy_name == "bm25":
        return fn(query, passages, top_k)
    elif strategy_name in ("dense",):
        return fn(query, top_k)
    elif strategy_name in ("hybrid_rrf", "hybrid_hyde", "hybrid_hyde_rerank"):
        return fn(query, passages, top_k)
    elif strategy_name == "agentic":
        return fn(query, passages, top_k)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")
