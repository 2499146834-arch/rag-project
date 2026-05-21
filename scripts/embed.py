import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, EMBED_DIR, EMBED_MODEL, EMBED_BATCH_SIZE,
    EMBED_CHECKPOINT_INTERVAL, NUM_PASSAGES,
)


def load_passages():
    with open(DATA_DIR / "passages.json", "r", encoding="utf-8") as f:
        return json.load(f)


class DenseEmbedder:
    def __init__(self, model_name=EMBED_MODEL, batch_size=EMBED_BATCH_SIZE):
        from sentence_transformers import SentenceTransformer

        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embed:dense] Loading {model_name} on {self.device}...")
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, texts, show_progress=True):
        return self.model.encode(
            texts, batch_size=self.batch_size, show_progress_bar=show_progress,
            convert_to_numpy=True, normalize_embeddings=True,
        )

    def cleanup(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SparseEmbedder:
    def __init__(self):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
        self.bm25 = None

    def fit(self, texts):
        from rank_bm25 import BM25Okapi

        tokenized = [self.tokenizer.tokenize(t.lower()) for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def get_scores(self, query):
        tokenized = self.tokenizer.tokenize(query.lower())
        return np.array(self.bm25.get_scores(tokenized), dtype=np.float32)

    def get_top_k(self, query, k=10):
        tokenized = self.tokenizer.tokenize(query.lower())
        return self.bm25.get_top_n(tokenized, list(range(len(self.bm25.doc_len))), n=k)


def run_dense_embeddings(passages, state):
    dense_path = EMBED_DIR / "dense_embeddings.npy"
    offset = state.get("dense_offset", 0)
    texts = [p["text"] for p in passages]

    if offset >= len(texts):
        print(f"[embed:dense] Already complete ({offset}/{len(texts)})")
        return

    embedder = DenseEmbedder()
    try:
        if offset == 0:
            all_embs = np.zeros((len(texts), 768), dtype=np.float32)
        else:
            all_embs = np.load(dense_path)

        for start in range(offset, len(texts), EMBED_CHECKPOINT_INTERVAL):
            end = min(start + EMBED_CHECKPOINT_INTERVAL, len(texts))
            batch_texts = texts[start:end]
            embs = embedder.encode(batch_texts, show_progress=False)
            all_embs[start:end] = embs
            np.save(dense_path, all_embs)
            state["dense_offset"] = end
            print(f"[embed:dense] Checkpoint {end}/{len(texts)}")
            time.sleep(0.5)
    finally:
        embedder.cleanup()


def run_sparse_embeddings(passages, state):
    sparse_done = state.get("sparse_done", False)
    if sparse_done:
        print("[embed:sparse] Already complete")
        return

    texts = [p["text"] for p in passages]
    embedder = SparseEmbedder()
    embedder.fit(texts)

    import pickle

    with open(EMBED_DIR / "bm25_index.pkl", "wb") as f:
        pickle.dump(embedder, f)
    state["sparse_done"] = True
    print(f"[embed:sparse] BM25 fitted, {len(embedder.bm25.doc_len)} docs")


def run(state=None):
    if state is None:
        state = {"dense_offset": 0, "sparse_done": False}
    # Auto-detect completed dense embeddings
    dense_path = EMBED_DIR / "dense_embeddings.npy"
    if dense_path.exists() and state.get("dense_offset", 0) == 0:
        import numpy as np
        existing = np.load(dense_path)
        if existing.shape[0] >= NUM_PASSAGES:
            state["dense_offset"] = existing.shape[0]
            print(f"[embed] Found existing dense embeddings: {existing.shape[0]} vectors")

    passages = load_passages()
    print(f"[embed] Loaded {len(passages)} passages")

    run_dense_embeddings(passages, state)
    run_sparse_embeddings(passages, state)
    print("[embed] Done")


if __name__ == "__main__":
    run()
