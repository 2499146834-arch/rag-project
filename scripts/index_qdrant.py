import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, EMBED_DIR, QDRANT_URL, COLLECTION_DENSE, VECTOR_DIM,
)


def get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(url=QDRANT_URL, timeout=60)


def index_dense(client, passages, embeddings):
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client.recreate_collection(
        collection_name=COLLECTION_DENSE,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=p["id"], vector=emb.tolist(), payload={"text": p["text"]})
        for p, emb in zip(passages, embeddings)
    ]
    client.upload_points(collection_name=COLLECTION_DENSE, points=points, wait=True)
    print(f"[index:dense] Indexed {len(points)} vectors -> {COLLECTION_DENSE}")


def run(state=None):
    print("[index] Connecting to Qdrant...")
    client = get_client()
    try:
        client.get_collections()
    except Exception as e:
        print(f"[index] Qdrant connection failed: {e}")
        sys.exit(1)

    passages_path = DATA_DIR / "passages.json"
    dense_path = EMBED_DIR / "dense_embeddings.npy"

    if state and state.get("dense_indexed"):
        print("[index:dense] Already indexed")
    else:
        with open(passages_path, "r", encoding="utf-8") as f:
            passages = json.load(f)
        dense = np.load(dense_path)
        index_dense(client, passages, dense)
        if state:
            state["dense_indexed"] = True

    print("[index] Done")


if __name__ == "__main__":
    run()
