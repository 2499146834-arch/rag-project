import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()

# API
RAG_API_KEY = os.getenv("RAG_API_KEY")
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "https://api.aipaibox.com")
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-sonnet-4-6"

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EMBED_DIR = PROJECT_ROOT / "embeddings"
EVAL_DIR = PROJECT_ROOT / "evaluation"
QDRANT_STORAGE = PROJECT_ROOT / "qdrant_storage"
STATE_FILE = PROJECT_ROOT / "pipeline_state.json"
API_USAGE_FILE = PROJECT_ROOT / "api_usage.json"

# Datasets
NUM_PASSAGES = 10_000
NUM_QUERIES = 100
FALLBACK_DATASET = ("BeIR/scifact",)

# Embedding
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_BATCH_SIZE = 32
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANKER_BATCH_SIZE = 16
EMBED_CHECKPOINT_INTERVAL = 500

# Qdrant
QDRANT_URL = "http://localhost:6333"
COLLECTION_DENSE = "msmarco_dense"
COLLECTION_SPARSE = "msmarco_bm25"
VECTOR_DIM = 768
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 10

# Cost
MAX_API_COST_CNY = float("inf")  # unlimited
USD_TO_CNY = 7.2
# Claude Sonnet 4.6 pricing (USD per 1M tokens)
COST_PER_1M_INPUT = 3.0
COST_PER_1M_OUTPUT = 15.0

# RAGAS
RAGAS_LLM_TIMEOUT = 120
RAGAS_MAX_RETRIES = 3

for d in [DATA_DIR, EMBED_DIR, EVAL_DIR, QDRANT_STORAGE]:
    d.mkdir(parents=True, exist_ok=True)
