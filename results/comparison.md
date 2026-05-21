# RAG Retrieval Strategy Comparison

| Strategy | Context Precision | Context Recall |
|---|---|---|
| BM25 | 0.1887 | 0.4650 |
| Dense (BGE) | 0.3007 | 0.4700 |
| Hybrid RRF | 0.2589 | 0.3462 |
| Hybrid + HyDE | 0.1596 | 0.3933 |
| Hybrid + HyDE + Rerank | 0.3785 | 0.5102 |
| Agentic (ReAct) | 0.0667 | 0.1429 |

**Best overall: Hybrid + HyDE + Rerank** (highest in both metrics)
