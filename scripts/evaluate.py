import json
import sys
import time
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, EVAL_DIR, API_USAGE_FILE, MAX_API_COST_CNY, USD_TO_CNY,
    COST_PER_1M_INPUT, COST_PER_1M_OUTPUT,
    RAGAS_LLM_TIMEOUT, RAGAS_MAX_RETRIES, TOP_K_RETRIEVAL,
    RAG_API_KEY, RAG_BASE_URL, MODEL_SONNET, MODEL_HAIKU,
    NUM_QUERIES,
)

from retrieval import retrieve, STRATEGIES


def load_api_usage():
    if API_USAGE_FILE.exists():
        with open(API_USAGE_FILE, "r") as f:
            return json.load(f)
    return {"total_cost_cny": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": []}


def save_api_usage(usage):
    with open(API_USAGE_FILE, "w") as f:
        json.dump(usage, f, indent=2)


def track_cost(usage, input_tokens, output_tokens, operation):
    cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT + (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    cny = cost * USD_TO_CNY
    usage["input_tokens"] += input_tokens
    usage["output_tokens"] += output_tokens
    usage["total_cost_cny"] += cny
    usage["calls"].append({
        "operation": operation, "input_tokens": input_tokens,
        "output_tokens": output_tokens, "cost_cny": round(cny, 4),
    })
    return usage


def check_cost_limit(usage):
    if usage["total_cost_cny"] >= MAX_API_COST_CNY:
        raise RuntimeError(
            f"[evaluate] API cost limit reached: {usage['total_cost_cny']:.2f} CNY >= {MAX_API_COST_CNY} CNY"
        )


def get_ragas_llm():
    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(ChatAnthropic(
        model=MODEL_SONNET,
        anthropic_api_key=RAG_API_KEY,
        anthropic_api_url=RAG_BASE_URL,
        timeout=RAGAS_LLM_TIMEOUT,
        max_retries=RAGAS_MAX_RETRIES,
    ))


def get_ragas_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")


def run_ragas_evaluation(strategy_name, passages, qas, usage):
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall
    import warnings
    warnings.filterwarnings("ignore")

    eval_file = EVAL_DIR / f"{strategy_name}.jsonl"
    completed = set()
    if eval_file.exists():
        with open(eval_file, "r") as f:
            for line in f:
                rec = json.loads(line)
                completed.add(rec["query_id"])

    llm = get_ragas_llm()
    embeddings = get_ragas_embeddings()

    for qa in qas:
        qid = qa["id"]
        if qid in completed:
            continue

        check_cost_limit(usage)

        try:
            retrieved_ids = retrieve(strategy_name, qa["question"], passages, TOP_K_RETRIEVAL)
            contexts = [passages[pid]["text"] for pid in retrieved_ids]
            reference = qa.get("answers") or [qa.get("answer", "")]
            if isinstance(reference, list):
                reference = reference[0] if reference else ""
            if not isinstance(reference, str):
                reference = str(reference)

            dataset = Dataset.from_dict({
                "user_input": [qa["question"]],
                "retrieved_contexts": [contexts],
                "reference": [reference],
            })

            result = evaluate(
                dataset,
                metrics=[context_precision, context_recall],
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=False,
            )

            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                scores = {k: float(df[k].iloc[0]) if k in df and not df[k].isna().iloc[0] else None
                          for k in ["context_precision", "context_recall"]}
            else:
                scores = {k: float(v) if v is not None else None for k, v in dict(result).items()}
            scores["query_id"] = qid
            scores["strategy"] = strategy_name
            scores["retrieved_ids"] = retrieved_ids

            with open(eval_file, "a") as f:
                f.write(json.dumps(scores, ensure_ascii=False) + "\n")

            # Estimate token cost (conservative)
            est_input = 2000 + len(contexts) * 500
            est_output = 500
            track_cost(usage, est_input, est_output, f"ragas_{strategy_name}_q{qid}")
            save_api_usage(usage)

        except RuntimeError:
            raise
        except Exception as e:
            error_rec = {
                "query_id": qid, "strategy": strategy_name,
                "error": str(e), "retrieved_ids": [],
            }
            with open(eval_file, "a") as f:
                f.write(json.dumps(error_rec, ensure_ascii=False) + "\n")

        time.sleep(0.5)

    return usage


def compute_averages(strategy_name):
    eval_file = EVAL_DIR / f"{strategy_name}.jsonl"
    if not eval_file.exists():
        return None

    metrics_names = ["context_precision", "context_recall"]
    metrics = {k: [] for k in metrics_names}
    with open(eval_file, "r") as f:
        for line in f:
            rec = json.loads(line)
            if "error" in rec:
                continue
            for k in metrics:
                if k in rec and rec[k] is not None:
                    metrics[k].append(rec[k])

    return {k: sum(v) / len(v) if v else None for k, v in metrics.items()}


def run(strategies=None, state=None):
    if strategies is None:
        strategies = list(STRATEGIES.keys())

    with open(DATA_DIR / "passages.json", "r", encoding="utf-8") as f:
        passages = json.load(f)
    with open(DATA_DIR / "qas.json", "r", encoding="utf-8") as f:
        qas = json.load(f)[:NUM_QUERIES]

    usage = load_api_usage()
    print(f"[evaluate] Current API cost: {usage['total_cost_cny']:.2f} CNY")

    summary = {}
    for strategy_name in strategies:
        try:
            check_cost_limit(usage)
            print(f"\n[evaluate] Strategy: {strategy_name}")
            run_ragas_evaluation(strategy_name, passages, qas, usage)
            avg = compute_averages(strategy_name)
            summary[strategy_name] = avg
            if avg:
                print(f"  Avg -> {', '.join(f'{k}={v:.3f}' for k, v in avg.items() if v is not None)}")
        except RuntimeError as e:
            print(f"  COST LIMIT: {e}")
            break

    summary_path = EVAL_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[evaluate] Summary saved to {summary_path}")
    print(f"[evaluate] Total API cost: {usage['total_cost_cny']:.2f} CNY")

    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy to evaluate")
    args = parser.parse_args()
    strategies = [args.strategy] if args.strategy else None
    run(strategies=strategies)
