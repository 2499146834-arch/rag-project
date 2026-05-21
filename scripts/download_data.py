import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, NUM_PASSAGES, NUM_QUERIES


def download_msmarco():
    from datasets import load_dataset

    print("[download] Building passage corpus from MS MARCO v2.1...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)

    text_to_id = {}
    queries = []

    # Collect rows and extract all unique passages first
    rows = []
    for row in ds:
        rows.append(row)
        for text in row["passages"]["passage_text"]:
            if text not in text_to_id:
                text_to_id[text] = len(text_to_id)
        # Stop when we have enough passages and QA candidates
        if len(text_to_id) >= NUM_PASSAGES and len(rows) >= NUM_QUERIES:
            break

    # Supplement from v1.1 train if not enough passages
    if len(text_to_id) < NUM_PASSAGES:
        print(f"[download] Only {len(text_to_id)} unique passages, supplementing from v1.1...")
        v1_ds = load_dataset("microsoft/ms_marco", "v1.1", split="train", streaming=True)
        for row in v1_ds:
            if "passages" in row and "passage_text" in row["passages"]:
                for text in row["passages"]["passage_text"]:
                    if text not in text_to_id:
                        text_to_id[text] = len(text_to_id)
                    if len(text_to_id) >= NUM_PASSAGES:
                        break
            if len(text_to_id) >= NUM_PASSAGES:
                break

    print(f"[download] Total {len(text_to_id)} unique passages")

    # Build passages list
    id_to_text = {v: k for k, v in text_to_id.items()}
    passages = [{"id": i, "text": id_to_text[i]} for i in range(len(text_to_id))]

    # Extract QA pairs with relevant passage IDs
    for i, row in enumerate(rows):
        if len(queries) >= NUM_QUERIES:
            break
        query_text = row["query"]
        answers = row.get("answers") or [""]
        if isinstance(answers, str):
            answers = [answers]
        answers = [a for a in answers if a.strip()]

        # Find relevant passage IDs
        p_texts = row["passages"]["passage_text"]
        selected = row["passages"]["is_selected"]
        relevant = [text_to_id[txt] for txt, sel in zip(p_texts, selected) if sel == 1]

        if query_text.strip() and answers:
            queries.append({
                "id": i,
                "question": query_text.strip(),
                "answers": answers,
                "relevant_passages": relevant,
            })

    print(f"[download] Got {len(passages)} passages, {len(queries)} QA pairs")
    return passages, queries


def download_scifact():
    from datasets import load_dataset

    print("[download] Falling back to BeIR/scifact...")
    corpus_ds = load_dataset("BeIR/scifact", "corpus", split="train", streaming=True)
    passages = []
    text_to_id = {}
    for row in corpus_ds:
        text = row["text"].strip()
        if text not in text_to_id:
            text_to_id[text] = len(text_to_id)
            passages.append({"id": len(passages), "text": text})
        if len(passages) >= NUM_PASSAGES:
            break
    print(f"[download] Corpuses: {len(passages)}")

    qrels_ds = load_dataset("BeIR/scifact", "qrels", split="train", streaming=True)
    qrels = {}
    for row in qrels_ds:
        qid = str(row["query-id"])
        pid = str(row["corpus-id"])
        qrels.setdefault(qid, []).append(pid)

    queries_ds = load_dataset("BeIR/scifact", "queries", split="train", streaming=True)
    qas = []
    for row in queries_ds:
        qid = str(row["_id"])
        qas.append({
            "id": qid,
            "question": row["text"].strip(),
            "answers": [],
            "relevant_passages": qrels.get(qid, []),
        })
        if len(qas) >= NUM_QUERIES:
            break
    print(f"[download] Got {len(qas)} QA pairs")

    return passages, qas


def run():
    passages_path = DATA_DIR / "passages.json"
    qas_path = DATA_DIR / "qas.json"

    passages, qas = None, None
    for strategy in [download_msmarco, download_scifact]:
        try:
            passages, qas = strategy()
            if passages and qas:
                break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[download] {strategy.__name__} failed: {e}")

    if not passages or not qas:
        print("[download] FATAL: all datasets failed")
        sys.exit(1)

    with open(passages_path, "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    with open(qas_path, "w", encoding="utf-8") as f:
        json.dump(qas, f, ensure_ascii=False)

    print(f"[download] Saved {len(passages)} passages -> {passages_path}")
    print(f"[download] Saved {len(qas)} QA pairs   -> {qas_path}")


if __name__ == "__main__":
    run()
