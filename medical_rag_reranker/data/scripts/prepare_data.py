"""
Подготавливает воспроизводимый датасет для baseline‑оценки retrieval в RAG‑системе.

Артефакты:
- qa.jsonl: пары вопрос–ответ (ground truth);
- corpus.jsonl: корпус для индексации (baseline: ответы как документы);
- splits.json: детерминированные train/val/test по question_id;
- eval_queries.jsonl: поднабор тестовых запросов для быстрых прогонов;
- qrels.tsv: релевантности в формате TREC для метрик Precision@k / Recall@k / NDCG@k.

Используется для честного сравнения BM25 / dense / hybrid retrieval
без утечек данных. При наличии внешнего корпуса (PubMed, StatPearls)
его можно подмешать как дополнительные документы.
"""

import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple


SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EVAL_SIZE = 300  # набор запросов для промежуточных прогонов


def read_nih_qa(path: Path) -> List[Dict[str, Any]]:
    """
    Читает QA-данные из JSONL или CSV (MedQuAD).
    Важно: вернуть список dict с полями question_id, question, answer, source, topic (опционально).
    """
    data: List[Dict[str, Any]] = []

    def pick(row: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return row[k]
        return None

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                obj = json.loads(line)
                data.append({
                    "question_id": obj.get("question_id") or str(idx),
                    "question": obj.get("question"),
                    "answer": obj.get("answer"),
                    "source": obj.get("source", "NIH"),
                    "topic": obj.get("topic"),
                })
        return data

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                question = pick(row, ["question"])
                answer = pick(row, ["answer"])
                source = pick(row, ["source"]) or "MedQuAD"
                topic = pick(row, ["focus_area"])

                data.append({
                    "question_id": row.get("question_id") or row.get("id") or str(idx),
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "topic": topic,
                })
        return data

    raise ValueError(f"Unsupported file format: {suffix}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_splits(path: Path, train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
    obj = {"seed": SEED, "train": train_ids, "val": val_ids, "test": test_ids}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def make_splits(ids: List[str]) -> Tuple[List[str], List[str], List[str]]:
    rng = random.Random(SEED)
    ids = ids[:]  # copy
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]
    # sanity
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    return train_ids, val_ids, test_ids


def build_corpus_from_answers(qa_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus = []
    for r in qa_rows:
        qid = r["question_id"]
        source = r.get("source", "nih").lower()

        corpus.append({
            "doc_id": f"{source}_ans_{qid}",
            "text": r["answer"],
            "source": source.upper(),
            "question_id": qid,
        })
    return corpus


def build_eval_pack(
    qa_rows: List[Dict[str, Any]],
    test_ids: List[str],
    eval_size: int = EVAL_SIZE,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Возьмём eval queries из test, чтобы избежать утечки.
    Возвращает: eval_queries_rows, qrels_lines
    """
    rng = random.Random(SEED)
    test_set = set(test_ids)

    candidates = [r for r in qa_rows if r["question_id"] in test_set]
    rng.shuffle(candidates)
    chosen = candidates[: min(eval_size, len(candidates))]

    eval_queries = [{"query_id": r["question_id"], "question": r["question"]} for r in chosen]

    # qrels: единственный релевантный doc — answer-doc этого вопроса
    qrels_lines = []
    for r in chosen:
        source = r.get("source", "nih").lower()
        doc_id = f"{source}_ans_{r['question_id']}"
        qrels_lines.append(f'{r["question_id"]}\t0\t{doc_id}\t1')
    
    return eval_queries, qrels_lines


def main():
    # === ПУТИ: поменяй под свой репо ===
    raw_nih_path = Path("data/raw/nih_qa.jsonl")   # <- подстрой
    out_dir = Path("data/processed")

    qa_out = out_dir / "qa.jsonl"
    corpus_out = out_dir / "corpus.jsonl"
    splits_out = out_dir / "splits.json"
    eval_queries_out = out_dir / "eval_queries.jsonl"
    qrels_out = out_dir / "qrels.tsv"

    qa_rows = read_nih_qa(raw_nih_path)

    # 1) qa.jsonl
    write_jsonl(qa_out, qa_rows)

    # 2) splits.json
    ids = [r["question_id"] for r in qa_rows]
    train_ids, val_ids, test_ids = make_splits(ids)
    write_splits(splits_out, train_ids, val_ids, test_ids)

    # 3) corpus.jsonl
    corpus_rows = build_corpus_from_answers(qa_rows)
    write_jsonl(corpus_out, corpus_rows)

    # 4) eval pack
    eval_queries, qrels_lines = build_eval_pack(qa_rows, test_ids, EVAL_SIZE)
    write_jsonl(eval_queries_out, eval_queries)
    qrels_out.parent.mkdir(parents=True, exist_ok=True)
    qrels_out.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")

    print("DONE")
    print(f"- {qa_out} ({len(qa_rows)} rows)")
    print(f"- {corpus_out} ({len(corpus_rows)} docs)")
    print(f"- {splits_out}")
    print(f"- {eval_queries_out} ({len(eval_queries)} queries)")
    print(f"- {qrels_out} ({len(qrels_lines)} lines)")


if __name__ == "__main__":
    main()