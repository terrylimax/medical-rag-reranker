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

import csv
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

from medical_rag_reranker.data.metadata import extract_medical_metadata
from medical_rag_reranker.graph.builder import build_medquad_graph
from medical_rag_reranker.utils.progress import count_text_lines, progress, timed_stage


SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EVAL_SIZE = 300  # набор запросов для промежуточных прогонов

GRAPH_METADATA_FIELDS = [
    "document_id",
    "document_source",
    "document_url",
    "question_focus",
    "question_type",
    "umls_cui",
    "umls_semantic_types",
    "umls_semantic_group",
    "synonyms",
]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(value != value)
    except Exception:
        return False


def pick_value(row: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in row and not is_missing(row[key]):
            return row[key]
    return None


def normalize_metadata_value(value: Any) -> Any:
    if is_missing(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for item in value:
            if is_missing(item):
                continue
            text = str(item).strip()
            if text:
                cleaned.append(text)
        return cleaned or None
    return str(value).strip()


def build_qa_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question_id = pick_value(row, ["question_id", "id"]) or str(idx)
    question_focus = pick_value(row, ["question_focus", "focus_area", "topic"])
    source = pick_value(row, ["source"])
    document_source = pick_value(row, ["document_source"])

    out: Dict[str, Any] = {
        "question_id": str(question_id),
        "original_question_id": str(question_id),
        "base_question_id": str(question_id),
        "question": pick_value(row, ["question", "q"]),
        "answer": pick_value(row, ["answer", "a"]),
        "source": source or document_source or "MedQuAD",
        "topic": question_focus,
        "document_id": pick_value(row, ["document_id", "doc_id"]),
        "document_source": document_source,
        "document_url": pick_value(row, ["document_url", "url"]),
        "question_focus": question_focus,
        "question_type": pick_value(row, ["question_type", "qtype", "type"]),
        "umls_cui": pick_value(row, ["umls_cui", "cui", "cuis"]),
        "umls_semantic_types": pick_value(
            row, ["umls_semantic_types", "semantic_types"]
        ),
        "umls_semantic_group": pick_value(
            row, ["umls_semantic_group", "semantic_group"]
        ),
        "synonyms": pick_value(row, ["synonyms", "synonym"]),
    }

    normalized: Dict[str, Any] = {}
    for key, value in out.items():
        if key in ("question", "answer"):
            normalized[key] = value
        else:
            normalized[key] = normalize_metadata_value(value)
    return normalized


def safe_doc_source_slug(source: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(source or "nih").strip().lower())
    return slug.strip("_") or "nih"


def read_nih_qa(path: Path) -> List[Dict[str, Any]]:
    """
    Читает QA-данные из JSONL или CSV(MedQuAD)
    Возвращает список dict с полями question_id, question, answer, source, topic.
    """
    data: List[Dict[str, Any]] = []

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        total = count_text_lines(path)
        with path.open("r", encoding="utf-8") as f:
            rows = progress(
                f,
                desc="Reading raw QA JSONL",
                total=total,
                unit="row",
            )
            for idx, line in enumerate(rows):
                obj = json.loads(line)
                data.append(build_qa_row(obj, idx))
        return data

    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = progress(reader, desc="Reading raw QA CSV", unit="row")
            for idx, row in enumerate(rows):
                data.append(build_qa_row(dict(row), idx))
        return data

    if suffix == ".parquet":
        # Keep this path dependency-light by using datasets parquet loader.
        from datasets import load_dataset

        ds = load_dataset("parquet", data_files=str(path), split="train")
        rows = progress(
            ds,
            desc="Reading raw QA parquet",
            total=len(ds),
            unit="row",
        )
        for idx, row in enumerate(rows):
            data.append(build_qa_row(dict(row), idx))
        return data

    raise ValueError(f"Unsupported file format: {suffix}")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in progress(
            rows, desc=f"Writing {path.name}", total=len(rows), unit="row"
        ):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_splits(
    path: Path, train_ids: List[str], val_ids: List[str], test_ids: List[str]
) -> None:
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
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]
    # sanity
    assert len(set(train_ids) & set(val_ids)) == 0
    assert len(set(train_ids) & set(test_ids)) == 0
    assert len(set(val_ids) & set(test_ids)) == 0
    return train_ids, val_ids, test_ids


def build_corpus_from_answers(qa_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    corpus = []
    for r in progress(
        qa_rows,
        desc="Building answer corpus",
        total=len(qa_rows),
        unit="row",
    ):
        qid = r["question_id"]
        source = safe_doc_source_slug(r.get("source") or "nih")
        doc = {
            "doc_id": f"{source}_ans_{qid}",
            "text": r["answer"],
            "source": source.upper(),
            "question_id": qid,
            "group_id": r.get("group_id"),
            "question_intent": r.get("question_intent"),
            "diagnosis_or_topic": r.get("diagnosis_or_topic"),
        }

        for field in [
            "original_question_id",
            "base_question_id",
            "topic",
            *GRAPH_METADATA_FIELDS,
        ]:
            value = r.get(field)
            if value not in (None, "", []):
                doc[field] = value

        corpus.append(doc)
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

    eval_queries = []

    # qrels: единственный релевантный doc — answer-doc этого вопроса
    qrels_lines = []
    for r in chosen:
        source = r.get("source") or "nih"
        doc_id = f"{safe_doc_source_slug(source)}_ans_{r['question_id']}"
        query = {
            "query_id": r["question_id"],
            "question": r["question"],
            "question_id": r["question_id"],
            "gold_doc_id": doc_id,
        }
        for field in ["question_focus", "question_type", "umls_cui"]:
            value = r.get(field)
            if value not in (None, "", []):
                query[field] = value
        eval_queries.append(query)
        qrels_lines.append(f'{r["question_id"]}\t0\t{doc_id}\t1')

    return eval_queries, qrels_lines


def filter_valid_qa_rows(qa_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep only rows with non-empty question and answer text."""
    cleaned: List[Dict[str, Any]] = []
    for row in progress(
        qa_rows,
        desc="Filtering QA rows",
        total=len(qa_rows),
        unit="row",
    ):
        question = str(row.get("question") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if not question or not answer:
            continue
        copied = dict(row)
        copied["question"] = question
        copied["answer"] = answer
        cleaned.append(copied)
    return cleaned


def ensure_unique_question_ids(qa_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Ensure question_id is unique and non-empty across rows.

    Some sources (e.g. MedQuAD parquet) may reuse `id` values.
    We keep the original id when possible and suffix duplicates deterministically.
    """
    seen: set[str] = set()
    duplicate_counters: dict[str, int] = {}
    normalized: List[Dict[str, Any]] = []

    rows = progress(
        qa_rows,
        desc="Normalizing question ids",
        total=len(qa_rows),
        unit="row",
    )
    for idx, row in enumerate(rows):
        copied = dict(row)
        base = str(copied.get("question_id") or "").strip() or f"q_{idx}"
        qid = base

        if qid in seen:
            next_idx = duplicate_counters.get(base, 1)
            while True:
                candidate = f"{base}__dup{next_idx}"
                next_idx += 1
                if candidate not in seen:
                    qid = candidate
                    duplicate_counters[base] = next_idx
                    break

        seen.add(qid)
        copied.setdefault("original_question_id", base)
        copied.setdefault("base_question_id", base)
        copied["question_id"] = qid
        copied.update(extract_medical_metadata(copied))
        normalized.append(copied)

    return normalized


def prepare_data(
    raw_nih_path: str,
    out_dir: str,
    eval_size: int = EVAL_SIZE,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Build baseline retrieval artifacts from NIH QA data."""
    global SEED
    prev_seed = SEED
    SEED = int(seed)

    raw_path = Path(raw_nih_path)
    out_root = Path(out_dir)
    qa_out = out_root / "qa.jsonl"
    corpus_out = out_root / "corpus.jsonl"
    splits_out = out_root / "splits.json"
    eval_queries_out = out_root / "eval_queries.jsonl"
    qrels_out = out_root / "qrels.tsv"
    graph_out = out_root / "medquad_graph.json"

    try:
        with timed_stage("Prepare QA artifacts"):
            qa_rows = ensure_unique_question_ids(
                filter_valid_qa_rows(read_nih_qa(raw_path))
            )

        # 1) qa.jsonl
        write_jsonl(qa_out, qa_rows)

        # 2) splits.json
        ids = [r["question_id"] for r in qa_rows]
        train_ids, val_ids, test_ids = make_splits(ids)
        write_splits(splits_out, train_ids, val_ids, test_ids)

        # 3) corpus.jsonl
        corpus_rows = build_corpus_from_answers(qa_rows)
        write_jsonl(corpus_out, corpus_rows)

        # 4) graph artifact from preserved MedQuAD metadata
        build_medquad_graph(corpus_out, graph_out)

        # 5) eval pack
        eval_queries, qrels_lines = build_eval_pack(qa_rows, test_ids, int(eval_size))
        write_jsonl(eval_queries_out, eval_queries)
        qrels_out.parent.mkdir(parents=True, exist_ok=True)
        qrels_out.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")
    finally:
        SEED = prev_seed

    return {
        "qa_path": str(qa_out),
        "corpus_path": str(corpus_out),
        "graph_path": str(graph_out),
        "splits_path": str(splits_out),
        "eval_queries_path": str(eval_queries_out),
        "qrels_path": str(qrels_out),
        "num_qa_rows": len(qa_rows),
        "num_corpus_docs": len(corpus_rows),
        "num_eval_queries": len(eval_queries),
        "num_qrels": len(qrels_lines),
        "seed": int(seed),
    }


def main():
    # === ПУТИ: поменяй под свой репо ===
    result = prepare_data(
        raw_nih_path="data/raw/nih_qa.jsonl",
        out_dir="data/processed",
        eval_size=EVAL_SIZE,
        seed=SEED,
    )

    print("DONE")
    print(f"- {result['qa_path']} ({result['num_qa_rows']} rows)")
    print(f"- {result['corpus_path']} ({result['num_corpus_docs']} docs)")
    print(f"- {result['splits_path']}")
    print(f"- {result['eval_queries_path']} " f"({result['num_eval_queries']} queries)")
    print(f"- {result['qrels_path']} ({result['num_qrels']} lines)")


if __name__ == "__main__":
    main()
