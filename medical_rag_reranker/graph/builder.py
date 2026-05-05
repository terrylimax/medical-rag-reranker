from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


GRAPH_FORMAT = "medical-rag-reranker.medquad-graph"
GRAPH_VERSION = 1


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(value != value)
    except Exception:
        return False


def normalize_key(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def coerce_list(value: object) -> list[str]:
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    values = parsed
                else:
                    values = [parsed]
            except json.JSONDecodeError:
                values = re.split(r"[|;]", text)
        else:
            values = re.split(r"[|;]", text)

    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip().strip("'\"")
        if not text:
            continue
        key = normalize_key(text)
        if key not in seen:
            out.append(text)
            seen.add(key)
    return out


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if _is_missing(value):
            continue
        if isinstance(value, (list, tuple, set)):
            values = coerce_list(value)
            return values[0] if values else None
        text = str(value).strip()
        if text:
            return text
    return None


def _add_index(
    indexes: dict[str, dict[str, list[str]]],
    index_name: str,
    value: object,
    doc_id: str,
) -> None:
    values = coerce_list(value)
    if not values:
        return
    bucket = indexes.setdefault(index_name, {})
    for item in values:
        key = normalize_key(item)
        if not key:
            continue
        docs = bucket.setdefault(key, [])
        if doc_id not in docs:
            docs.append(doc_id)


def build_medquad_graph(
    corpus_path: str | Path, out_path: str | Path
) -> dict[str, Any]:
    """Build a compact metadata graph from processed MedQuAD corpus JSONL."""
    corpus = Path(corpus_path)
    out = Path(out_path)

    docs: dict[str, dict[str, Any]] = {}
    indexes: dict[str, dict[str, list[str]]] = {
        "document_id": {},
        "focus": {},
        "cui": {},
        "question_type": {},
        "semantic_group": {},
        "synonym": {},
    }

    with corpus.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("doc_id") or "").strip()
            if not doc_id:
                continue

            document_id = _first_text(row, ("document_id", "base_question_id"))
            focus = _first_text(row, ("question_focus", "focus_area", "topic"))
            question_type = _first_text(row, ("question_type",))
            semantic_group = coerce_list(row.get("umls_semantic_group"))
            cuis = coerce_list(row.get("umls_cui"))
            synonyms = coerce_list(row.get("synonyms"))

            meta = {
                "document_id": document_id,
                "document_source": _first_text(row, ("document_source", "source")),
                "document_url": _first_text(row, ("document_url", "url")),
                "question_focus": focus,
                "question_type": question_type,
                "umls_cui": cuis,
                "umls_semantic_types": coerce_list(row.get("umls_semantic_types")),
                "umls_semantic_group": semantic_group,
                "synonyms": synonyms,
            }
            docs[doc_id] = {k: v for k, v in meta.items() if v not in (None, [], "")}

            _add_index(indexes, "document_id", document_id, doc_id)
            _add_index(indexes, "focus", focus, doc_id)
            _add_index(indexes, "cui", cuis, doc_id)
            _add_index(indexes, "question_type", question_type, doc_id)
            _add_index(indexes, "semantic_group", semantic_group, doc_id)
            _add_index(indexes, "synonym", synonyms, doc_id)
            _add_index(indexes, "synonym", focus, doc_id)

    payload: dict[str, Any] = {
        "format": GRAPH_FORMAT,
        "version": GRAPH_VERSION,
        "corpus_path": str(corpus),
        "doc_count": len(docs),
        "docs": docs,
        "indexes": indexes,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def load_medquad_graph(path: str | Path) -> dict[str, Any]:
    graph_path = Path(path)
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if payload.get("format") != GRAPH_FORMAT:
        raise ValueError(f"Unsupported graph artifact format: {payload.get('format')}")
    return payload
