"""Reference-free heuristics for scoring RAG generation quality.

The goal of this module is not to replace human evaluation. It provides a
stable, reproducible baseline that can be logged to MLflow and compared across
retriever/generator configurations without requiring reference answers.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]+\b")
INSUFFICIENT_PATTERNS = (
    "insufficient",
    "not enough context",
    "not enough information",
    "cannot answer",
    "can't answer",
    "do not know",
)


def _clip(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")}


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return _clip(len(left_tokens & right_tokens) / float(len(left_tokens)))


def _contains_insufficient_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    return any(pattern in normalized for pattern in INSUFFICIENT_PATTERNS)


def evaluate_generation_result(result: dict[str, Any]) -> dict[str, float]:
    """Score one generated answer with lightweight reference-free heuristics.

    Expected fields are aligned with `generate_from_cfg`: `question`, `retrieved`,
    `answer`, and citation-related keys.
    """

    question = str(result.get("question") or "")
    answer = str(result.get("answer") or "").strip()
    retrieved = list(result.get("retrieved") or [])
    context_text = "\n".join(str(doc.get("text") or "") for doc in retrieved)

    citations = list(result.get("citations_detected") or [])
    supported_citations = list(result.get("supported_citations_detected") or [])
    unsupported_citations = list(result.get("unsupported_citations_detected") or [])

    context_relevance = _overlap_score(question, context_text)
    answer_relevance = _overlap_score(question, answer)
    answer_context_overlap = _overlap_score(answer, context_text)

    citation_presence_rate = 1.0 if citations else 0.0
    supported_citation_rate = (
        len(supported_citations) / float(len(citations)) if citations else 0.0
    )
    unsupported_citation_rate = (
        len(unsupported_citations) / float(len(citations)) if citations else 0.0
    )

    empty_answer = 1.0 if not answer else 0.0
    insufficient_context = 1.0 if _contains_insufficient_answer(answer) else 0.0

    groundedness = _clip(
        0.55 * answer_context_overlap
        + 0.35 * supported_citation_rate
        + 0.10 * citation_presence_rate
        - 0.30 * unsupported_citation_rate
        - 0.25 * empty_answer
    )

    answer_relevance = _clip(
        0.80 * answer_relevance
        + 0.20 * context_relevance
        - 0.20 * insufficient_context
        - 0.25 * empty_answer
    )

    return {
        "context_relevance": float(context_relevance),
        "groundedness": float(groundedness),
        "answer_relevance": float(answer_relevance),
        "citation_presence_rate": float(citation_presence_rate),
        "supported_citation_rate": float(supported_citation_rate),
        "unsupported_citation_rate": float(unsupported_citation_rate),
        "empty_answer": float(empty_answer),
        "insufficient_context": float(insufficient_context),
        "num_retrieved_docs": float(len(retrieved)),
        "answer_chars": float(len(answer)),
        "retrieval_latency_ms": float(result.get("retrieval_latency_ms", 0.0)),
        "generation_latency_ms": float(result.get("generation_latency_ms", 0.0)),
        "rerank_latency_ms": float(result.get("rerank_latency_ms", 0.0)),
        "end_to_end_latency_ms": float(result.get("end_to_end_latency_ms", 0.0)),
    }


def summarize_generation_evaluations(
    rows: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate per-example generation evaluations into mean metrics."""
    if not rows:
        raise ValueError("Cannot summarize empty generation evaluation rows.")

    metrics = [
        row["evaluation"]
        if isinstance(row.get("evaluation"), dict)
        else evaluate_generation_result(row)
        for row in rows
    ]
    numeric_keys = sorted(
        {
            key
            for item in metrics
            for key, value in item.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    summary = {}
    for key in numeric_keys:
        values = [
            float(item[key])
            for item in metrics
            if isinstance(item.get(key), (int, float))
            and not isinstance(item.get(key), bool)
        ]
        if values:
            summary[f"avg_{key}"] = float(mean(values))

    verdicts = [
        str(item.get("verdict", "")).strip().lower()
        for item in metrics
        if item.get("verdict") is not None
    ]
    if verdicts:
        summary["pass_rate"] = float(
            mean(1.0 if verdict == "pass" else 0.0 for verdict in verdicts)
        )
        summary["fail_rate"] = float(
            mean(1.0 if verdict == "fail" else 0.0 for verdict in verdicts)
        )

    summary["num_examples"] = float(len(rows))
    summary["reranker_enabled_rate"] = float(
        mean(1.0 if bool(row.get("reranker_enabled")) else 0.0 for row in rows)
    )
    return summary
