from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Any

from medical_rag_reranker.data.metadata import (
    infer_diagnosis_or_topic,
    infer_question_intent,
)

from . import Retriever, ScoredDoc


_INTENT_VARIANTS: dict[str, tuple[str, ...]] = {
    "symptoms": (
        "{topic} symptoms signs clinical features",
        "clinical presentation of {topic}",
    ),
    "treatment": (
        "{topic} treatment therapy management",
        "medical management of {topic}",
    ),
    "diagnosis": (
        "{topic} diagnosis testing screening",
        "how {topic} is diagnosed",
    ),
    "causes": (
        "{topic} causes risk factors etiology",
        "why {topic} occurs",
    ),
    "prevention": (
        "{topic} prevention risk reduction",
        "how to prevent {topic}",
    ),
    "inheritance": (
        "{topic} inheritance genetic hereditary",
        "genetic cause of {topic}",
    ),
    "frequency": (
        "{topic} prevalence frequency epidemiology",
        "how common is {topic}",
    ),
    "other": (
        "{topic} medical overview",
        "{topic} definition disease information",
    ),
}


def _normalize_query(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        cleaned = _normalize_query(item).strip(" ?.").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _compact_question_keywords(query: str) -> str:
    text = _normalize_query(query).strip(" ?.")
    text = re.sub(
        r"^(what|which|how|when|where|why)\s+(are|is|can|do|does|should|would)\s+",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^(what|which|how|when|where|why)\s+", "", text, flags=re.I)
    return text


def build_medical_query_variants(
    query: str,
    *,
    max_queries: int = 5,
    include_original: bool = True,
) -> list[str]:
    """Build deterministic medical query variants for query-side RAG Fusion.

    This is intentionally local and reproducible: it avoids adding an LLM call
    into retrieval evaluation, while still testing the core RAG Fusion mechanics.
    """
    original = _normalize_query(query)
    topic = infer_diagnosis_or_topic(original)
    intent = infer_question_intent(original)

    variants: list[str] = []
    if include_original:
        variants.append(original)

    compact = _compact_question_keywords(original)
    if compact and compact.lower() != original.lower():
        variants.append(compact)

    if topic:
        topic = _normalize_query(topic).strip(" ?.")
        templates = _INTENT_VARIANTS.get(intent) or _INTENT_VARIANTS["other"]
        variants.extend(template.format(topic=topic) for template in templates)
        variants.append(topic)

    if not topic:
        variants.append(f"medical {compact or original}")

    return _dedupe(variants)[: max(1, int(max_queries))]


@dataclass
class RagFusionRetriever(Retriever):
    base: Retriever
    num_queries: int = 5
    cand_k: int = 50
    rrf_k: int = 60
    include_original: bool = True
    last_queries: list[str] = field(default_factory=list, init=False)
    last_payloads: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def expand_query(self, query: str) -> list[str]:
        return build_medical_query_variants(
            query,
            max_queries=int(self.num_queries),
            include_original=bool(self.include_original),
        )

    def retrieve(self, query: str, top_k: int) -> List[ScoredDoc]:
        if int(top_k) <= 0:
            return []

        variants = self.expand_query(query)
        self.last_queries = variants
        self.last_payloads = {}
        scores: dict[str, float] = {}
        candidate_k = max(int(top_k), int(self.cand_k))

        for variant in variants:
            hits = self.base.retrieve(variant, top_k=candidate_k)
            self.last_payloads.update(getattr(self.base, "last_payloads", {}) or {})
            for rank, doc in enumerate(hits, start=1):
                scores[doc.doc_id] = scores.get(doc.doc_id, 0.0) + (
                    1.0 / (float(self.rrf_k) + float(rank))
                )

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[
            : int(top_k)
        ]
        final_doc_ids = [doc_id for doc_id, _ in ranked]
        missing = [
            doc_id for doc_id in final_doc_ids if doc_id not in self.last_payloads
        ]
        fetch_payloads = getattr(self.base, "fetch_payloads", None)
        if missing and callable(fetch_payloads):
            self.last_payloads.update(fetch_payloads(missing))
        return [ScoredDoc(doc_id, float(score)) for doc_id, score in ranked]
