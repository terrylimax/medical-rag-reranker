from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from medical_rag_reranker.graph.aspects import aspects_from_metadata, detect_aspects
from medical_rag_reranker.graph.builder import normalize_key

from . import Retriever, ScoredDoc


DEFAULT_RELATION_WEIGHTS: dict[str, float] = {
    "same_document_id": 1.0,
    "same_focus": 1.0,
    "same_cui": 0.9,
    "focus_synonym_match": 0.7,
    "same_semantic_group": 0.075,
    "question_type_boost": 0.2,
}


def _minmax_map(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    ids = list(values)
    raw = [float(values[i]) for i in ids]
    mn, mx = min(raw), max(raw)
    if mx - mn < 1e-9:
        return {doc_id: 1.0 for doc_id in ids}
    return {doc_id: (score - mn) / (mx - mn) for doc_id, score in zip(ids, raw)}


@dataclass
class GraphExpandedRetriever(Retriever):
    """Retriever wrapper that expands seed hits through MedQuAD metadata graph."""

    base: Retriever
    graph: dict[str, Any]
    seed_k: int = 20
    expand_k: int = 50
    max_hops: int = 2
    base_weight: float = 0.7
    graph_weight: float = 0.3
    hop_decay: float = 0.65
    relation_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_RELATION_WEIGHTS)
    )
    last_payloads: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        if int(top_k) <= 0:
            return []

        seed_limit = max(int(self.seed_k), int(top_k))
        base_hits = self.base.retrieve(query, top_k=seed_limit)
        if not base_hits:
            return []

        base_scores = {hit.doc_id: float(hit.score) for hit in base_hits}
        base_norm = _minmax_map(base_scores)
        if len(base_hits) == 1:
            base_norm[base_hits[0].doc_id] = 1.0

        graph_scores: dict[str, float] = {}
        seeds = base_hits[: max(1, int(self.seed_k))]
        query_aspects = detect_aspects(query)
        query_normalized = normalize_key(query)

        first_hop_seen: dict[str, float] = {}
        for rank, hit in enumerate(seeds, start=1):
            rank_weight = 1.0 / float(rank)
            seed_weight = max(base_norm.get(hit.doc_id, 0.0), rank_weight)
            expanded = self._expand_doc(
                doc_id=hit.doc_id,
                query_normalized=query_normalized,
                seed_weight=seed_weight * rank_weight,
                decay=1.0,
                graph_scores=graph_scores,
            )
            for doc_id, value in expanded.items():
                first_hop_seen[doc_id] = max(first_hop_seen.get(doc_id, 0.0), value)

        if int(self.max_hops) >= 2 and first_hop_seen:
            hop2_sources = sorted(
                first_hop_seen.items(), key=lambda item: item[1], reverse=True
            )[: max(1, int(self.expand_k))]
            for doc_id, source_weight in hop2_sources:
                self._expand_doc(
                    doc_id=doc_id,
                    query_normalized=query_normalized,
                    seed_weight=source_weight,
                    decay=float(self.hop_decay),
                    graph_scores=graph_scores,
                    include_weak_edges=False,
                )

        candidate_ids = set(base_scores)
        if graph_scores:
            top_graph = sorted(
                graph_scores.items(), key=lambda item: item[1], reverse=True
            )[: max(1, int(self.expand_k))]
            graph_scores = dict(top_graph)
            candidate_ids.update(graph_scores)

        if query_aspects:
            for doc_id in list(candidate_ids):
                doc_aspects = aspects_from_metadata(self._doc_meta(doc_id))
                if doc_aspects & query_aspects:
                    graph_scores[doc_id] = graph_scores.get(doc_id, 0.0) + float(
                        self.relation_weights.get("question_type_boost", 0.2)
                    )

        graph_norm = _minmax_map(graph_scores)
        scored: list[ScoredDoc] = []
        for doc_id in candidate_ids:
            base_component = base_norm.get(doc_id, 0.0)
            graph_component = graph_norm.get(doc_id, 0.0)
            final_score = (
                float(self.base_weight) * base_component
                + float(self.graph_weight) * graph_component
            )
            scored.append(ScoredDoc(doc_id=doc_id, score=float(final_score)))

        scored.sort(key=lambda item: item.score, reverse=True)
        final = scored[: int(top_k)]
        self._sync_payloads([item.doc_id for item in final])
        return final

    def _sync_payloads(self, doc_ids: list[str]) -> None:
        payloads = dict(getattr(self.base, "last_payloads", {}) or {})
        missing = [doc_id for doc_id in doc_ids if doc_id not in payloads]
        fetch_payloads = getattr(self.base, "fetch_payloads", None)
        if missing and callable(fetch_payloads):
            payloads.update(fetch_payloads(missing))
        self.last_payloads = {
            doc_id: payloads[doc_id] for doc_id in doc_ids if doc_id in payloads
        }

    def _doc_meta(self, doc_id: str) -> dict[str, Any]:
        docs = self.graph.get("docs", {})
        meta = docs.get(doc_id, {})
        return meta if isinstance(meta, dict) else {}

    def _index_docs(self, index_name: str, value: object) -> set[str]:
        key = normalize_key(value)
        if not key:
            return set()
        bucket = self.graph.get("indexes", {}).get(index_name, {})
        values = bucket.get(key, [])
        return set(str(v) for v in values)

    def _expand_doc(
        self,
        doc_id: str,
        query_normalized: str,
        seed_weight: float,
        decay: float,
        graph_scores: dict[str, float],
        include_weak_edges: bool = True,
    ) -> dict[str, float]:
        meta = self._doc_meta(doc_id)
        if not meta:
            return {}

        relation_docs: dict[str, float] = {}

        self._collect_relation(
            relation_docs,
            "document_id",
            meta.get("document_id"),
            self.relation_weights.get("same_document_id", 1.0),
        )
        self._collect_relation(
            relation_docs,
            "focus",
            meta.get("question_focus"),
            self.relation_weights.get("same_focus", 1.0),
        )
        for cui in meta.get("umls_cui", []) or []:
            self._collect_relation(
                relation_docs,
                "cui",
                cui,
                self.relation_weights.get("same_cui", 0.9),
            )

        if include_weak_edges:
            for group in meta.get("umls_semantic_group", []) or []:
                self._collect_relation(
                    relation_docs,
                    "semantic_group",
                    group,
                    self.relation_weights.get("same_semantic_group", 0.075),
                )

        synonym_weight = self.relation_weights.get("focus_synonym_match", 0.7)
        for term in [meta.get("question_focus"), *(meta.get("synonyms", []) or [])]:
            term_key = normalize_key(term)
            if len(term_key) >= 3 and term_key in query_normalized:
                self._collect_relation(relation_docs, "synonym", term, synonym_weight)

        touched: dict[str, float] = {}
        for candidate_id, relation_weight in relation_docs.items():
            if candidate_id == doc_id:
                continue
            increment = float(seed_weight) * float(relation_weight) * float(decay)
            graph_scores[candidate_id] = graph_scores.get(candidate_id, 0.0) + increment
            touched[candidate_id] = max(touched.get(candidate_id, 0.0), increment)
        return touched

    def _collect_relation(
        self,
        relation_docs: dict[str, float],
        index_name: str,
        value: object,
        weight: float,
    ) -> None:
        if not value:
            return
        for candidate_id in self._index_docs(index_name, value):
            relation_docs[candidate_id] = max(
                relation_docs.get(candidate_id, 0.0), float(weight)
            )
