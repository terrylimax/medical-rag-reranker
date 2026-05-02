from dataclasses import dataclass

from medical_rag_reranker.retrieval import ScoredDoc
from medical_rag_reranker.retrieval.rag_fusion import (
    RagFusionRetriever,
    build_medical_query_variants,
)


@dataclass
class _VariantRetriever:
    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        if "clinical features" in query:
            return [ScoredDoc("doc-symptoms", 1.0), ScoredDoc("doc-overview", 0.5)][
                :top_k
            ]
        return [ScoredDoc("doc-overview", 1.0), ScoredDoc("doc-symptoms", 0.5)][:top_k]


def test_build_medical_query_variants_extracts_topic_and_intent() -> None:
    variants = build_medical_query_variants(
        "What are the symptoms of X-linked lymphoproliferative syndrome?",
        max_queries=5,
    )

    assert variants[0] == (
        "What are the symptoms of X-linked lymphoproliferative syndrome"
    )
    assert (
        "X-linked lymphoproliferative syndrome symptoms signs clinical features"
        in variants
    )
    assert "X-linked lymphoproliferative syndrome" in variants


def test_rag_fusion_merges_variant_results_with_rrf() -> None:
    retriever = RagFusionRetriever(
        base=_VariantRetriever(),
        num_queries=5,
        cand_k=2,
        rrf_k=10,
    )

    results = retriever.retrieve(
        "What are the symptoms of X-linked lymphoproliferative syndrome?",
        top_k=2,
    )

    assert {doc.doc_id for doc in results} == {"doc-symptoms", "doc-overview"}
    assert any("clinical features" in query for query in retriever.last_queries)
