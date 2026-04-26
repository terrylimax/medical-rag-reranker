from dataclasses import dataclass

from medical_rag_reranker.retrieval import ScoredDoc
from medical_rag_reranker.retrieval.hybrid import HybridRetriever


@dataclass
class _DummyRetriever:
    docs: list[ScoredDoc]

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        return self.docs[:top_k]


def test_hybrid_rrf_prefers_documents_supported_by_both_retrievers() -> None:
    bm25 = _DummyRetriever(
        [
            ScoredDoc("doc-1", 10.0),
            ScoredDoc("doc-2", 9.0),
        ]
    )
    dense = _DummyRetriever(
        [
            ScoredDoc("doc-2", 0.9),
            ScoredDoc("doc-3", 0.8),
        ]
    )

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        fusion="rrf",
        cand_k=2,
        rrf_k=10,
    )

    results = hybrid.retrieve("heart disease", top_k=3)

    assert [doc.doc_id for doc in results] == ["doc-2", "doc-1", "doc-3"]
