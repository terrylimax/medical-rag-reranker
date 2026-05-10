from medical_rag_reranker.retrieval import ScoredDoc
from medical_rag_reranker.retrieval.graph_expanded import GraphExpandedRetriever


class _GraphBaseRetriever:
    last_payloads = {
        "doc-a": {"doc_id": "doc-a", "text": "base payload"},
    }

    def retrieve(self, query: str, top_k: int) -> list[ScoredDoc]:
        return [ScoredDoc("doc-a", 1.0)]

    def fetch_payloads(self, doc_ids: list[str]):
        return {
            doc_id: {"doc_id": doc_id, "text": f"fetched {doc_id}"}
            for doc_id in doc_ids
        }


def test_graph_expansion_fetches_payloads_for_expanded_docs() -> None:
    graph = {
        "docs": {
            "doc-a": {"question_focus": "metformin"},
            "doc-b": {"question_focus": "metformin"},
        },
        "indexes": {
            "document_id": {},
            "focus": {"metformin": ["doc-a", "doc-b"]},
            "cui": {},
            "semantic_group": {},
            "synonym": {},
        },
    }
    retriever = GraphExpandedRetriever(
        base=_GraphBaseRetriever(),
        graph=graph,
        seed_k=1,
        expand_k=2,
        max_hops=1,
    )

    hits = retriever.retrieve("metformin", top_k=2)

    assert {hit.doc_id for hit in hits} == {"doc-a", "doc-b"}
    assert retriever.last_payloads["doc-a"]["text"] == "base payload"
    assert retriever.last_payloads["doc-b"]["text"] == "fetched doc-b"
