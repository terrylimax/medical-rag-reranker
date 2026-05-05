from types import SimpleNamespace

from medical_rag_reranker.inference import generate as generation_module


class DummyRetriever:
    def retrieve(self, question: str, top_k: int):
        assert question == "What is metformin used for?"
        assert top_k == 1
        return [SimpleNamespace(doc_id="doc1", score=4.2)]


class DummyGenerator:
    def generate(self, prompt: str) -> str:
        assert "[doc1]" in prompt
        return "Metformin is used for type 2 diabetes [doc1]"


def test_query_helpers_support_text_and_question_keys() -> None:
    assert generation_module._resolve_query_text({"text": "hello"}) == "hello"
    assert generation_module._resolve_query_text({"question": "world"}) == "world"
    assert generation_module._resolve_query_id({"query_id": "q1"}, 1) == "q1"
    assert generation_module._resolve_query_id({"question_id": "q2"}, 1) == "q2"


def test_run_one_question_tracks_latency_and_supported_citations() -> None:
    docstore = {
        "doc1": {
            "doc_id": "doc1",
            "text": "Metformin is used for treatment of type 2 diabetes.",
            "source": "TEST",
        }
    }

    result = generation_module._run_one_question(
        llm=DummyGenerator(),
        retriever=DummyRetriever(),
        docstore=docstore,
        question="What is metformin used for?",
        top_k=1,
        retrieve_top_k=1,
        reranker=None,
        query_id="q1",
    )

    assert result["query_id"] == "q1"
    assert result["citations_detected"] == ["doc1"]
    assert result["supported_citations_detected"] == ["doc1"]
    assert result["unsupported_citations_detected"] == []
    assert result["retrieval_latency_ms"] >= 0.0
    assert result["generation_latency_ms"] >= 0.0
    assert result["end_to_end_latency_ms"] >= 0.0
