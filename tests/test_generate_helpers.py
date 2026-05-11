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


class PayloadGenerator:
    def generate(self, prompt: str) -> str:
        assert "[doc2]" in prompt
        assert "Metformin is used for treatment" in prompt
        return "Metformin is used for type 2 diabetes [doc2]"


class PayloadRetriever:
    def retrieve(self, question: str, top_k: int):
        assert question == "What is metformin used for?"
        assert top_k == 1
        self.last_payloads = {
            "doc2": {
                "doc_id": "doc2",
                "text": "Metformin is used for treatment of type 2 diabetes.",
                "source": "REMOTE",
            }
        }
        return [SimpleNamespace(doc_id="doc2", score=0.9)]


def test_query_helpers_support_text_and_question_keys() -> None:
    assert generation_module._resolve_query_text({"text": "hello"}) == "hello"
    assert generation_module._resolve_query_text({"question": "world"}) == "world"
    assert generation_module._resolve_query_id({"query_id": "q1"}, 1) == "q1"
    assert generation_module._resolve_query_id({"question_id": "q2"}, 1) == "q2"


def test_load_queries_none_limit_reads_all_rows(tmp_path) -> None:
    path = tmp_path / "queries.jsonl"
    path.write_text('{"query_id":"q1"}\n{"query_id":"q2"}\n', encoding="utf-8")

    assert len(generation_module._load_queries(path, limit=None)) == 2
    assert len(generation_module._load_queries(path, limit=1)) == 1


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


def test_run_one_question_can_use_retriever_payload_docstore() -> None:
    result = generation_module._run_one_question(
        llm=PayloadGenerator(),
        retriever=PayloadRetriever(),
        docstore={},
        question="What is metformin used for?",
        top_k=1,
        retrieve_top_k=1,
        reranker=None,
        query_id="q2",
    )

    assert result["retrieved"][0]["doc_id"] == "doc2"
    assert result["retrieved"][0]["text"].startswith("Metformin is used")
    assert result["retrieved"][0]["source"] == "REMOTE"


def test_remote_openai_compatible_generator_uses_chat_endpoint(monkeypatch) -> None:
    generator = generation_module.RemoteOpenAICompatibleGenerator(
        base_url="https://example.test/v1",
        model_name="test-model",
        max_new_tokens=32,
        do_sample=False,
        temperature=0.7,
    )
    captured = {}

    def fake_request(endpoint, payload):
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr(generator, "_request", fake_request)

    assert generator.generate("prompt") == "answer"
    assert captured["endpoint"] == "/chat/completions"
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["messages"][0]["content"] == "prompt"
    assert captured["payload"]["temperature"] == 0.0
