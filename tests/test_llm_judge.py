import io
import json
import urllib.error

import pytest

from medical_rag_reranker.evaluation import llm_judge as llm_judge_module
from medical_rag_reranker.evaluation.llm_judge import (
    JudgeResponseError,
    LocalOpenAICompatibleJudge,
    build_judge_messages,
    parse_judge_response,
)


def test_parse_judge_response_accepts_json_object() -> None:
    parsed = parse_judge_response(
        """
        {
          "faithfulness": 5,
          "relevance": 4,
          "completeness": 3,
          "safety": 5,
          "verdict": "pass",
          "rationale": "The answer is supported by the provided context."
        }
        """
    )

    assert parsed["faithfulness"] == 5.0
    assert parsed["relevance"] == 4.0
    assert parsed["verdict"] == "pass"
    assert "supported" in parsed["rationale"]


def test_parse_judge_response_accepts_fenced_json() -> None:
    parsed = parse_judge_response(
        """```json
        {"faithfulness": 1, "relevance": 2, "completeness": 1, "safety": 3,
         "verdict": "fail", "rationale": "unsupported"}
        ```"""
    )

    assert parsed["verdict"] == "fail"
    assert parsed["safety"] == 3.0


def test_parse_judge_response_rejects_malformed_payload() -> None:
    with pytest.raises(JudgeResponseError):
        parse_judge_response("not json")


def test_build_judge_messages_includes_context_and_answer() -> None:
    messages = build_judge_messages(
        {
            "question": "What is metformin used for?",
            "retrieved": [{"doc_id": "doc1", "text": "Metformin treats diabetes."}],
            "answer": "Metformin treats diabetes [doc1].",
            "reference_answer": "Metformin is used to treat type 2 diabetes.",
            "citations_detected": ["doc1"],
        }
    )

    joined = "\n".join(message["content"] for message in messages)
    assert "What is metformin used for?" in joined
    assert "Reference answer" in joined
    assert "type 2 diabetes" in joined
    assert "[doc1] Metformin treats diabetes." in joined
    assert "Metformin treats diabetes [doc1]." in joined


def test_local_judge_retries_retryable_http_errors(monkeypatch) -> None:
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "faithfulness": 5,
                                        "relevance": 4,
                                        "completeness": 3,
                                        "safety": 5,
                                        "verdict": "pass",
                                        "rationale": "Supported.",
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    calls = {"count": 0}

    def fake_urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url=req.full_url,
                code=500,
                msg="Internal Server Error",
                hdrs={},
                fp=io.BytesIO(b"temporary failure"),
            )
        return _Response()

    monkeypatch.setattr(llm_judge_module.urllib.request, "urlopen", fake_urlopen)

    judge = LocalOpenAICompatibleJudge(
        base_url="http://judge.test/v1",
        model="local-judge",
        max_retries=1,
        retry_backoff_seconds=0,
    )

    result = judge.evaluate(
        {
            "question": "What is metformin used for?",
            "answer": "Metformin treats diabetes [doc1].",
            "retrieved": [{"doc_id": "doc1", "text": "Metformin treats diabetes."}],
        }
    )

    assert calls["count"] == 2
    assert result["verdict"] == "pass"
    assert result["faithfulness"] == 5.0


def test_local_judge_can_send_single_user_message(monkeypatch) -> None:
    captured_payloads = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "faithfulness": 5,
                                        "relevance": 5,
                                        "completeness": 5,
                                        "safety": 5,
                                        "verdict": "pass",
                                        "rationale": "Supported.",
                                    }
                                )
                            }
                        }
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(req, timeout):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr(llm_judge_module.urllib.request, "urlopen", fake_urlopen)

    judge = LocalOpenAICompatibleJudge(
        base_url="http://judge.test/v1",
        model="local-judge",
        single_user_message=True,
    )
    result = judge.evaluate(
        {
            "question": "What is metformin used for?",
            "answer": "Metformin treats diabetes [doc1].",
            "retrieved": [{"doc_id": "doc1", "text": "Metformin treats diabetes."}],
        }
    )

    messages = captured_payloads[0]["messages"]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "strict medical RAG evaluation judge" in messages[0]["content"]
    assert result["verdict"] == "pass"


def test_local_judge_retries_unparsable_responses(monkeypatch) -> None:
    responses = [
        "I would rate this as mostly supported.",
        json.dumps(
            {
                "faithfulness": 5,
                "relevance": 5,
                "completeness": 4,
                "safety": 5,
                "verdict": "pass",
                "rationale": "Supported.",
            }
        ),
    ]
    captured_payloads = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def read(self):
            content = responses.pop(0)
            return json.dumps({"choices": [{"message": {"content": content}}]}).encode(
                "utf-8"
            )

    def fake_urlopen(req, timeout):
        captured_payloads.append(json.loads(req.data.decode("utf-8")))
        return _Response()

    monkeypatch.setattr(llm_judge_module.urllib.request, "urlopen", fake_urlopen)

    judge = LocalOpenAICompatibleJudge(
        base_url="http://judge.test/v1",
        model="local-judge",
        parse_max_retries=1,
        single_user_message=True,
    )

    result = judge.evaluate(
        {
            "question": "What is metformin used for?",
            "answer": "Metformin treats diabetes [doc1].",
            "retrieved": [{"doc_id": "doc1", "text": "Metformin treats diabetes."}],
        }
    )

    assert len(captured_payloads) == 2
    assert "Previous response excerpt" in captured_payloads[1]["messages"][0]["content"]
    assert result["verdict"] == "pass"
