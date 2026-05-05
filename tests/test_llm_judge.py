import pytest

from medical_rag_reranker.evaluation.llm_judge import (
    JudgeResponseError,
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
            "citations_detected": ["doc1"],
        }
    )

    joined = "\n".join(message["content"] for message in messages)
    assert "What is metformin used for?" in joined
    assert "[doc1] Metformin treats diabetes." in joined
    assert "Metformin treats diabetes [doc1]." in joined
