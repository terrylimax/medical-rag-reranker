from medical_rag_reranker.evaluation.reference_free import (
    evaluate_generation_result,
    summarize_generation_evaluations,
)


def test_evaluate_generation_result_penalizes_unsupported_citations() -> None:
    row = {
        "question": "What is metformin used for?",
        "answer": "Metformin is used for type 2 diabetes [doc-good] [doc-bad]",
        "retrieved": [
            {
                "doc_id": "doc-good",
                "text": "Metformin is commonly used to improve glycemic control in type 2 diabetes.",
                "score": 1.0,
            }
        ],
        "citations_detected": ["doc-good", "doc-bad"],
        "supported_citations_detected": ["doc-good"],
        "unsupported_citations_detected": ["doc-bad"],
        "retrieval_latency_ms": 3.0,
        "generation_latency_ms": 8.0,
        "end_to_end_latency_ms": 12.0,
    }

    metrics = evaluate_generation_result(row)

    assert metrics["context_relevance"] > 0.0
    assert metrics["answer_relevance"] > 0.0
    assert metrics["supported_citation_rate"] == 0.5
    assert metrics["unsupported_citation_rate"] == 0.5
    assert metrics["groundedness"] < 1.0


def test_summarize_generation_evaluations_aggregates_mean_values() -> None:
    rows = [
        {
            "question": "What is metformin used for?",
            "answer": "It is used for type 2 diabetes [doc1]",
            "retrieved": [
                {"doc_id": "doc1", "text": "Metformin treats type 2 diabetes."}
            ],
            "citations_detected": ["doc1"],
            "supported_citations_detected": ["doc1"],
            "unsupported_citations_detected": [],
            "reranker_enabled": False,
            "retrieval_latency_ms": 1.0,
            "generation_latency_ms": 2.0,
            "end_to_end_latency_ms": 4.0,
        },
        {
            "question": "What causes migraines?",
            "answer": "The provided context is insufficient.",
            "retrieved": [
                {"doc_id": "doc2", "text": "Migraines have multiple triggers."}
            ],
            "citations_detected": [],
            "supported_citations_detected": [],
            "unsupported_citations_detected": [],
            "reranker_enabled": True,
            "retrieval_latency_ms": 2.0,
            "generation_latency_ms": 4.0,
            "rerank_latency_ms": 5.0,
            "end_to_end_latency_ms": 12.0,
        },
    ]

    summary = summarize_generation_evaluations(rows)

    assert summary["num_examples"] == 2.0
    assert 0.0 <= summary["avg_context_relevance"] <= 1.0
    assert 0.0 <= summary["avg_groundedness"] <= 1.0
    assert summary["reranker_enabled_rate"] == 0.5
