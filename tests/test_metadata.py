from medical_rag_reranker.data.metadata import (
    extract_medical_metadata,
    infer_group_id,
    infer_question_intent,
)


def test_infer_group_id_uses_medquad_prefix() -> None:
    assert infer_group_id("0001072-3") == "0001072"
    assert infer_group_id("0001072-3__dup1") == "0001072"


def test_infer_question_intent_handles_common_medical_questions() -> None:
    assert infer_question_intent("What are the symptoms of X?") == "symptoms"
    assert infer_question_intent("How is X treated?") == "treatment"
    assert infer_question_intent("How is X diagnosed?") == "diagnosis"


def test_extract_medical_metadata_prefers_topic_for_diagnosis() -> None:
    row = {
        "question_id": "0001-2",
        "question": "What are the symptoms of X-linked disease?",
        "topic": "X-linked disease",
        "source": "MedQuAD",
    }

    metadata = extract_medical_metadata(row)

    assert metadata["group_id"] == "0001"
    assert metadata["question_intent"] == "symptoms"
    assert metadata["diagnosis_or_topic"] == "X-linked disease"
    assert metadata["source"] == "MedQuAD"
