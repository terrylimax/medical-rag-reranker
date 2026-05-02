from __future__ import annotations

import re
from typing import Any


_INTENT_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("symptoms", ("symptom", "signs of", "clinical features")),
    ("treatment", ("treat", "treatment", "therapy", "management", "cure")),
    ("diagnosis", ("diagnos", "test", "testing", "screening")),
    ("causes", ("cause", "causes", "risk factor")),
    ("prevention", ("prevent", "prevention")),
    ("inheritance", ("inherit", "genetic", "hereditary")),
    ("frequency", ("how common", "frequency", "prevalence")),
)


_DIAGNOSIS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"what (?:are|is) (?:the )?symptoms of (?P<value>.+?)\??$", re.I),
    re.compile(r"what causes (?P<value>.+?)\??$", re.I),
    re.compile(r"how (?:is|are) (?P<value>.+?) (?:diagnosed|treated)\??$", re.I),
    re.compile(r"what (?:is|are) (?:the )?treatments? for (?P<value>.+?)\??$", re.I),
    re.compile(r"how can (?P<value>.+?) be prevented\??$", re.I),
    re.compile(r"is (?P<value>.+?) inherited\??$", re.I),
    re.compile(r"what (?:is|are) (?P<value>.+?)\??$", re.I),
)


def infer_group_id(question_id: Any) -> str:
    text = str(question_id or "").strip()
    if not text:
        return ""
    text = text.split("__dup", 1)[0]
    return text.split("-", 1)[0]


def infer_question_intent(question: str) -> str:
    normalized = re.sub(r"\s+", " ", str(question or "").strip().lower())
    for intent, markers in _INTENT_PATTERNS:
        if any(marker in normalized for marker in markers):
            return intent
    return "other"


def infer_diagnosis_or_topic(question: str, topic: Any = None) -> str:
    topic_text = str(topic or "").strip()
    if topic_text:
        return topic_text

    normalized = re.sub(r"\s+", " ", str(question or "").strip())
    for pattern in _DIAGNOSIS_PATTERNS:
        match = pattern.search(normalized)
        if not match:
            continue
        value = match.group("value").strip(" ?.").strip()
        return value
    return ""


def extract_medical_metadata(row: dict[str, Any]) -> dict[str, str]:
    question = str(row.get("question") or "")
    return {
        "group_id": infer_group_id(row.get("question_id")),
        "question_intent": infer_question_intent(question),
        "diagnosis_or_topic": infer_diagnosis_or_topic(
            question,
            topic=row.get("topic"),
        ),
        "source": str(row.get("source") or "").strip(),
    }
