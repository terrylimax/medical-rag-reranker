from __future__ import annotations

import re
from typing import Iterable


ASPECT_ALIASES: dict[str, tuple[str, ...]] = {
    "symptoms": ("symptom", "sign", "manifestation"),
    "causes": ("cause", "causes", "etiology", "risk factor", "why"),
    "treatments": ("treatment", "therapy", "therapies", "drug", "medicine"),
    "diagnosis": ("diagnosis", "diagnose", "diagnostic", "test", "screening"),
    "inheritance": ("inherit", "inherited", "inheritance", "genetic"),
    "management": ("management", "manage", "prevention", "prevent", "living with"),
}


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def coerce_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def detect_aspects(text: object) -> set[str]:
    normalized = normalize_text(text)
    found: set[str] = set()
    for aspect, aliases in ASPECT_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            found.add(aspect)
    return found


def aspects_from_question_type(question_type: object) -> set[str]:
    return detect_aspects(question_type)


def aspects_from_metadata(row: dict[str, object]) -> set[str]:
    aspects = set()
    for key in ("requested_aspects", "aspects"):
        for value in coerce_str_list(row.get(key)):
            value_norm = normalize_text(value).replace(" ", "_")
            if value_norm in ASPECT_ALIASES:
                aspects.add(value_norm)
            else:
                aspects.update(detect_aspects(value))
    aspects.update(aspects_from_question_type(row.get("question_type")))
    return aspects


def format_aspect_list(aspects: Iterable[str]) -> str:
    labels = {
        "symptoms": "symptoms",
        "causes": "causes",
        "treatments": "treatments",
        "diagnosis": "diagnosis",
        "inheritance": "inheritance",
        "management": "management",
    }
    values = [labels.get(a, a) for a in aspects]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + f", and {values[-1]}"
