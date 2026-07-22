"""Text, list, and unanswerable normalization shared by evaluation."""

from __future__ import annotations

import ast
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Sequence


def normalize_text(text: Any, *, use_vietnamese_tokenization: bool = False) -> str:
    value = unicodedata.normalize("NFC", "" if text is None else str(text)).strip().lower()
    if use_vietnamese_tokenization:
        try:
            from pyvi import ViTokenizer

            value = ViTokenizer.tokenize(value)
        except ImportError:
            pass
    value = " ".join(value.split())
    return value.rstrip(".").rstrip()


def parse_list_items(text: Any) -> list[str] | None:
    raw = str(text).strip() if text is not None else ""
    if not raw:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)) and parsed and all(str(item).strip() for item in parsed):
            return [str(item).strip() for item in parsed]
    separator = ";" if ";" in raw else "," if "," in raw else None
    if separator is None:
        return None
    items = [item.strip() for item in raw.split(separator)]
    return items if len(items) > 1 and all(items) else None


def has_list_hint(hints: Sequence[Any] | None) -> bool:
    for hint in hints or []:
        value = unicodedata.normalize("NFD", normalize_text(hint))
        value = "".join(char for char in value if unicodedata.category(char) != "Mn")
        value = re.sub(r"[^a-z0-9]+", "", value.replace("đ", "d"))
        if value == "list" or "lietke" in value:
            return True
    return False


def exact_text_match(prediction: Any, reference: Any, *, hints: Sequence[Any] | None = None, use_vietnamese_tokenization: bool = False) -> bool:
    if has_list_hint(hints):
        pred_items = parse_list_items(prediction)
        ref_items = parse_list_items(reference)
        if pred_items and ref_items:
            normalize = lambda item: normalize_text(item, use_vietnamese_tokenization=use_vietnamese_tokenization)
            return Counter(map(normalize, pred_items)) == Counter(map(normalize, ref_items))
    return normalize_text(prediction, use_vietnamese_tokenization=use_vietnamese_tokenization) == normalize_text(reference, use_vietnamese_tokenization=use_vietnamese_tokenization)


def is_unanswerable_reference(answer: Any) -> bool:
    return normalize_text(answer) in {"null", "nul"}


def is_unanswerable_prediction(answer: Any, *, use_vietnamese_tokenization: bool = False) -> bool:
    return normalize_text(answer, use_vietnamese_tokenization=use_vietnamese_tokenization) in {"", "null", "nul", "none", "nan", "n/a", "na", "không thể trả lời"}


def prediction_is_unanswerable(candidates: Iterable[str], *, use_vietnamese_tokenization: bool = False) -> bool:
    return any(is_unanswerable_prediction(candidate, use_vietnamese_tokenization=use_vietnamese_tokenization) for candidate in candidates)

