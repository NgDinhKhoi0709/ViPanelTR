"""
LLM call wrapper with application-level retry and parse validation.

Retries when:
- The HTTP call itself fails (exception)
- The response cannot be parsed into valid JSON
- Required fields are missing from the parsed result
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .json_parser import parse_json_response
from .logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_RETRIES = 4
_RETRY_SLEEP_SECONDS = 15.0


def _is_parse_ok(parsed: Dict[str, Any], required_fields: List[str]) -> bool:
    """Return True if *parsed* looks like a valid response with the expected keys."""
    if not parsed:
        return False
    if parsed.get("_parse_failed"):
        return False
    for field in required_fields:
        if field not in parsed:
            return False
    return True


def _stringify_answer(value: Any) -> str:
    """Normalise an answer value that may be a list into a plain string."""
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if v is not None]
        return ", ".join(parts)
    return str(value)


def call_llm_with_retry(
    llm_client,
    prompt: str,
    required_fields: List[str],
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    caller_name: str = "",
    temperature: Optional[float] = None,
    qa_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str, bool]:
    """Call *llm_client.generate* with up to *max_retries* attempts.

    Each attempt:
      1. ``llm_client.generate(prompt)``
      2. ``parse_json_response(response)``
      3. Check ``_parse_failed`` flag **and** presence of *required_fields*.

    Returns
    -------
    (parsed_dict, raw_response, success)
        *success* is ``True`` only when all required fields are present.
        On total failure the best-effort parsed dict is returned with
        ``success=False`` so callers can still extract partial data.
    """
    best_parsed: Dict[str, Any] = {}
    best_raw: str = ""
    best_score: int = -1

    gen_kwargs: Dict[str, Any] = {}
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if qa_id:
        gen_kwargs["qa_id"] = qa_id

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = llm_client.generate(prompt, **gen_kwargs)
            parsed = parse_json_response(raw_response)

            if _is_parse_ok(parsed, required_fields):
                return parsed, raw_response, True

            score = sum(1 for f in required_fields if f in parsed and not parsed.get("_parse_failed"))
            if score > best_score:
                best_score = score
                best_parsed = parsed
                best_raw = raw_response

            logger.warning(
                "[%s] Attempt %d/%d: parsed response missing required fields %s (got keys: %s)",
                caller_name,
                attempt,
                max_retries,
                [f for f in required_fields if f not in parsed],
                list(parsed.keys()),
            )

        except Exception as exc:
            logger.warning(
                "[%s] Attempt %d/%d failed with exception: %s",
                caller_name,
                attempt,
                max_retries,
                exc,
            )

        if attempt < max_retries:
            time.sleep(_RETRY_SLEEP_SECONDS)

    return best_parsed, best_raw, False
