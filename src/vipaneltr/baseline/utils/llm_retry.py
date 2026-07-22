"""
LLM call wrapper with application-level retry for zero-shot text generation.

Retries when:
- The HTTP call itself fails (exception)
- The response is empty or functionally empty
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from ...utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_RETRIES = 4
_RETRY_SLEEP_SECONDS = 15.0


def call_llm_with_retry(
    llm_client,
    model: str,
    prompt: str,
    cfg: Any,
    *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    caller_name: str = "zeroshot",
    generate_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[str, bool]:
    """Call *llm_client.generate* with up to *max_retries* attempts.

    Each attempt:
      1. ``llm_client.generate(model=model, prompt=prompt, config=cfg)``
      2. Check if the response is empty.
      
    If exception occurs or response is empty, it sleeps and retries.

    Returns
    -------
    (response_text, success)
        *success* is ``True`` only when the call succeeds and returns non-empty text.
        On total failure, the best effort (or empty/error string) is returned with
        ``success=False``.
    """
    best_raw: str = ""
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            extra_kwargs = generate_kwargs or {}
            raw_response = llm_client.generate(model=model, prompt=prompt, config=cfg, **extra_kwargs)
            
            # Zero-shot typically expects a single short answer
            if raw_response and raw_response.strip():
                return raw_response.strip(), True

            logger.warning(
                "[%s] Attempt %d/%d: generated response is empty.",
                caller_name,
                attempt,
                max_retries,
            )
            
        except Exception as exc:
            err_str = str(exc).upper()
            if "429" in err_str or "RATE_LIMIT" in err_str or "QUOTA" in err_str:
                # the retry logic inside llm_client.generate() might have exhausted keys.
                # we should still retry if we are below max_retries at the application level?
                # or just fail fast if all keys are exhausted.
                # Here we continue to sleep and retry, maybe the limit resets.
                pass
                
            last_err = exc
            logger.warning(
                "[%s] Attempt %d/%d failed with exception: %s",
                caller_name,
                attempt,
                max_retries,
                exc,
            )

        if attempt < max_retries:
            time.sleep(_RETRY_SLEEP_SECONDS)

    if last_err is not None:
        best_raw = f"ERROR: {last_err}"
    else:
        best_raw = "ERROR: Empty response after retries."
        
    return best_raw, False
