from __future__ import annotations

import os
import sys
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


MODEL_PRICING = {
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}


def calculate_call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    model_name = (model or "").lower().split("/", 1)[-1]
    input_price, output_price = MODEL_PRICING["gpt-4o-mini"]
    for name, prices in MODEL_PRICING.items():
        if name in model_name:
            input_price, output_price = prices
            break
    return (prompt_tokens * input_price) + (completion_tokens * output_price)


def _usage_get(usage: Any, *names: str) -> Any:
    for name in names:
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is not None:
            return value
    return None


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text or "") + 3) // 4) if text else 0


def _normalize_usage(usage: Any, prompt: str, response: str) -> Dict[str, Any]:
    prompt_tokens = _as_int(_usage_get(usage, "prompt_tokens", "input_tokens")) or _estimate_tokens(prompt)
    completion_tokens = _as_int(_usage_get(usage, "completion_tokens", "output_tokens")) or _estimate_tokens(response)
    total_tokens = _as_int(_usage_get(usage, "total_tokens")) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provided_cost": _usage_get(usage, "cost_usd", "cost"),
    }


def _split_keys(s: str) -> List[str]:
    raw: List[str] = []
    for part in s.replace(";", ",").replace("\n", ",").split(","):
        p = part.strip()
        if p:
            raw.append(p)
    return raw


def _dedupe_keep_order(keys: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def get_openai_api_keys() -> List[str]:
    """
    Accepted env vars (priority):
      - GPT_API_KEYS
      - GPT_API_KEY_1..N
      - GPT_API_KEY (single key or CSV)
      - OPENAI_API_KEY
    """
    keys: List[str] = []

    multi = os.environ.get("GPT_API_KEYS")
    if multi:
        keys.extend(_split_keys(multi))

    for i in range(1, 51):
        k = os.environ.get(f"GPT_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())

    single = os.environ.get("GPT_API_KEY")
    if single and single.strip():
        if "," in single or ";" in single or "\n" in single:
            keys.extend(_split_keys(single))
        else:
            keys.append(single.strip())

    fallback = os.environ.get("OPENAI_API_KEY")
    if fallback and fallback.strip():
        keys.append(fallback.strip())

    out = _dedupe_keep_order(keys)
    if not out:
        raise RuntimeError(
            "Missing OpenAI key. Set GPT_API_KEY/GPT_API_KEYS or OPENAI_API_KEY."
        )
    return out


def get_openrouter_api_keys() -> List[str]:
    """
    Accepted env vars (priority):
      - OPENROUTER_API_KEYS
      - OPENROUTER_API_KEY_1..N
      - OPENROUTER_API_KEY (single key or CSV)
    """
    keys: List[str] = []

    multi = os.environ.get("OPENROUTER_API_KEYS")
    if multi:
        keys.extend(_split_keys(multi))

    for i in range(1, 51):
        k = os.environ.get(f"OPENROUTER_API_KEY_{i}")
        if k and k.strip():
            keys.append(k.strip())

    single = os.environ.get("OPENROUTER_API_KEY")
    if single and single.strip():
        if "," in single or ";" in single or "\n" in single:
            keys.extend(_split_keys(single))
        else:
            keys.append(single.strip())

    out = _dedupe_keep_order(keys)
    if not out:
        raise RuntimeError(
            "Missing OpenRouter key. Set OPENROUTER_API_KEY or OPENROUTER_API_KEYS."
        )
    return out


def parse_model_spec(model: str) -> Tuple[str, str]:
    """
    Parse model id:
      - openai/gpt-4o-mini            -> ("openai", "gpt-4o-mini")
      - openrouter/qwen/qwen3-8b      -> ("openrouter", "qwen/qwen3-8b")
      - gpt-4o-mini                   -> ("openai", "gpt-4o-mini")
    """
    s = str(model or "").strip()
    if not s:
        raise ValueError("Model id is empty.")

    if "/" in s:
        provider, rest = s.split("/", 1)
        p = provider.strip().lower()
        if p in {"openai", "openrouter"}:
            if not rest.strip():
                raise ValueError(f"Invalid model id: {model!r}")
            return p, rest.strip()
    return "openai", s


@dataclass
class GenConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 2048
    timeout: int = 60
    openrouter_provider: Optional[Dict[str, Any]] = None


class LLMZeroShotClient:
    """
    Zero-shot LLM wrapper for OpenAI + OpenRouter.
    """

    def __init__(
        self,
        openai_api_keys: Optional[List[str]] = None,
        openrouter_api_keys: Optional[List[str]] = None,
        openrouter_api_base: str = "https://openrouter.ai/api/v1",
    ):
        self._openai_keys = openai_api_keys
        self._openrouter_keys = openrouter_api_keys
        self._openai_idx = 0
        self._openrouter_idx = 0
        self._openrouter_api_base = openrouter_api_base.rstrip("/")
        self._thread_local = threading.local()

        self._OpenAI = None
        self._openai_client = None
        self._maybe_init_openai_client()

    def usage_cursor(self) -> int:
        return len(getattr(self._thread_local, "usage_history", []))

    def usage_since(self, cursor: int) -> Dict[str, Any]:
        records = getattr(self._thread_local, "usage_history", [])[cursor:]
        return {
            "prompt_tokens": sum(r["prompt_tokens"] for r in records),
            "completion_tokens": sum(r["completion_tokens"] for r in records),
            "total_tokens": sum(r["total_tokens"] for r in records),
            "cost_usd": round(sum(r["cost_usd"] for r in records), 6),
        }

    def _record_usage(self, model: str, prompt: str, response: str, usage: Any = None) -> None:
        normalized = _normalize_usage(usage, prompt, response)
        try:
            cost_usd = float(normalized["provided_cost"])
        except (TypeError, ValueError):
            cost_usd = calculate_call_cost(model, normalized["prompt_tokens"], normalized["completion_tokens"])
        history = getattr(self._thread_local, "usage_history", None)
        if history is None:
            history = []
            self._thread_local.usage_history = history
        history.append({**normalized, "cost_usd": cost_usd})

    def _maybe_init_openai_client(self) -> None:
        keys = self._openai_keys
        if keys is None:
            try:
                keys = get_openai_api_keys()
            except RuntimeError:
                return
            self._openai_keys = keys

        if not keys:
            return

        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("Missing dependency `openai`. Install it: pip install openai") from e

        self._OpenAI = OpenAI
        self._openai_client = OpenAI(api_key=keys[self._openai_idx])

    def _rotate_openai_key(self) -> bool:
        keys = self._openai_keys or []
        if self._OpenAI is None or self._openai_client is None or self._openai_idx + 1 >= len(keys):
            return False
        self._openai_idx += 1
        print(
            f"[key-rotate] OpenAI key {self._openai_idx + 1}/{len(keys)}",
            file=sys.stderr,
            flush=True,
        )
        self._openai_client = self._OpenAI(api_key=keys[self._openai_idx])
        return True

    def _rotate_openrouter_key(self) -> bool:
        keys = self._openrouter_keys
        if keys is None:
            try:
                keys = get_openrouter_api_keys()
            except RuntimeError:
                return False
            self._openrouter_keys = keys

        if self._openrouter_idx + 1 >= len(keys):
            return False
        self._openrouter_idx += 1
        print(
            f"[key-rotate] OpenRouter key {self._openrouter_idx + 1}/{len(keys)}",
            file=sys.stderr,
            flush=True,
        )
        return True

    @staticmethod
    def _is_quota_exhausted(err: BaseException) -> bool:
        blob = (repr(err) + " " + str(err)).upper()
        return ("429" in blob) or ("RATE_LIMIT" in blob) or ("QUOTA" in blob)

    @staticmethod
    def _openrouter_input_from_prompt(prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ]

    @staticmethod
    def _extract_openrouter_output_text(payload: Dict[str, Any]) -> str:
        parts: List[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
        return "".join(parts).strip()

    def _generate_openai(self, model_name: str, prompt: str, cfg: GenConfig) -> str:
        if self._openai_client is None:
            self._maybe_init_openai_client()
        if self._openai_client is None:
            raise RuntimeError(
                "OpenAI client is unavailable. Set GPT_API_KEY/GPT_API_KEYS or OPENAI_API_KEY."
            )

        response = self._openai_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
        )
        if response.choices and len(response.choices) > 0:
            content = response.choices[0].message.content
            text = content.strip() if content else ""
            self._record_usage(f"openai/{model_name}", prompt, text, response.usage)
            return text
        return ""

    def _generate_openrouter(self, model_name: str, prompt: str, cfg: GenConfig) -> str:
        try:
            import requests  # type: ignore
        except Exception as e:
            raise RuntimeError("Missing dependency `requests`. Install it: pip install requests") from e

        if self._openrouter_keys is None:
            self._openrouter_keys = get_openrouter_api_keys()
        if not self._openrouter_keys:
            raise RuntimeError("OpenRouter client is unavailable: missing OPENROUTER_API_KEY.")

        api_key = self._openrouter_keys[self._openrouter_idx]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body: Dict[str, Any] = {
            "model": model_name,
            "input": self._openrouter_input_from_prompt(prompt),
            "max_output_tokens": cfg.max_tokens,
            "temperature": cfg.temperature,
        }
        if cfg.top_p is not None:
            body["top_p"] = cfg.top_p
        if cfg.openrouter_provider is not None:
            body["provider"] = cfg.openrouter_provider

        url = f"{self._openrouter_api_base}/responses"
        resp = requests.post(url, headers=headers, json=body, timeout=cfg.timeout)
        if resp.status_code != 200:
            excerpt = (resp.text or "").strip()
            if len(excerpt) > 2000:
                excerpt = excerpt[:2000] + "...(truncated)"
            raise RuntimeError(f"OpenRouter request failed (status={resp.status_code}): {excerpt}")

        data = resp.json()
        text = self._extract_openrouter_output_text(data)
        if text:
            self._record_usage(f"openrouter/{model_name}", prompt, text, data.get("usage"))
            return text
        raise RuntimeError("OpenRouter response did not contain any output_text content.")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        config: Optional[GenConfig] = None,
        max_retries: int = 6,
    ) -> str:
        cfg = config or GenConfig()
        provider, model_name = parse_model_spec(model)
        last_err: Optional[BaseException] = None

        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    return self._generate_openai(model_name, prompt, cfg)
                if provider == "openrouter":
                    return self._generate_openrouter(model_name, prompt, cfg)
                raise ValueError(
                    f"Unsupported provider: {provider}. Use openai/<model> or openrouter/<model>."
                )
            except Exception as e:
                last_err = e
                if self._is_quota_exhausted(e):
                    if provider == "openai" and self._rotate_openai_key():
                        continue
                    if provider == "openrouter" and self._rotate_openrouter_key():
                        continue
                time.sleep(min(8.0, 0.5 * (2**attempt)))

        raise RuntimeError(f"Generation failed for model={model}. Last error: {last_err!r}") from last_err
