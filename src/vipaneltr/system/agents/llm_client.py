"""
LLM client for multiple providers.

Supports OpenAI, Google Gemini, Anthropic Claude, and OpenRouter.
"""

from __future__ import annotations

import os
import sys
import time
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...config import ModelConfig


def _estimate_tokens_from_text(text: str) -> int:
    """
    Very rough heuristic: ~4 chars/token for English-like text.
    Vietnamese + JSON can vary, but this is good enough for diagnostics.
    """
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


# Keep this pricing policy aligned with POMA. Values are USD per token.
MODEL_PRICING = {
    "gpt-4o-mini": (0.15 / 1_000_000, 0.60 / 1_000_000),
    "gpt-4o": (2.50 / 1_000_000, 10.00 / 1_000_000),
    "gpt-3.5-turbo": (0.50 / 1_000_000, 1.50 / 1_000_000),
}


def calculate_call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate USD cost using POMA's model-price table and fallback."""
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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_usage(usage: Any, prompt: str, response: str) -> Dict[str, Any]:
    """Normalize provider usage; estimate unavailable fields from text."""
    prompt_tokens = _as_int(_usage_get(usage, "prompt_tokens", "input_tokens", "prompt_token_count"))
    completion_tokens = _as_int(
        _usage_get(usage, "completion_tokens", "output_tokens", "candidates_token_count")
    )
    total_tokens = _as_int(_usage_get(usage, "total_tokens", "total_token_count"))
    if prompt_tokens <= 0:
        prompt_tokens = _estimate_tokens_from_text(prompt)
    if completion_tokens <= 0:
        completion_tokens = _estimate_tokens_from_text(response)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "provided_cost": _usage_get(usage, "cost_usd", "cost"),
    }


class UsageTrackingMixin:
    """Thread-safe per-QA usage accumulator shared by all provider clients."""

    def _init_usage_tracking(self) -> None:
        self._usage_lock = threading.Lock()
        self._usage_by_qa: Dict[str, Dict[str, Any]] = {}

    def _record_usage(self, qa_id: Optional[str], prompt: str, response: str, usage: Any = None) -> None:
        if not qa_id:
            return
        normalized = _normalize_usage(usage, prompt, response)
        try:
            cost_usd = float(normalized["provided_cost"])
        except (TypeError, ValueError):
            cost_usd = calculate_call_cost(
                self.model, normalized["prompt_tokens"], normalized["completion_tokens"]
            )
        with self._usage_lock:
            total = self._usage_by_qa.setdefault(
                str(qa_id),
                {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
            )
            total["prompt_tokens"] += normalized["prompt_tokens"]
            total["completion_tokens"] += normalized["completion_tokens"]
            total["total_tokens"] += normalized["total_tokens"]
            total["cost_usd"] += cost_usd

    def get_usage(self, qa_id: str) -> Dict[str, Any]:
        with self._usage_lock:
            usage = dict(self._usage_by_qa.get(str(qa_id), {}))
        return {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "cost_usd": round(float(usage.get("cost_usd", 0.0)), 6),
        }


def _looks_like_context_overflow_error(err_text: str) -> bool:
    if not err_text:
        return False
    s = err_text.lower()
    needles = (
        "context_length_exceeded",
        "maximum context length",
        "max context length",
        "context length",
        "too many tokens",
        "prompt is too long",
        "input is too long",
        "request too large",
        "payload too large",
        "token limit",
        "exceeds the maximum",
    )
    return any(n in s for n in needles)


def _print_context_overflow_diagnostics(
    *,
    provider: str,
    model: str,
    messages: List[Dict[str, str]],
    max_output_tokens: Optional[int],
    err_text: str,
) -> None:
    """Print a concise diagnostic block to stderr."""
    # Only print when we strongly suspect a context/token overflow.
    if not _looks_like_context_overflow_error(err_text):
        return

    joined = "\n".join(str(m.get("content") or "") for m in (messages or []))
    chars = len(joined)
    est_tokens = _estimate_tokens_from_text(joined)
    roles = [str(m.get("role") or "") for m in (messages or [])]

    print(
        "\n[context-overflow] Request likely exceeded model context window\n"
        f"- provider: {provider}\n"
        f"- model: {model}\n"
        f"- messages: {len(messages or [])} roles={roles}\n"
        f"- prompt_chars: {chars}\n"
        f"- prompt_tokens_est(~chars/4): {est_tokens}\n"
        f"- max_output_tokens: {max_output_tokens}\n"
        f"- error_excerpt: {err_text[:500]}\n",
        file=sys.stderr,
        flush=True,
    )


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_with_messages(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """
        Generate response from chat messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text response
        """
        pass


class OpenAIClient(UsageTrackingMixin, LLMClient):
    """OpenAI API client."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize OpenAI client.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self.model = config.name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._init_usage_tracking()
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env var.")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=config.api_base if config.api_base else None,
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, **kwargs)
    
    def generate_with_messages(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """Generate response from chat messages."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            self._record_usage(kwargs.get("qa_id"), "\n".join(m.get("content", "") for m in messages), text, response.usage)
            return text
        except Exception as e:
            _print_context_overflow_diagnostics(
                provider="openai",
                model=self.model,
                messages=messages,
                max_output_tokens=max_tokens,
                err_text=f"{type(e).__name__}: {e}",
            )
            raise


class GeminiClient(UsageTrackingMixin, LLMClient):
    """Google Gemini API client."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Gemini client.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.api_key = config.api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = config.name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._init_usage_tracking()
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY env var.")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        import google.generativeai as genai
        
        generation_config = genai.types.GenerationConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config,
        )
        
        text = response.text or ""
        self._record_usage(kwargs.get("qa_id"), prompt, text, getattr(response, "usage_metadata", None))
        return text
    
    def generate_with_messages(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """Generate response from chat messages."""
        # Convert messages to Gemini format
        chat = self.client.start_chat(history=[])
        
        for msg in messages[:-1]:  # Add history
            if msg["role"] == "user":
                chat.send_message(msg["content"])
        
        # Send last message and get response
        last_msg = messages[-1]["content"] if messages else ""
        response = chat.send_message(last_msg)
        
        text = response.text or ""
        self._record_usage(kwargs.get("qa_id"), "\n".join(m.get("content", "") for m in messages), text, getattr(response, "usage_metadata", None))
        return text


class ClaudeClient(UsageTrackingMixin, LLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize Claude client.
        
        Args:
            config: Model configuration
        """
        self.config = config
        self.api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = config.name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self._init_usage_tracking()
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY env var.")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, **kwargs)
    
    def generate_with_messages(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> str:
        """Generate response from chat messages."""
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=max_tokens,
            )
            text = response.content[0].text or ""
            self._record_usage(kwargs.get("qa_id"), "\n".join(m.get("content", "") for m in messages), text, response.usage)
            return text
        except Exception as e:
            _print_context_overflow_diagnostics(
                provider="claude",
                model=self.model,
                messages=messages,
                max_output_tokens=max_tokens,
                err_text=f"{type(e).__name__}: {e}",
            )
            raise


class OpenRouterClient(UsageTrackingMixin, LLMClient):
    """
    OpenRouter Responses API client.

    This client intentionally follows the request shape used in `tmp.py`:
    - POST https://openrouter.ai/api/v1/responses
    - JSON body: { model, input: [{type:'message', role, content:[{type:'input_text', text}]}], max_output_tokens }
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = config.name  # OpenRouter model id, e.g. "qwen/qwen3-8b"
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.api_base = (config.api_base or "https://openrouter.ai/api/v1").rstrip("/")
        # Optional OpenRouter provider routing preferences (passed verbatim as request field "provider").
        # Example: {"only": ["anthropic", "amazon-bedrock", "google-vertex"]}
        self.provider_routing = getattr(config, "openrouter_provider", None)
        self._init_usage_tracking()

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY env var (or model.api_key in config)."
            )

    @staticmethod
    def _messages_to_openrouter_input(messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        # Map our simple {role, content} list to OpenRouter's Responses API input format.
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = (msg.get("role") or "user").strip()
            content = msg.get("content") or ""
            out.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [
                        {
                            "type": "input_text",
                            "text": content,
                        }
                    ],
                }
            )
        return out

    @staticmethod
    def _extract_output_text(payload: Dict[str, Any]) -> str:
        """
        Extract assistant text from OpenRouter non-streaming Responses API payload.

        Expected shape (docs):
          payload["output"][...]["content"][...] where content items have type == "output_text" and include "text".
        """
        parts: List[str] = []
        for item in payload.get("output", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            for content_item in item.get("content", []) or []:
                if not isinstance(content_item, dict):
                    continue
                if content_item.get("type") == "output_text":
                    text = content_item.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
        return "".join(parts).strip()

    def generate(self, prompt: str, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.generate_with_messages(messages, **kwargs)

    def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            import requests  # type: ignore
        except ImportError as e:
            raise ImportError("requests package not installed. Run: pip install requests") from e

        url = f"{self.api_base}/responses"
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        max_retries = int(kwargs.get("max_retries", 4))
        provider_routing = kwargs.get("openrouter_provider", kwargs.get("provider_routing", self.provider_routing))

        body: Dict[str, Any] = {
            "model": self.model,
            "input": self._messages_to_openrouter_input(messages),
            "max_output_tokens": max_tokens,
        }
        # Temperature is supported by the Responses API; include it for parity with other providers.
        if temperature is not None:
            body["temperature"] = temperature
        # OpenRouter routing: include provider preference if configured.
        if provider_routing is not None:
            body["provider"] = provider_routing

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_err: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
                if resp.status_code != 200:
                    # Include a short excerpt of response text to aid debugging without flooding logs.
                    excerpt = (resp.text or "").strip()
                    if len(excerpt) > 2000:
                        excerpt = excerpt[:2000] + "...(truncated)"
                    _print_context_overflow_diagnostics(
                        provider="openrouter",
                        model=self.model,
                        messages=messages,
                        max_output_tokens=max_tokens,
                        err_text=f"status={resp.status_code} {excerpt}",
                    )
                    raise RuntimeError(
                        f"OpenRouter request failed (status={resp.status_code}). Response: {excerpt}"
                    )

                data = resp.json()
                text = self._extract_output_text(data)
                if text:
                    self._record_usage(kwargs.get("qa_id"), "\n".join(m.get("content", "") for m in messages), text, data.get("usage"))
                    return text

                # Fallback: if no output_text found, surface the payload for debugging.
                raise RuntimeError("OpenRouter response did not contain any output_text content.")
            except Exception as e:  # requests errors, JSON errors, runtime errors
                last_err = e
                _print_context_overflow_diagnostics(
                    provider="openrouter",
                    model=self.model,
                    messages=messages,
                    max_output_tokens=max_tokens,
                    err_text=f"{type(e).__name__}: {e}",
                )
                # Simple exponential backoff for transient failures
                if attempt + 1 < max_retries:
                    time.sleep(min(8.0, 0.5 * (2**attempt)))
                    continue
                break

        raise RuntimeError(f"OpenRouter generation failed for model={self.model}. Last error: {last_err!r}") from last_err


def create_llm_client(config: ModelConfig) -> LLMClient:
    """
    Factory function to create LLM client based on provider.
    
    Args:
        config: Model configuration
        
    Returns:
        LLMClient instance
    """
    provider = config.provider.lower()
    
    if provider == "openai":
        return OpenAIClient(config)
    elif provider == "gemini":
        return GeminiClient(config)
    elif provider == "claude" or provider == "anthropic":
        return ClaudeClient(config)
    elif provider == "openrouter":
        return OpenRouterClient(config)
    else:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: openai, gemini, claude, openrouter"
        )
