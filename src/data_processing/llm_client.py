"""Async vLLM OpenAI-compatible client with retry and backoff."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from typing import Any

import httpx

from .config import PipelineConfig

LOGGER = logging.getLogger("data_processing.llm")

# Regex to extract JSON from markdown code fences
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)

# Regex to strip Qwen3 thinking blocks (<think>...</think> or raw thinking preamble)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class TokenBucketRateLimiter:
    """Token-bucket rate limiter to prevent overwhelming the vLLM scheduler.

    Each worker gets its own bucket. Tokens refill at a steady rate;
    a request must acquire a token before proceeding.
    """

    def __init__(self, rate: float = 10.0, capacity: float = 20.0) -> None:
        """Initialize the rate limiter.

        Args:
            rate: Tokens added per second (sustained request rate).
            capacity: Maximum burst size (bucket capacity).
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a single token, waiting if the bucket is empty."""
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Wait a short interval before retrying
            await asyncio.sleep(1.0 / self.rate)

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now


class LLMClientError(RuntimeError):
    """Raised when the LLM client encounters an unrecoverable error."""


class LLMResponseParseError(LLMClientError):
    """Raised when the LLM response cannot be parsed as JSON."""


class VLLMClient:
    """Async client for vLLM's OpenAI-compatible chat completions API."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        rate_limit: float = 10.0,
        rate_capacity: float = 20.0,
    ) -> None:
        self.config = config
        self.base_url = config.vllm_base_url
        self.model = config.vllm_model
        self.max_retries = config.max_retries
        self.backoff_base = config.retry_backoff_base
        self.timeout = config.request_timeout
        self._client: httpx.AsyncClient | None = None
        self._rate_limiter = TokenBucketRateLimiter(
            rate=rate_limit, capacity=rate_capacity
        )

    async def __aenter__(self) -> VLLMClient:
        await self._ensure_client()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if the vLLM server is healthy."""
        client = await self._ensure_client()
        try:
            # vLLM exposes /health at the root, NOT under /v1
            # Use absolute URL to bypass base_url prefix
            health_url = f"http://{self.config.vllm_host}:{self.config.vllm_port}/health"
            response = await client.get(health_url, timeout=5.0)
            return response.status_code == 200
        except (httpx.HTTPError, httpx.TimeoutException):
            return False

    async def wait_for_health(self, max_wait_seconds: float = 300.0) -> None:
        """Wait until the vLLM server is healthy or timeout."""
        elapsed = 0.0
        interval = 2.0
        while elapsed < max_wait_seconds:
            if await self.health_check():
                LOGGER.info("vLLM server is healthy at %s", self.base_url)
                return
            await asyncio.sleep(interval)
            elapsed += interval
            LOGGER.debug("Waiting for vLLM health... (%.0fs elapsed)", elapsed)
        raise LLMClientError(
            f"vLLM server not healthy after {max_wait_seconds}s at {self.base_url}"
        )

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 12288,
        thinking: bool = False,
    ) -> str:
        """Send a chat completion request and return the response content.

        Retries with exponential backoff on transient errors (429, 500, 503).
        Rate-limited via a per-worker token bucket.
        """
        # Acquire rate-limit token before sending request
        await self._rate_limiter.acquire()

        client = await self._ensure_client()
        payload = {
            "model": self.model,
            "messages": messages,
            # Qwen3 recommended sampling. Greedy/near-greedy decoding is
            # explicitly discouraged for Qwen3 — it degenerates into
            # repetition loops and confabulated output (observed: validator
            # issue lists repeating the same complaint 6x at temperature=0.0).
            # Thinking mode uses its own recommended set (0.6 / 0.95).
            "temperature": 0.6 if thinking else temperature,
            "top_p": 0.95 if thinking else 0.8,
            "top_k": 20,
            # Suppresses the residual repetition loops sampling alone misses
            "presence_penalty": 1.5,
            "max_tokens": max_tokens,
            # Thinking mode trades tokens/latency for reasoning depth;
            # parse_json_response strips the <think> block either way.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await client.post("/chat/completions", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    choice = data["choices"][0]
                    # Truncated output produces unparseable JSON downstream —
                    # surface it here so the root cause is visible in logs
                    if choice.get("finish_reason") == "length":
                        LOGGER.warning(
                            "LLM response truncated at max_tokens=%d — "
                            "output is likely incomplete",
                            payload["max_tokens"],
                        )
                    return choice["message"]["content"]

                # Prompt + max_tokens exceeded the model context window:
                # shrink the completion budget and retry immediately
                if (
                    response.status_code == 400
                    and "maximum context length" in response.text
                    and payload["max_tokens"] > 1024
                ):
                    payload["max_tokens"] = payload["max_tokens"] // 2
                    LOGGER.warning(
                        "Context length exceeded — retrying with max_tokens=%d",
                        payload["max_tokens"],
                    )
                    continue

                # Retryable status codes
                if response.status_code in (429, 500, 503):
                    last_error = LLMClientError(
                        f"vLLM returned {response.status_code}: {response.text[:200]}"
                    )
                    if attempt < self.max_retries:
                        await self._backoff(attempt)
                        continue

                # Non-retryable error
                raise LLMClientError(
                    f"vLLM request failed with status {response.status_code}: "
                    f"{response.text[:500]}"
                )

            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    LOGGER.warning(
                        "Request failed (attempt %d/%d): %s",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                    )
                    await self._backoff(attempt)
                    continue

        raise LLMClientError(
            f"All {self.max_retries + 1} attempts failed. Last error: {last_error}"
        )

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter."""
        delay = self.backoff_base ** (attempt + 1)
        jitter = random.uniform(0, delay * 0.1)
        await asyncio.sleep(delay + jitter)

    async def chat_completion_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 12288,
        thinking: bool = False,
    ) -> Any:
        """Send a chat completion request and parse the response as JSON.

        Handles responses wrapped in markdown code fences.
        """
        content = await self.chat_completion(
            messages, temperature=temperature, max_tokens=max_tokens,
            thinking=thinking,
        )
        return parse_json_response(content)


def parse_json_response(content: str) -> Any:
    """Parse JSON from an LLM response, handling markdown fences and thinking blocks."""
    content = content.strip()

    # Strip Qwen3 <think>...</think> blocks (safety net if thinking mode leaks)
    content = _THINK_BLOCK_RE.sub("", content).strip()

    # Try direct parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = _JSON_FENCE_RE.search(content)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try finding JSON object or array boundaries
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        end = content.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise LLMResponseParseError(
        f"Could not parse JSON from LLM response: {content[:300]}..."
    )


__all__ = [
    "LLMClientError",
    "LLMResponseParseError",
    "TokenBucketRateLimiter",
    "VLLMClient",
    "parse_json_response",
]
