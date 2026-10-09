"""HTTP clients for the ASR server and the corrector LLM (stdlib only).

Each thread keeps one persistent HTTP/1.1 connection per host, so the
millions of requests of a full run do not each pay a TCP handshake. Transient
failures (connection reset, timeouts, 5xx, 429) are retried with jittered
backoff; 4xx answers are permanent and surface immediately.
"""

from __future__ import annotations

import http.client
import json
import random
import re
import threading
import time
from urllib.parse import urlsplit

from .text import LANGUAGE_NAMES, local_path

AUDIO_PLACEHOLDER_URL = "file://"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class PermanentError(RuntimeError):
    """The server rejected the request itself (4xx): retrying cannot help."""


class TransientError(RuntimeError):
    """Gave up after retries on network errors / 5xx / 429."""


class JsonClient:
    def __init__(self, base_url: str, timeout: float = 600.0, retries: int = 6) -> None:
        parts = urlsplit(base_url.rstrip("/"))
        self.host, self.port = parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)
        self.https = parts.scheme == "https"
        self.prefix = parts.path
        self.timeout = timeout
        self.retries = retries
        self._local = threading.local()

    def _conn(self) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
            conn = cls(self.host, self.port, timeout=self.timeout)
            self._local.conn = conn
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def request(self, method: str, route: str, body: dict | None = None) -> dict:
        payload = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json"} if payload else {}
        last = ""
        for attempt in range(self.retries + 1):
            try:
                conn = self._conn()
                conn.request(method, self.prefix + route, body=payload, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                if resp.status == 200:
                    return json.loads(data) if data else {}
                detail = data.decode("utf-8", errors="replace")[:1500]
                if resp.status in (429,) or resp.status >= 500:
                    last = f"HTTP {resp.status}: {detail}"
                else:
                    raise PermanentError(f"{route} -> HTTP {resp.status}: {detail}")
            except PermanentError:
                raise
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                self._drop()
            if attempt < self.retries:
                time.sleep(min(60.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
        raise TransientError(f"{route} failed after {self.retries + 1} attempts: {last}")

    def healthy(self) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=10)
            conn.request("GET", self.prefix + "/health")
            ok = conn.getresponse().status == 200
            conn.close()
            return ok
        except OSError:
            return False


class ASRClient:
    """QASR on vLLM (scripts/serve_asr_vllm.sh). Round-robins over several servers."""

    def __init__(self, urls: list[str], model: str = "qasr", timeout: float = 600.0,
                 max_tokens: int = 512) -> None:
        self.clients = [JsonClient(u, timeout=timeout) for u in urls]
        self.model = model
        self.max_tokens = max_tokens
        self._rr = 0
        self._lock = threading.Lock()

    def _pick(self) -> JsonClient:
        with self._lock:
            self._rr += 1
            return self.clients[self._rr % len(self.clients)]

    def transcribe(self, path: str, lang: str, repetition_penalty: float | None = None) -> dict:
        """Greedy transcript of one clip, with the model's own language label and finish reason.

        ``repetition_penalty`` is only for re-running a clip that looped; plain
        transcription never uses it (it would also penalize real repetitions).
        """
        body = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            **({"repetition_penalty": repetition_penalty} if repetition_penalty else {}),
            "messages": [
                {"role": "system", "content": LANGUAGE_NAMES.get(lang, lang)},
                {"role": "user", "content": [
                    {"type": "audio_url", "audio_url": {"url": AUDIO_PLACEHOLDER_URL + local_path(path)}}]},
            ],
        }
        res = self._pick().request("POST", "/v1/chat/completions", body)
        choice = res["choices"][0]
        raw = choice["message"].get("content") or ""
        head, sep, text = raw.rpartition("<asr_text>")
        label = head.strip()[len("language "):].strip() if sep and head.strip().startswith("language ") else None
        return {"text": (text if sep else raw).strip(), "language": label,
                "finish_reason": choice.get("finish_reason")}


class LLMClient:
    """The corrector (vLLM, OpenAI chat API) on this node."""

    def __init__(self, url: str, model: str = "corrector", timeout: float = 1800.0) -> None:
        self.client = JsonClient(url, timeout=timeout)
        self.model = model

    def chat(self, messages: list[dict], *, thinking: bool, max_tokens: int) -> tuple[str, dict]:
        if thinking:
            # Qwen's guidance for thinking mode: sample, never greedy.
            sampling = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
        else:
            # Editing, not writing: stay close to the most likely tokens.
            sampling = {"temperature": 0.2, "top_p": 0.9, "top_k": 20}
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking},
            **sampling,
        }
        res = self.client.request("POST", "/v1/chat/completions", body)
        choice = res["choices"][0]
        content = choice["message"].get("content") or ""
        # Servers without --reasoning-parser (the 27B nodes) inline the reasoning.
        content = _THINK_RE.sub("", content)
        if "<think>" in content:  # unterminated: the answer never started
            content = ""
        usage = res.get("usage") or {}
        return content.strip(), {"finish_reason": choice.get("finish_reason"),
                                 "prompt_tokens": usage.get("prompt_tokens", 0),
                                 "completion_tokens": usage.get("completion_tokens", 0)}
