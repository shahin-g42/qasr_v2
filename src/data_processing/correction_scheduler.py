"""Continuous strict correction scheduler for the exhaustive stage 2.

Unlike the legacy ``BatchCorrector`` (which processes one accent group at a
time behind chunk-wide read/correct/place barriers), this scheduler keeps a
node's full request budget in flight continuously:

* one persistent pooled ``httpx.Client`` and one shared thread pool sized to
  ``concurrency`` (default 80 — aggregate in-flight HTTP requests per node; a
  request carrying 16 transcripts is ONE vLLM sequence, not 16);
* strict batch parsing — indices must be unique and in range, text must be a
  nonempty string, review verdicts must be literal booleans. Anything invalid
  becomes an unresolved item, never silently accepted;
* failure healing by recursive splitting: when a batch fails (transport,
  parse, or a ``finish_reason="length"`` truncation), BOTH halves are retried,
  down to single items. Nothing is ever dropped to a silent fallback —
  exhausted items return ``ok=False`` results that the caller persists as
  quarantine.

The scheduler is transport-agnostic (``transport`` accepts any
``httpx.BaseTransport``, so tests use ``httpx.MockTransport``) and reports
through injected callbacks; it owns no durable state of its own — the caller
bridges stage events into ``run_state.SliceState``.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

LOGGER = logging.getLogger("data_processing.scheduler")

__all__ = [
    "CorrectionScheduler",
    "SchedulerConfig",
    "parse_batch_response",
]


@dataclass
class SchedulerConfig:
    """One authoritative set of stage-2 correction knobs."""

    url: str = "http://localhost:8010/v1"
    model: str = "corrector"
    #: Aggregate concurrent HTTP requests per node (not per-CPU-core workers).
    concurrency: int = 80
    #: Transcripts packed into one request. 16 is one completion sequence.
    batch_size: int = 16
    #: Completion budget; a local vLLM deployment makes this cheap.
    max_tokens: int = 12288
    timeout: float = 600.0
    #: Transport attempts per request (transient statuses/backoff within).
    max_transport_attempts: int = 5
    temperature: float = 0.6
    progress_interval: float = 30.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_transport_attempts < 1:
            raise ValueError("max_transport_attempts must be >= 1")


def _messages(lang: str, stage: str, items: list[dict]) -> list[dict[str, str]]:
    """Build the request body. Delegates to the rich prompt families.

    Correct batches carry just the normalized transcripts; review batches
    carry (original, cleaned) pairs so the reviewer can compare them; judge
    batches (stage 3) carry the FINAL corrected text under ``candidate``.
    """
    from . import generic_prompts, prompts

    if stage == "review":
        pairs = [
            (it.get("text") or it.get("normalized_text") or "",
             it.get("candidate") or it.get("normalized_text") or "")
            for it in items
        ]
        if lang == "ar":
            return prompts.build_batch_review_messages(pairs)
        return generic_prompts.build_generic_batch_review_messages(pairs, lang)
    if stage == "judge":
        texts = [it.get("candidate") or it.get("text")
                 or it.get("normalized_text") or "" for it in items]
        if lang == "ar":
            return prompts.build_batch_judge_messages(texts)
        return generic_prompts.build_generic_batch_judge_messages(texts, lang)
    texts = [it.get("normalized_text") or it.get("text") or "" for it in items]
    if lang == "ar":
        return prompts.build_batch_cleaner_messages(texts)
    return generic_prompts.build_generic_batch_cleaner_messages(texts, lang)


def parse_batch_response(content: str, n: int, *,
                         verdict_key: str = "review",
                         verdict_required: bool = False,
                         require_text: bool = True) -> tuple[dict[int, dict], set[int]]:
    """Strictly parse one batch response into (valid by index, unresolved).

    Enforces: items sit in a JSON array (optionally under ``"items"``); ``i``
    is a unique int in ``[0, n)``; ``text`` is a nonempty string unless
    ``require_text=False`` (the judge stage returns verdicts only, so its
    responses carry no text); the verdict named by ``verdict_key``
    (``review`` for the review stage, ``keep`` for the judge stage) is a
    literal boolean when present -- and must be present when
    ``verdict_required`` is set, so a judge cannot stay silent on an item.
    The truncated-array salvage in ``llm_client`` is deliberately NOT used
    for verdicts: a fabricated or duplicated index is an unresolved item for
    the split-retry path, never accepted data.
    """
    from .llm_client import _THINK_BLOCK_RE, parse_json_response

    cleaned = _THINK_BLOCK_RE.sub("", content).strip()
    try:
        data = parse_json_response(cleaned)
    except Exception:
        return {}, set(range(n))
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list):
        return {}, set(range(n))
    valid: dict[int, dict] = {}
    for raw in data:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("i", raw.get("index"))
        if not isinstance(idx, int) or isinstance(idx, bool) or not 0 <= idx < n \
                or idx in valid:
            continue
        text = raw.get("text")
        if require_text and (not isinstance(text, str) or not text.strip()):
            continue
        verdict = raw.get(verdict_key, None)
        if verdict is None:
            if verdict_required:
                continue
        elif not isinstance(verdict, bool):
            continue
        valid[idx] = raw
    return valid, set(range(n)) - set(valid)


class CorrectionScheduler:
    """Runs correction/review batches continuously at a fixed request budget."""

    def __init__(self, config: SchedulerConfig, *,
                 transport: httpx.BaseTransport | None = None,
                 event_callback: Callable[[dict], None] | None = None,
                 progress_callback: Callable[[dict], None] | None = None) -> None:
        self.cfg = config
        self._transport = transport
        self._event = event_callback or (lambda ev: None)
        self._progress = progress_callback or (lambda p: None)
        self._client: httpx.Client | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._sem: threading.Semaphore | None = None
        self._lock = threading.Lock()
        self.submitted = 0
        self.ok_items = 0
        self.failed_items = 0
        self.requests = 0
        self._last_progress = time.monotonic()

    # -- lifecycle -----------------------------------------------------------
    def __enter__(self) -> CorrectionScheduler:
        self._client = httpx.Client(
            base_url=self.cfg.url,
            timeout=httpx.Timeout(self.cfg.timeout, connect=10.0),
            limits=httpx.Limits(
                max_connections=self.cfg.concurrency,
                max_keepalive_connections=self.cfg.concurrency,
            ),
            transport=self._transport,
        )
        self._pool = ThreadPoolExecutor(max_workers=self.cfg.concurrency)
        self._sem = threading.Semaphore(self.cfg.concurrency)
        return self

    def __exit__(self, *exc) -> None:
        if self._pool:
            self._pool.shutdown(wait=True)
        if self._client:
            self._client.close()

    # -- public API ------------------------------------------------------------
    def process(self, items: list[dict], *, lang: str, stage: str = "correct") -> list[dict]:
        """Correct/review every item; return one result dict per input item.

        Items are packed into ``batch_size`` requests submitted concurrently
        through the shared pool (the semaphore caps aggregate in-flight
        requests). Results carry ``index``, ``ok`` and, on success, the
        response payload; exhausted items carry ``reason`` and are the
        caller's quarantine input.
        """
        assert self._pool
        results: dict[int, dict] = {}
        futures = []
        for lo in range(0, len(items), self.cfg.batch_size):
            chunk = items[lo:lo + self.cfg.batch_size]
            indices = list(range(lo, lo + len(chunk)))
            futures.append(self._pool.submit(
                self._run, chunk, indices, lang, stage, results))
        for fut in futures:
            fut.result()
        out = [results.get(i, {"index": i, "ok": False, "reason": "internal"})
               for i in range(len(items))]
        self._progress(self.snapshot())
        return out

    # -- batch machinery ---------------------------------------------------------
    def _run(self, items: list[dict], indices: list[int], lang: str, stage: str,
             results: dict[int, dict]) -> None:
        """Try one batch; unresolved items split BOTH halves recursively.

        ``items[k]`` corresponds to output position ``indices[k]``; splitting
        only ever partitions both lists in lockstep, so results always land on
        the right position.
        """
        n = len(items)
        with self._lock:
            self.submitted += n
        payload = self._post(messages=_messages(lang, stage, items))
        if payload is None:  # transport exhausted or truncated response
            self._split(items, indices, lang, stage, results)
            return
        judge = stage == "judge"
        valid, unresolved = parse_batch_response(
            payload.get("content", ""), n,
            verdict_key="keep" if judge else "review",
            verdict_required=judge, require_text=not judge)
        for k, raw in valid.items():
            idx, item = indices[k], items[k]
            if judge:
                if raw.get("keep") is not True:
                    results[idx] = {"index": idx, "ok": False,
                                    "reason": "judge_rejected",
                                    "issues": raw.get("issues")}
                    with self._lock:
                        self.failed_items += 1
                else:
                    # The judge returns a verdict, not a rewrite: the kept
                    # text is the final text it was shown (same fallback
                    # chain as the judge prompt builder).
                    results[idx] = {"index": idx, "ok": True,
                                    "text": item.get("candidate")
                                    or item.get("text")
                                    or item.get("normalized_text") or ""}
                    with self._lock:
                        self.ok_items += 1
            elif stage == "review" and raw.get("review") is not True:
                results[idx] = {"index": idx, "ok": False, "reason": "review_failed",
                                "issues": raw.get("issues")}
                with self._lock:
                    self.failed_items += 1
            else:
                results[idx] = {"index": idx, "ok": True, "text": raw["text"],
                                "confidence": raw.get("confidence"),
                                "dialect": raw.get("dialect")}
                with self._lock:
                    self.ok_items += 1
            self._event({"type": "stage_result", "id": item["id"], "stage": stage,
                         "data": results[idx]})
        retry = sorted(unresolved)
        if retry:
            self._split([items[k] for k in retry], [indices[k] for k in retry],
                        lang, stage, results)

    def _split(self, items: list[dict], indices: list[int], lang: str, stage: str,
               results: dict[int, dict]) -> None:
        """Halve a failed batch and retry both halves down to single items.

        A single unresolved item is terminal for this call: it returns an
        ``ok=False`` result (quarantine input) — never a silent fallback.
        """
        if len(items) == 1:
            idx = indices[0]
            results[idx] = {"index": idx, "ok": False, "reason": "exhausted"}
            with self._lock:
                self.failed_items += 1
            self._event({"type": "stage_result", "id": items[0]["id"], "stage": stage,
                         "data": results[idx]})
            return
        half = len(items) // 2
        for lo, hi in ((0, half), (half, len(items))):
            self._run(items[lo:hi], indices[lo:hi], lang, stage, results)

    # -- transport ----------------------------------------------------------------
    def _post(self, *, messages: list[dict[str, str]]) -> dict | None:
        """One chat completion under the concurrency semaphore.

        Returns ``{"content", "finish_reason"}`` or None when all transport
        attempts failed or the response came back truncated. Transient
        statuses retry with jittered exponential backoff.
        """
        assert self._sem and self._client
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "max_tokens": self.cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with self._sem:
            last: Exception | None = None
            for attempt in range(self.cfg.max_transport_attempts):
                try:
                    with self._lock:
                        self.requests += 1
                    resp = self._client.post("/chat/completions", json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        choice = data["choices"][0]
                        if choice.get("finish_reason") == "length":
                            # Truncated: treat as a batch failure so the
                            # split path shrinks the request instead of
                            # salvaging or retrying an identical oversized one.
                            LOGGER.warning(
                                "response truncated at max_tokens; splitting batch")
                            return None
                        self._tick()
                        return {"content": choice["message"]["content"],
                                "finish_reason": choice.get("finish_reason")}
                    if resp.status_code in (429, 500, 502, 503):
                        last = RuntimeError(f"HTTP {resp.status_code}")
                    else:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except (httpx.TimeoutException, httpx.ConnectError,
                        httpx.ReadError, RuntimeError) as exc:
                    last = exc
                time.sleep(min(30.0, 0.5 * (2 ** attempt) * (1.0 + random.random())))
            LOGGER.warning("batch transport failed after %d attempts: %s",
                           self.cfg.max_transport_attempts, last)
            return None

    def _tick(self) -> None:
        now = time.monotonic()
        if now - self._last_progress < self.cfg.progress_interval:
            return
        self._last_progress = now
        self._progress(self.snapshot())

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "submitted_items": self.submitted,
                "ok_items": self.ok_items,
                "failed_items": self.failed_items,
                "requests": self.requests,
            }
