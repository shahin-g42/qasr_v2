"""Tests for the strict continuous correction scheduler."""

from __future__ import annotations

import json
import threading
import unittest

import httpx

from data_processing.correction_scheduler import (
    CorrectionScheduler,
    SchedulerConfig,
    parse_batch_response,
)


def _items(n: int) -> list[dict]:
    return [{"id": f"id{i}", "audio_filepath": f"{i}.wav", "lang": "en",
             "normalized_text": f"line {i}"} for i in range(n)]


def _reply(items: list[dict[str, object]]) -> str:
    return json.dumps(items)


def _ok_transport(n_seen: list[int], *, fail_batches: set[int] | None = None,
                  truncate_batches: set[int] | None = None):
    """Mock server echoing one corrected item per input line, 0-based."""
    lock = threading.Lock()
    fail_batches = fail_batches or set()
    truncate_batches = truncate_batches or set()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = [ln.split("<<<")[1].split(">>>")[0]
                 for ln in body["messages"][1]["content"].splitlines()
                 if "<<<" in ln]
        with lock:
            batch_no = len(n_seen)
            n_seen.append(len(texts))
        if batch_no in fail_batches:
            return httpx.Response(500, text="boom")
        if batch_no in truncate_batches:
            return httpx.Response(200, json={"choices": [{
                "message": {"content": _reply([
                    {"i": i, "text": t.upper()} for i, t in enumerate(texts)][:1]),
                    "finish_reason": "length"}}]})
        return httpx.Response(200, json={"choices": [{
            "message": {"content": _reply([
                {"i": i, "text": t.upper(), "confidence": 0.9} for i, t in enumerate(texts)]),
                "finish_reason": "stop"}}]})

    return httpx.MockTransport(handler), n_seen


class TestParseBatchResponse(unittest.TestCase):
    def test_valid_items_parse(self) -> None:
        valid, unresolved = parse_batch_response(
            _reply([{"i": 1, "text": "b"}, {"i": 0, "text": "a"}]), 2)
        self.assertEqual(set(valid), {0, 1})
        self.assertEqual(unresolved, set())

    def test_items_envelope_accepted(self) -> None:
        valid, _ = parse_batch_response(
            json.dumps({"items": [{"i": 0, "text": "a"}]}), 1)
        self.assertEqual(valid[0]["text"], "a")

    def test_bad_indices_and_text_are_unresolved(self) -> None:
        content = _reply([{"i": 0, "text": ""}, {"i": 5, "text": "x"},
                          {"i": 1, "text": "ok"}, {"i": 1, "text": "dup"},
                          {"i": "0", "text": "stridx"}, {"i": True, "text": "t"}])
        valid, unresolved = parse_batch_response(content, 3)
        self.assertEqual(set(valid), {1})
        self.assertEqual(unresolved, {0, 2})

    def test_non_boolean_review_is_unresolved(self) -> None:
        # A non-literal verdict is rejected by the parser itself, not just by
        # the caller: silent truthiness coercion is exactly what we forbid.
        valid, unresolved = parse_batch_response(
            _reply([{"i": 0, "text": "a", "review": "yes"},
                    {"i": 1, "text": "b", "review": False}]), 2)
        self.assertEqual(set(valid), {1})
        self.assertEqual(unresolved, {0})
        self.assertTrue(valid[1]["review"] is False)

    def test_unparseable_content_unresolves_everything(self) -> None:
        valid, unresolved = parse_batch_response("complete garbage", 3)
        self.assertEqual((valid, unresolved), ({}, {0, 1, 2}))

    def test_truncated_prefix_never_fabricates(self) -> None:
        # The generic salvage is deliberately absent: a cut array yields only
        # what strictly validated.
        content = '[{"i": 0, "text": "a"}, {"i": 1, "text": "b"'
        valid, unresolved = parse_batch_response(content, 2)
        self.assertEqual(set(valid), {0})
        self.assertEqual(unresolved, {1})

    def test_judge_verdict_required(self) -> None:
        # The judge must answer every item: a missing keep would silently
        # read as a rejection, which is exactly the coercion we forbid.
        valid, unresolved = parse_batch_response(
            _reply([{"i": 0, "keep": True}, {"i": 1}]), 2,
            verdict_key="keep", verdict_required=True, require_text=False)
        self.assertEqual(set(valid), {0})
        self.assertEqual(unresolved, {1})

    def test_judge_non_boolean_keep_is_unresolved(self) -> None:
        valid, unresolved = parse_batch_response(
            _reply([{"i": 0, "keep": "yes"}]), 1,
            verdict_key="keep", verdict_required=True, require_text=False)
        self.assertEqual((valid, unresolved), ({}, {0}))


class TestScheduler(unittest.TestCase):
    def _cfg(self, **kw) -> SchedulerConfig:
        defaults: dict = {"url": "http://t/v1", "concurrency": 4,
                          "batch_size": 4, "max_transport_attempts": 2}
        defaults.update(kw)
        return SchedulerConfig(**defaults)

    def test_happy_path_all_items_corrected(self) -> None:
        seen: list[int] = []
        transport, seen = _ok_transport(seen)
        with CorrectionScheduler(self._cfg(), transport=transport) as s:
            out = s.process(_items(4), lang="en")
        self.assertTrue(all(r["ok"] for r in out))
        self.assertEqual([r["text"] for r in out],
                         [f"LINE {i}" for i in range(4)])
        self.assertEqual(s.snapshot()["ok_items"], 4)

    def test_transport_failure_splits_to_single_items(self) -> None:
        seen: list[int] = []
        transport, _ = _ok_transport(seen, fail_batches={0})
        # attempts=1: no in-request retry, so the failure must heal via the
        # split path (whole batch, then both halves).
        with CorrectionScheduler(self._cfg(max_transport_attempts=1),
                                 transport=transport) as s:
            out = s.process(_items(4), lang="en")
        # Batch of 4 failed -> halves (2,2); every item recovered.
        self.assertTrue(all(r["ok"] for r in out))
        self.assertGreaterEqual(len(seen), 3)

    def test_truncation_splits_and_recovers(self) -> None:
        seen: list[int] = []
        transport, _ = _ok_transport(seen, truncate_batches={0})
        with CorrectionScheduler(self._cfg(), transport=transport) as s:
            out = s.process(_items(4), lang="en")
        self.assertTrue(all(r["ok"] for r in out),
                        f"truncation must heal, got {out}")

    def test_single_unrecoverable_item_returns_quarantine_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="boom")
        with CorrectionScheduler(self._cfg(max_transport_attempts=1),
                                 transport=httpx.MockTransport(handler)) as s:
            out = s.process(_items(2), lang="en")
        self.assertEqual([r["ok"] for r in out], [False, False])
        self.assertEqual({r["reason"] for r in out}, {"exhausted"})
        self.assertEqual(s.snapshot()["failed_items"], 2)

    def test_review_stage_requires_literal_true(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            content = _reply([
                {"i": 0, "text": "a", "review": True},
                {"i": 1, "text": "b", "review": False, "issues": ["dropped word"]},
            ])
            return httpx.Response(200, json={"choices": [{
                "message": {"content": content, "finish_reason": "stop"}}]})
        with CorrectionScheduler(self._cfg(), transport=httpx.MockTransport(handler)) as s:
            out = s.process(_items(2), lang="en", stage="review")
        self.assertTrue(out[0]["ok"])
        self.assertFalse(out[1]["ok"])
        self.assertEqual(out[1]["reason"], "review_failed")
        self.assertEqual(out[1]["issues"], ["dropped word"])

    def test_judge_stage_verdicts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            content = _reply([
                {"i": 0, "keep": True},
                {"i": 1, "keep": False, "issues": ["degenerate repetition"]},
            ])
            return httpx.Response(200, json={"choices": [{
                "message": {"content": content, "finish_reason": "stop"}}]})
        with CorrectionScheduler(self._cfg(), transport=httpx.MockTransport(handler)) as s:
            out = s.process(_items(2), lang="en", stage="judge")
        self.assertTrue(out[0]["ok"])
        # The judge returns a verdict, not a rewrite: the kept text is the
        # final text it was shown.
        self.assertEqual(out[0]["text"], "line 0")
        self.assertFalse(out[1]["ok"])
        self.assertEqual(out[1]["reason"], "judge_rejected")
        self.assertEqual(out[1]["issues"], ["degenerate repetition"])

    def test_judge_silence_quarantines_neither_keeps_nor_rejects(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            # Verdicts omitted entirely: unresolved, split down to singles,
            # and finally exhausted -- never a silent rejection.
            return httpx.Response(200, json={"choices": [{
                "message": {"content": _reply([{"i": 0}]),
                             "finish_reason": "stop"}}]})
        with CorrectionScheduler(self._cfg(max_transport_attempts=1),
                                 transport=httpx.MockTransport(handler)) as s:
            out = s.process(_items(2), lang="en", stage="judge")
        self.assertEqual([r["ok"] for r in out], [False, False])
        self.assertEqual({r["reason"] for r in out}, {"exhausted"})

    def test_results_keep_positions_when_splitting(self) -> None:
        seen: list[int] = []
        # Fail only the FIRST batch (the whole 4) with truncation so the split
        # re-requests items 2..3 first; positions must stay aligned.
        transport, _ = _ok_transport(seen, truncate_batches={0})
        with CorrectionScheduler(self._cfg(), transport=transport) as s:
            out = s.process(_items(4), lang="en")
        self.assertEqual([r["index"] for r in out], [0, 1, 2, 3])
        self.assertEqual([r["text"] for r in out],
                         [f"LINE {i}" for i in range(4)])

    def test_events_and_progress_callbacks_fire(self) -> None:
        events: list[dict] = []
        transport, _ = _ok_transport([])
        with CorrectionScheduler(self._cfg(), transport=transport,
                                 event_callback=events.append) as s:
            s.process(_items(2), lang="en")
        self.assertEqual(len(events), 2)
        self.assertEqual({e["type"] for e in events}, {"stage_result"})

    def test_config_validation(self) -> None:
        for bad in ({"concurrency": 0}, {"batch_size": 0},
                    {"max_transport_attempts": 0}):
            with self.assertRaises(ValueError):
                SchedulerConfig(**bad)


if __name__ == "__main__":
    unittest.main()
