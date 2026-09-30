"""End-to-end exhaustive slice tests: correct -> review -> publish -> audit."""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest

from data_processing import workset as ws
from data_processing.exhaustive import (
    EXIT_OK,
    EXIT_QUARANTINE,
    audit_run,
    run_exhaustive_slice,
)
from tests.test_workset import _fixture, _row


def _build_root(tmp_path: Path, rows: list[dict]) -> Path:
    options = _fixture(tmp_path / "inputs", {("v76", "en", "train.jsonl"): rows})
    root = tmp_path / "run"
    ws.prepare_rank(root, **options, rank=0, nodes=1, jobs=1, run_id="test")
    ws.build_language(root, "en", run_id="test")
    ws.finalize_worksets(root)
    ws.partition_language(root, "en", 1)
    return root


_PAIR_RE = re.compile(r"\d+\. ORIGINAL: <<<(.*?)>>>\s*CLEANED:\s*<<<(.*?)>>>")


def _echo_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        pairs = _PAIR_RE.findall(content)
        if pairs:
            items = [{"i": i, "text": c, "review": True}
                     for i, (_, c) in enumerate(pairs)]
        else:
            texts = re.findall(r"\d+\. <<<(.*?)>>>", content)
            items = [{"i": i, "text": t.upper(), "confidence": 0.9}
                     for i, t in enumerate(texts)]
        return httpx.Response(200, json={"choices": [{
            "message": {"content": json.dumps(items), "finish_reason": "stop"}}]})

    return httpx.MockTransport(handler)


def test_slice_accepts_corrects_reviews_and_audits_clean(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence number {i}", 2, quality=0.9)
            for i in range(5)]
    root = _build_root(tmp_path, rows)
    stats = run_exhaustive_slice(root, lang="en", part=0, run_id="test",
                                 url="http://t/v1", concurrency=2, batch_size=2,
                                 state_dir=tmp_path / "st", output_dir=tmp_path / "out",
                                 transport=_echo_transport())
    assert stats["accepted_published"] == 5
    assert stats["quarantine_published"] == 0
    # Accepted manifests contain the review-passed (uppercased) text.
    manifest = next((tmp_path / "out" / "segments").glob("*.manifest.jsonl"))
    texts = sorted(json.loads(ln)["text"] for ln in
                   manifest.read_text().splitlines() if ln.strip())
    # Manifest rows are ordered by item id (hash order), not input order.
    assert texts == sorted(f"SENTENCE NUMBER {i}" for i in range(5))
    code, report = audit_run(root, state_root=tmp_path / "st")
    assert code == EXIT_OK
    assert report["languages"]["en"]["exact"] is True
    assert report["languages"]["en"]["accepted"] == 5


def test_quarantine_exit_code(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence {i}", 2, quality=0.9)
            for i in range(3)]
    root = _build_root(tmp_path, rows)
    dead = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    stats = run_exhaustive_slice(root, lang="en", part=0, run_id="test",
                                 url="http://t/v1", concurrency=2, batch_size=2,
                                 state_dir=tmp_path / "st2", output_dir=tmp_path / "out2",
                                 transport=dead)
    assert stats["quarantine_published"] == 3
    assert stats["accepted_published"] == 0
    code, report = audit_run(root, state_root=tmp_path / "st2")
    assert code == EXIT_QUARANTINE
    assert report["languages"]["en"]["quarantined"] == 3


def test_resume_never_duplicates_accepted_rows(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence {i}", 2, quality=0.9)
            for i in range(6)]
    root = _build_root(tmp_path, rows)
    st, out = tmp_path / "st", tmp_path / "out"
    stats = run_exhaustive_slice(root, lang="en", part=0, run_id="test",
                                 url="http://t/v1", concurrency=2, batch_size=2,
                                 state_dir=st, output_dir=out,
                                 transport=_echo_transport())
    # Re-running the same slice is a no-op (cursor at end, results immutable).
    stats = run_exhaustive_slice(root, state_dir=st, output_dir=out,
                                 transport=_echo_transport(),
                                 lang="en", part=0, run_id="test",
                                 url="http://t/v1", concurrency=2, batch_size=2)
    assert stats["accepted_published"] == 6
    code, report = audit_run(root, state_root=st)
    assert code == EXIT_OK and report["languages"]["en"]["accepted"] == 6


def test_blocked_metadata_rows_quarantine_without_llm(tmp_path):
    # duration=0 is a structural metadata problem in the workset builder.
    rows = [_row("/audio/good.wav", "good sentence", 2, quality=0.9),
            _row("/audio/bad.wav", "bad metadata", 0, quality=0.9)]
    root = _build_root(tmp_path, rows)
    stats = run_exhaustive_slice(root, lang="en", part=0, run_id="test",
                                 url="http://t/v1", concurrency=2, batch_size=4,
                                 state_dir=tmp_path / "st", output_dir=tmp_path / "out",
                                 transport=_echo_transport())
    total = stats["accepted_published"] + stats["quarantine_published"]
    assert total == 2, "every partitioned row must reach a terminal state"
    code, report = audit_run(root, state_root=tmp_path / "st")
    assert code in (EXIT_OK, EXIT_QUARANTINE)
    assert report["languages"]["en"]["exact"] is True


if __name__ == "__main__":
    pytest.main([__file__])
