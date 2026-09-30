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
    run_vet_slice,
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


def _judge_transport(keep):
    """Mock judge: one keep verdict per transcript line, decided by predicate."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        texts = [ln.split("<<<")[1].split(">>>")[0]
                 for ln in body["messages"][1]["content"].splitlines()
                 if "<<<" in ln]
        items = [{"i": i, "keep": bool(keep(t)),
                  "issues": [] if keep(t) else ["degenerate repetition"]}
                 for i, t in enumerate(texts)]
        return httpx.Response(200, json={"choices": [{
            "message": {"content": json.dumps(items), "finish_reason": "stop"}}]})

    return httpx.MockTransport(handler)


def _run_slice(root, st, out):
    return run_exhaustive_slice(root, lang="en", part=0, run_id="test",
                                url="http://t/v1", concurrency=2, batch_size=2,
                                state_dir=st, output_dir=out,
                                transport=_echo_transport())


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


def test_vet_filters_judge_rejected_rows(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence {i}", 2, quality=0.9)
            for i in range(4)]
    root = _build_root(tmp_path, rows)
    st, out = tmp_path / "st", tmp_path / "out"
    _run_slice(root, st, out)
    reject_ones = _judge_transport(lambda t: "1" not in t)
    stats = run_vet_slice(root, lang="en", part=0, run_id="test",
                          url="http://t/v1", concurrency=2, batch_size=2,
                          state_dir=tmp_path / "stv", output_dir=tmp_path / "outv",
                          stage2_state_dir=st, transport=reject_ones)
    assert stats["accepted_published"] == 3
    assert stats["rejected_published"] == 1
    # Vetted manifests hold only the kept rows.
    manifest = next((tmp_path / "outv" / "segments").glob("*.manifest.jsonl"))
    texts = sorted(json.loads(ln)["text"] for ln in
                   manifest.read_text().splitlines() if ln.strip())
    assert texts == sorted(f"SENTENCE {i}" for i in (0, 2, 3))
    # Rejections are recorded with their issues.
    rej = list((tmp_path / "outv" / "segments").glob("*.rejected.jsonl"))
    assert len(rej) == 1
    rejected = [json.loads(ln) for ln in
                rej[0].read_text().splitlines() if ln.strip()]
    assert len(rejected) == 1 and rejected[0]["status"] == "rejected"
    assert rejected[0]["reason"] == "judge_rejected"
    # Re-running the vet pass replays (including the rejected file) and is
    # a no-op.
    stats = run_vet_slice(root, lang="en", part=0, run_id="test",
                          url="http://t/v1", concurrency=2, batch_size=2,
                          state_dir=tmp_path / "stv", output_dir=tmp_path / "outv",
                          stage2_state_dir=st, transport=reject_ones)
    assert stats["accepted_published"] == 3
    assert stats["rejected_published"] == 1
    code, report = audit_run(root, state_root=st, vet_state_root=tmp_path / "stv")
    assert code == EXIT_OK
    assert report["vetted"] is True
    assert report["languages"]["en"]["accepted"] == 3
    assert report["languages"]["en"]["rejected"] == 1
    assert report["languages"]["en"]["exact"] is True


def test_vet_quarantines_when_judge_unreachable(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence {i}", 2, quality=0.9)
            for i in range(4)]
    root = _build_root(tmp_path, rows)
    st, out = tmp_path / "st", tmp_path / "out"
    _run_slice(root, st, out)
    dead = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    stats = run_vet_slice(root, lang="en", part=0, run_id="test",
                          url="http://t/v1", concurrency=2, batch_size=2,
                          state_dir=tmp_path / "stv", output_dir=tmp_path / "outv",
                          stage2_state_dir=st, transport=dead)
    # Unjudgeable rows quarantine (retryable), never silently kept/rejected.
    assert stats["quarantine_published"] == 4
    assert stats["accepted_published"] == 0
    code, report = audit_run(root, state_root=st, vet_state_root=tmp_path / "stv")
    assert code == EXIT_QUARANTINE
    assert report["languages"]["en"]["quarantined"] == 4
    assert report["languages"]["en"]["exact"] is True


def test_audit_without_vet_dbs_unchanged(tmp_path):
    rows = [_row(f"/audio/{i}.wav", f"sentence {i}", 2, quality=0.9)
            for i in range(3)]
    root = _build_root(tmp_path, rows)
    st, out = tmp_path / "st", tmp_path / "out"
    _run_slice(root, st, out)
    code, report = audit_run(root, state_root=st)
    assert code == EXIT_OK
    assert report["vetted"] is False
    assert report["languages"]["en"]["accepted"] == 3
    assert report["languages"]["en"]["rejected"] == 0


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
