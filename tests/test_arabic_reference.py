"""Provenance, inference-resume and review gates for benchmark reference curation."""

from __future__ import annotations

import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from data_processing.arabic_reference.core import (
    DATASETS,
    digest,
    load_plan,
    prepare,
    read_jsonl,
    sha256,
    write_jsonl,
)
from data_processing.arabic_reference.prompts import messages, parse_proposal
from data_processing.arabic_reference.review import import_reviews, review_pack, score
from data_processing.arabic_reference.runtime import doctor, propose


@pytest.fixture
def inputs(tmp_path):
    rows = []
    for ds in DATASETS:
        for ordinal in range(3):
            audio = tmp_path / "audio" / ds / f"{ordinal:06d}.npy"
            audio.parent.mkdir(parents=True, exist_ok=True)
            np.save(
                audio, np.full(1600 + ordinal, 0.01 * (DATASETS.index(ds) + 1), dtype=np.float32)
            )
            rows.append(
                {
                    "id": f"{ds}:{ordinal:06d}",
                    "dataset": ds,
                    "ordinal": ordinal,
                    "text": "مرحبا بكم",
                    "pred_text": "مرحبًا بكم.",
                    "mode": "arabic",
                    "audio": str(audio),
                    "audio_sha256": sha256(audio),
                    "preparation_id": digest([ds, ordinal]),
                    "actual_duration": (1600 + ordinal) / 16000,
                    "generation_limit_hit": False,
                    "metadata": {},
                    "source": {},
                }
            )
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_jsonl(a, rows)
    write_jsonl(b, rows)
    return [("one", a), ("two", b)]


@pytest.fixture
def plan(tmp_path, inputs):
    root = tmp_path / "plan"
    prepare(inputs, root, per_dataset=1)
    return root


def proposal_value():
    return {
        "proposed_verified_text": "مرحبا بكم",
        "proposed_ideal_text": "مرحبًا بكم.",
        "changes": [],
        "uncertainties": ["Requires listening"],
        "reason": "Text-only proposal",
    }


def human_review(row, decision="verified"):
    return {
        **{k: row[k] for k in ("id", "binding", "policy_sha256", "audio_sha256")},
        "reviewer_id": "reviewer-1",
        "audio_listened": True,
        "decision": decision,
        "verified_text": "مرحبا بكم" if decision == "verified" else None,
        "ideal_text": "مرحبًا بكم." if decision == "verified" else None,
        "reason": "Listened and compared wording",
    }


def test_prepare_balanced_deterministic_and_preserves_original(plan, inputs, tmp_path):
    original_hash = sha256(inputs[0][1])
    metadata, rows = load_plan(plan)
    assert Counter(r["dataset"] for r in rows) == Counter(dict.fromkeys(DATASETS, 1))
    assert all(
        r["text"] == "مرحبا بكم" and r["verified_text"] is None and r["ideal_text"] is None
        for r in rows
    )
    assert all(r["purpose"] == "evaluation_only" and r["split"] == "test" for r in rows)
    assert prepare(inputs, plan, per_dataset=1) == metadata
    assert sha256(inputs[0][1]) == original_hash
    reversed_file = tmp_path / "reversed.jsonl"
    write_jsonl(reversed_file, reversed(list(read_jsonl(inputs[0][1]))))
    prepare([("one", reversed_file), inputs[1]], tmp_path / "other", per_dataset=1)
    assert [r["id"] for r in load_plan(tmp_path / "other")[1]] == [r["id"] for r in rows]


@pytest.mark.parametrize("mutation", ["reference", "audio", "coverage", "duplicate", "mode"])
def test_candidate_mismatches_fail(inputs, tmp_path, mutation):
    rows = list(read_jsonl(inputs[1][1]))
    if mutation == "reference":
        rows[0]["text"] = "مختلف"
    if mutation == "audio":
        rows[0]["audio_sha256"] = "0" * 64
    if mutation == "coverage":
        rows.pop()
    if mutation == "duplicate":
        rows.append(rows[0])
    if mutation == "mode":
        rows[0]["mode"] = "auto"
    write_jsonl(inputs[1][1], rows)
    with pytest.raises(ValueError):
        prepare(inputs, tmp_path / "bad", per_dataset=1)


def test_diagnostics_are_separate_and_sampling_ignores_predictions(inputs, plan, tmp_path):
    prior = {r["id"] for r in load_plan(plan)[1]}
    rows = list(read_jsonl(inputs[0][1]))
    target = next(r for r in rows if r["id"] not in prior)
    target["pred_text"] = "English text"
    target["generation_limit_hit"] = True
    write_jsonl(inputs[0][1], rows)
    metadata = prepare(inputs, tmp_path / "flags", per_dataset=1)
    after = load_plan(tmp_path / "flags")[1]
    assert {r["id"] for r in after if "pilot" in r["cohorts"]} == prior
    assert next(r for r in after if r["id"] == target["id"])["cohorts"] == ["diagnostic"]
    assert metadata["diagnostic_clips"] == 1


def test_changed_manifest_and_changed_inputs_cannot_resume(plan, inputs):
    with pytest.raises(ValueError, match="differs"):
        prepare(inputs, plan, per_dataset=2)
    rows = list(read_jsonl(plan / "manifest.jsonl"))
    rows[0]["text"] = "tamper"
    write_jsonl(plan / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="changed"):
        load_plan(plan)


def test_blinding_and_long_candidate_bound(plan):
    row = load_plan(plan)[1][0]
    row["candidates"][0]["name"] = "PRIVATE_MODEL_NAME"
    row["candidates"][0]["text"] = "نحن " * 2000
    content = messages(row)[1]["content"]
    assert "PRIVATE_MODEL_NAME" not in content
    assert "original_characters" in content and len(content) < 5000


@pytest.mark.parametrize(
    "mutation", ["length", "verification", "number", "negation", "null_pair", "missing_reason"]
)
def test_invalid_proposals_rejected(mutation):
    value = proposal_value()
    finish = "stop"
    if mutation == "length":
        finish = "length"
    if mutation == "verification":
        value["review_status"] = "verified"
    if mutation == "number":
        value.update(proposed_verified_text="2400", proposed_ideal_text="2250")
    if mutation == "negation":
        value.update(proposed_verified_text="لا تذهب", proposed_ideal_text="تذهب")
    if mutation == "null_pair":
        value["proposed_ideal_text"] = None
    if mutation == "missing_reason":
        value["reason"] = ""
    with pytest.raises(ValueError):
        parse_proposal(json.dumps(value), finish)


def test_unresolved_proposal_and_safe_formatting():
    value = proposal_value()
    assert parse_proposal(json.dumps(value), "stop") == value
    value.update(proposed_verified_text=None, proposed_ideal_text=None)
    assert parse_proposal(json.dumps(value), "stop")["proposed_verified_text"] is None


@pytest.fixture
def server():
    state = {"calls": 0, "fail": False}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            assert self.path == "/v1/models"
            self.send(
                {"data": [{"id": "corrector", "root": "example/corrector", "max_model_len": 16384}]}
            )

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["temperature"] == 0 and not body["chat_template_kwargs"]["enable_thinking"]
            state["calls"] += 1
            self.send(
                {
                    "choices": [
                        {
                            "message": {"content": json.dumps(proposal_value())},
                            "finish_reason": "length" if state["fail"] else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "prompt_tokens_details": None,
                    },
                }
            )

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", state
    httpd.shutdown()
    httpd.server_close()
    thread.join()


def test_real_http_resume_and_configuration_binding(plan, tmp_path, server):
    url, state = server
    kwargs = {
        "url": url,
        "model": "corrector",
        "checkpoint": "example/corrector",
        "concurrency": 2,
        "emit": lambda _: None,
    }
    result = propose(plan, tmp_path / "run", **kwargs)
    assert result["counts"] == {"proposed": 6} and result["verified"] == 0
    assert state["calls"] == 6
    assert propose(plan, tmp_path / "run", **kwargs)["resumed"] == 6
    assert state["calls"] == 6
    with pytest.raises(ValueError, match="changed"):
        propose(plan, tmp_path / "run", max_tokens=3072, **kwargs)
    records = list(read_jsonl(tmp_path / "run/proposals.jsonl"))
    assert all(
        r["verified_text"] is None
        and r["ideal_text"] is None
        and not r["audio_supplied_to_corrector"]
        for r in records
    )


def test_failed_responses_are_checkpointed_and_retried(plan, tmp_path, server):
    url, state = server
    kwargs = {
        "url": url,
        "model": "corrector",
        "checkpoint": "example/corrector",
        "emit": lambda _: None,
    }
    state["fail"] = True
    assert propose(plan, tmp_path / "run", **kwargs)["counts"] == {"failed": 6}
    state["fail"] = False
    assert propose(plan, tmp_path / "run", **kwargs)["counts"] == {"proposed": 6}
    assert state["calls"] == 12
    with pytest.raises(ValueError, match="Server advertises"):
        doctor(url, "corrector", "wrong/checkpoint")


def test_audio_pack_checks_audio_and_keeps_candidate_text_inert(plan, tmp_path):
    output = tmp_path / "listening"
    result = review_pack(plan, output)
    assert result["audio_exported"]
    jobs = json.loads((output / "jobs.json").read_text())["jobs"]
    assert len(jobs) == 6
    assert all(sha256(output / j["audio_url"]) == j["review_audio_sha256"] for j in jobs)
    assert "__JOBS_JSON__" not in (output / "index.html").read_text()
    row = load_plan(plan)[1][0]
    from pathlib import Path

    Path(row["audio"]).write_bytes(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        review_pack(plan, tmp_path / "bad-audio")
    assert not review_pack(plan, tmp_path / "metadata", metadata_only=True)["audio_exported"]


@pytest.mark.parametrize(
    "mutation",
    [
        "no_audio",
        "wrong_hash",
        "wrong_binding",
        "format_changes_word",
        "missing_reason",
        "duplicate",
    ],
)
def test_review_evidence_gates(plan, tmp_path, mutation):
    review = human_review(load_plan(plan)[1][0])
    if mutation == "no_audio":
        review["audio_listened"] = False
    if mutation == "wrong_hash":
        review["audio_sha256"] = "0" * 64
    if mutation == "wrong_binding":
        review["binding"] = "0" * 64
    if mutation == "format_changes_word":
        review["ideal_text"] = "اذهب"
    if mutation == "missing_reason":
        review["reason"] = ""
    rows = [review, review] if mutation == "duplicate" else [review]
    path = tmp_path / "reviews.jsonl"
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        import_reviews(plan, [path], tmp_path / "reviewed.jsonl")


def test_paired_scoring_partial_coverage_and_no_unreviewed_gold(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    rows = load_plan(plan)[1]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(rows[0])])
    report = import_reviews(plan, [path], reviewed)
    assert report["statuses"] == {"verified": 1, "pending": 5}
    scores = score(plan, reviewed, tmp_path / "scores.json")
    assert scores["verified_clips"] == 1 and scores["pending_or_unresolved"] == 5
    assert scores["scores"]["pilot/one/verified_elm/macro"]["wer"] is None
    assert scores["scores"]["pilot/one/original_elm/casablanca"]["utterances"] == 1
    # Altering text and recomputing its outer checksum still fails the review binding.
    changed = list(read_jsonl(reviewed))
    changed[0]["verified_text"] = "anything"
    changed[0]["reviewed_sha256"] = digest(
        {k: v for k, v in changed[0].items() if k != "reviewed_sha256"}
    )
    write_jsonl(reviewed, changed)
    with pytest.raises(ValueError, match="attestation"):
        score(plan, reviewed, tmp_path / "bad.json")


def test_unresolved_never_scored_and_versions_immutable(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    rows = load_plan(plan)[1]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(rows[0], "unresolved")])
    import_reviews(plan, [path], reviewed)
    with pytest.raises(ValueError, match="No audio-reviewed"):
        score(plan, reviewed, tmp_path / "bad.json")
    write_jsonl(path, [human_review(rows[0])])
    with pytest.raises(ValueError, match="already exists"):
        import_reviews(plan, [path], reviewed)


def test_duplicate_audio_requires_consistent_review(inputs, tmp_path):
    for _, source in inputs:
        rows = list(read_jsonl(source))
        first, second = rows[0], rows[1]
        second.update(
            audio=first["audio"],
            audio_sha256=first["audio_sha256"],
            actual_duration=first["actual_duration"],
        )
        write_jsonl(source, rows)
    root = tmp_path / "duplicate-plan"
    prepare(inputs, root, per_dataset=1)
    rows = [r for r in load_plan(root)[1] if "duplicate_audio" in r["flags"]]
    assert len(rows) == 2
    reviews = [human_review(r) for r in rows]
    reviews[1].update(verified_text="نص مختلف", ideal_text="نص مختلف.")
    path = tmp_path / "reviews.jsonl"
    write_jsonl(path, reviews)
    with pytest.raises(ValueError, match="Conflicting verified"):
        import_reviews(root, [path], tmp_path / "conflict.jsonl")


def test_verified_six_dataset_macro_uses_same_subset(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    rows = load_plan(plan)[1]
    path = tmp_path / "reviews.jsonl"
    write_jsonl(path, [human_review(r) for r in rows])
    reviewed = tmp_path / "reviewed.jsonl"
    import_reviews(plan, [path], reviewed)
    result = score(plan, reviewed, tmp_path / "scores.json")
    for track in ["original_elm", "verified_elm"]:
        assert result["scores"][f"pilot/one/{track}/macro"] == {
            "datasets": 6,
            "full_suite": False,
            "wer": 0.0,
            "cer": 0.0,
        }
    assert result["verified_clips"] == result["selected_clips"] == 6


def test_launcher_uses_explicit_corrector_not_training_tuning(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "python"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    monkeypatch.setenv("WORKERS_PER_GPU", "8")
    monkeypatch.setenv("BATCH_SIZE", "16")
    env = {
        **os.environ,
        "PYTHON": str(binary),
        "ARABIC_REF_ROOT": str(tmp_path / "out"),
        "LLM_MODEL": "example/corrector",
        "LLM_URL": "http://localhost:8010",
        "LLM_MODEL_NAME": "corrector",
        "REF_CONCURRENCY": "12",
    }
    proc = subprocess.run(
        ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), "smoke"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(proc.stdout)
    assert "--smoke" in args
    assert args[args.index("--checkpoint") + 1] == "example/corrector"
    assert args[args.index("--concurrency") + 1] == "12"
    assert args[args.index("--output") + 1] == str(tmp_path / "out/smoke")
    assert "--batch-size" not in args and "--workers-per-gpu" not in args
    # An isolated code checkout can reuse the original checkout's data and environment.
    env.pop("ARABIC_REF_ROOT")
    env["ARABIC_REF_SOURCE_REPO"] = str(tmp_path / "source repo")
    proc = subprocess.run(
        ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), "plan"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(proc.stdout)
    assert args[args.index("--output") + 1] == str(
        tmp_path / "source repo/verification/arabic-reference-pilot-v1/plan"
    )
    assert (
        "qasr="
        + str(tmp_path / "source repo/verification/qasr-stage02-arabic/predictions-arabic.jsonl")
        in args
    )
