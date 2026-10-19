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
    format_signature,
    load_plan,
    prepare,
    read_jsonl,
    review_issues,
    sha256,
    write_jsonl,
)
from data_processing.arabic_reference.prompts import blinded, messages, parse_proposal
from data_processing.arabic_reference.review import import_reviews, review_pack, score
from data_processing.arabic_reference.runtime import check_smoke, doctor, propose, select_smoke


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


def proposal_row():
    return {
        "id": "sada:000000",
        "text": "مرحبا بكم",
        "candidates": [],
        "flags": [],
        "metadata": {},
        "actual_duration": 2.0,
    }


def proposal_value(row=None, payload=None):
    row = row or proposal_row()
    payload = payload or {"candidates": blinded(row), "review_issues": review_issues(row)}
    return {
        "base_candidate": payload["candidates"][0]["label"],
        "issue_notes": dict.fromkeys(
            payload["review_issues"], "Needs listening to resolve this issue"
        ),
        "proposed_verified_text": "مرحبا بكم",
        "proposed_ideal_text": "مرحبًا بكم.",
        "changes": [
            {
                "kind": "formatting",
                "before": "مرحبا بكم",
                "after": "مرحبًا بكم.",
                "reason": "Conservative punctuation and diacritics",
            }
        ],
        "uncertainties": ["Requires listening"],
        "reason": "Text-only proposal",
    }


def human_review(row, decision="verified"):
    return {
        **{k: row[k] for k in ("id", "binding", "policy_sha256", "audio_sha256")},
        "reviewer_id": "reviewer-1",
        "audio_listened": True,
        "ideal_content_preserved": decision == "verified",
        "decision": decision,
        "verified_text": "مرحبا بكم" if decision == "verified" else None,
        "ideal_text": "مرحبًا بكم." if decision == "verified" else None,
        "reason": "Listened and compared wording",
        "ideal_spelling_reason": "",
        "issue_resolutions": dict.fromkeys(review_issues(row), "Listened and resolved this issue"),
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
    if mutation in {"number", "negation"}:
        from data_processing.arabic_reference.prompts import proposal_checks

        row = proposal_row()
        changes = []
        draft = parse_proposal(json.dumps(value), finish, row, adjustments=changes)
        assert proposal_checks(draft, row, changes)["wording_to_ideal"]
        assert any(a["type"] == "metadata_warning" for a in changes)
    else:
        with pytest.raises(ValueError):
            parse_proposal(json.dumps(value), finish, proposal_row())


def test_unresolved_proposal_and_safe_formatting():
    value = proposal_value()
    assert parse_proposal(json.dumps(value), "stop", proposal_row()) == value
    value.update(
        base_candidate=None, proposed_verified_text=None, proposed_ideal_text=None, changes=[]
    )
    assert (
        parse_proposal(json.dumps(value), "stop", proposal_row())["proposed_verified_text"] is None
    )


@pytest.fixture
def server():
    state = {"calls": 0, "fail": False, "root": "example/corrector", "requests": []}

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
                {"data": [{"id": "corrector", "root": state["root"], "max_model_len": 16384}]}
            )

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert (
                body["temperature"] in {0, 0.2}
                and not body["chat_template_kwargs"]["enable_thinking"]
            )
            assert body["model"] == "corrector"
            state["calls"] += 1
            state["requests"].append(body)
            value = (
                state["reply"](body)
                if state.get("reply")
                else proposal_value(payload=json.loads(body["messages"][1]["content"]))
            )
            self.send(
                {
                    "choices": [
                        {
                            "message": {
                                "content": value if isinstance(value, str) else json.dumps(value)
                            },
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


@pytest.mark.parametrize("stage", ["doctor", "smoke", "run", "correct"])
def test_launcher_uses_explicit_corrector_not_training_tuning(tmp_path, monkeypatch, stage):
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
    env.pop("ARABIC_REF_CHECKPOINT", None)
    proc = subprocess.run(
        ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), stage],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(proc.stdout)
    assert ("--smoke" in args) == (stage == "smoke")
    assert ("--initial-cleanup" in args) == (stage == "correct")
    assert "--checkpoint" not in args
    assert args[args.index("--model") + 1] == "corrector"
    if stage != "doctor":
        assert args[args.index("--concurrency") + 1] == "12"
        destination = "smoke-v4" if stage == "smoke" else "proposals"
        assert args[args.index("--output") + 1] == str(tmp_path / "out" / destination)
    assert "--batch-size" not in args and "--workers-per-gpu" not in args
    # Only an explicit pilot checkpoint pin constrains the running service.
    env["ARABIC_REF_CHECKPOINT"] = "example/pinned-model"
    proc = subprocess.run(
        ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), "doctor"],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    args = json.loads(proc.stdout)
    assert args[args.index("--checkpoint") + 1] == "example/pinned-model"
    assert args[args.index("--model") + 1] == "corrector"
    env.pop("ARABIC_REF_CHECKPOINT")
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
        tmp_path / "source repo/verification/arabic-reference-pilot-v4/plan"
    )
    assert (
        "qasr="
        + str(tmp_path / "source repo/verification/qasr-stage02-arabic/predictions-arabic.jsonl")
        in args
    )


@pytest.mark.parametrize(
    "before,after",
    [
        ("3.09", "309"),
        ("- 5", "5"),
        (".5", "5"),
        ("https://example.org/Path", "https://example.org/path"),
        ("-5", "5"),
        ("2,400", "2400"),
        ("٥٫٢", "٥٢"),
        ("25%", "25"),
        ("2400", "ألفان وأربعمائة"),
        ("listener@example.org", "listenerexampleorg"),
        ("https://example.org/a", "https://example.org/b"),
        ("can't", "cant"),
        ("café", "cafe"),
        ("nowhere", "now here"),
        ("مافيش", "ما فيش"),
        ("الطالبات", "الطالبة"),
        ("رحمه", "رحمة"),
        ("على", "علي"),
        ("أحمد", "احمد"),
        ("آمال", "امال"),
        ("لا تذهب", "تذهب"),
        ("x²", "x2"),
    ],
)
def test_formatting_cannot_change_content_or_meaningful_punctuation(before, after):
    assert format_signature(before) != format_signature(after)
    value = proposal_value()
    value.update(proposed_verified_text=before, proposed_ideal_text=after)
    from data_processing.arabic_reference.prompts import proposal_checks

    row = proposal_row()
    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    checks = proposal_checks(parsed, row, adjustments)
    assert checks["wording_to_ideal"]
    assert checks["metadata_warnings"]  # A draft is retained, never silently treated as formatting.
    assert parsed["proposed_ideal_text"] == after


@pytest.mark.parametrize(
    "before,after",
    [
        ("مرحبا بكم", "مرحبًا بكم!"),
        ("لا تذهب", "لا، تَذْهَبْ."),
        ("hello,world", "Hello, world!"),
        ("hello\nworld", "Hello world."),
        ("السعر 3.09", "السعر: 3.09."),
        ("قيمته -5", "قيمته -5!"),
        ("listener@example.org", "listener@example.org."),
        ("25 %", "25%"),
        ("أحمد", "ا\u0654حمد"),
    ],
)
def test_permitted_formatting_preserves_content(before, after):
    assert format_signature(before) == format_signature(after)


def test_risk_notes_cover_unclear_foreign_loop_and_duplicate_audio():
    row = proposal_row()
    row.update(text="غير واضح", flags=["duplicate_audio"])
    row["candidates"] = [
        {"name": "secret", "text": "English speech", "generation_limit_hit": False},
        {"name": "secret2", "text": "نحن " * 375, "generation_limit_hit": True},
    ]
    payload = json.loads(messages(row)[1]["content"])
    assert set(payload["review_issues"]) == {
        "unclear_span",
        "foreign_script",
        "repetition_or_truncation",
        "duplicate_audio",
    }
    assert all("secret" not in c for c in payload["candidates"])
    value = proposal_value(row)
    value.update(
        base_candidate=None, proposed_verified_text=None, proposed_ideal_text=None, changes=[]
    )
    assert parse_proposal(json.dumps(value), "stop", row) == value
    value["issue_notes"].pop("unclear_span")
    with pytest.raises(ValueError, match="Every review issue"):
        parse_proposal(json.dumps(value), "stop", row)


@pytest.mark.parametrize(
    "mutation",
    [
        "no_uncertainty",
        "no_format_note",
        "unanchored_edit",
        "no_word_note",
        "clipped_basis",
        "loop",
        "placeholder",
    ],
)
def test_proposals_require_auditable_evidence(mutation):
    row = proposal_row()
    value = proposal_value(row)
    if mutation == "no_uncertainty":
        value["uncertainties"] = []
    elif mutation == "no_format_note":
        value["changes"] = []
    elif mutation == "unanchored_edit":
        value["changes"][0]["before"] = "absent evidence"
    elif mutation == "no_word_note":
        value.update(proposed_verified_text="لا تذهب", proposed_ideal_text="لا تذهب", changes=[])
    elif mutation == "clipped_basis":
        row["text"] = "س" * 1801
        value["issue_notes"] = dict.fromkeys(review_issues(row), "Needs listening")
    elif mutation == "loop":
        value.update(proposed_verified_text="نحن " * 375, proposed_ideal_text="نحن " * 375)
    elif mutation == "placeholder":
        value.update(
            proposed_verified_text="مرحبا [غير واضح]", proposed_ideal_text="مرحبا [غير واضح]"
        )
    if mutation in {"no_format_note", "unanchored_edit", "no_word_note"}:
        adjustments = []
        parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
        assert any(a["type"] == "metadata_warning" for a in adjustments)
        assert parsed["proposed_verified_text"] == value["proposed_verified_text"]
    else:
        with pytest.raises(ValueError):
            parse_proposal(json.dumps(value), "stop", row)


def test_review_requires_issue_resolution_and_allows_spoken_unclear_words():
    from data_processing.arabic_reference.review import validate_review

    row = proposal_row()
    row.update(text="غير واضح", binding="b", policy_sha256="p", audio_sha256="a")
    review = human_review(row)
    review.update(verified_text="غير واضح", ideal_text="غير واضح.")
    review["issue_resolutions"] = {
        "unclear_span": "The speaker actually says these two words; this is not an annotation placeholder."
    }
    validate_review(review, row)
    review["issue_resolutions"] = {}
    with pytest.raises(ValueError, match="Every review issue"):
        validate_review(review, row)


@pytest.mark.parametrize("marker", ["[غير واضح]", "<غيرواضح>", "[Unclear]", "(inaudible)"])
def test_placeholder_cannot_be_verified_even_with_listening_attestation(plan, marker):
    from data_processing.arabic_reference.review import validate_review

    row = load_plan(plan)[1][0]
    review = human_review(row)
    review.update(verified_text=marker, ideal_text=marker)
    with pytest.raises(ValueError, match="Marked unclear"):
        validate_review(review, row)


def test_duplicate_ideal_conflict_rejected_at_import_and_scoring(inputs, tmp_path):
    pytest.importorskip("rapidfuzz")
    for _, source in inputs:
        rows = list(read_jsonl(source))
        rows[1].update(
            audio=rows[0]["audio"],
            audio_sha256=rows[0]["audio_sha256"],
            actual_duration=rows[0]["actual_duration"],
        )
        write_jsonl(source, rows)
    root = tmp_path / "duplicate-plan"
    prepare(inputs, root, per_dataset=1)
    rows = [r for r in load_plan(root)[1] if "duplicate_audio" in r["flags"]]
    reviews = [human_review(r) for r in rows]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, reviews)
    import_reviews(root, [path], reviewed)
    reviews[1]["ideal_text"] = "مرحبا بكم!"
    write_jsonl(path, reviews)
    with pytest.raises(ValueError, match="Conflicting verified/ideal"):
        import_reviews(root, [path], tmp_path / "conflict.jsonl")
    # Scoring must enforce duplicate consistency even for externally constructed reviews.
    output = list(read_jsonl(reviewed))
    target = next(r for r in output if r["id"] == reviews[1]["id"])
    target["ideal_text"] = reviews[1]["ideal_text"]
    target["review"]["ideal_text"] = target["ideal_text"]
    target["reviewed_sha256"] = digest({k: v for k, v in target.items() if k != "reviewed_sha256"})
    write_jsonl(reviewed, output)
    with pytest.raises(ValueError, match="Conflicting verified/ideal"):
        score(root, reviewed, tmp_path / "bad-score.json")


def test_alias_routes_to_advertised_checkpoint_and_resume_detects_changes(plan, tmp_path, server):
    url, state = server
    state["root"] = "Qwen/Qwen3.8-27B-FP8"
    result = doctor(url, "corrector")
    assert result["declared_checkpoint"] is None
    assert result["server_model"]["root"] == state["root"]
    assert result["identity_check"] == "reported_by_server"
    kwargs = {"url": url, "model": "corrector", "emit": lambda _: None}
    output = tmp_path / "alias-run"
    assert propose(plan, output, **kwargs)["counts"] == {"proposed": 6}
    assert propose(plan, output, **kwargs)["resumed"] == 6
    assert json.loads((output / "run.json").read_text())["server"] == result
    state["root"] = "example/different-checkpoint"
    with pytest.raises(ValueError, match="changed"):
        propose(plan, output, **kwargs)
    assert state["calls"] == 6
    with pytest.raises(ValueError, match="Expected served model"):
        doctor(url, "missing-alias")


@pytest.mark.parametrize("reported_root", [None, "corrector"])
def test_alias_only_server_is_not_claimed_checkpoint_verified(server, reported_root):
    url, state = server
    state["root"] = reported_root
    result = doctor(url, "corrector")
    assert result["identity_check"] == "alias_only; checkpoint unverified"
    assert result["server_model"]["root"] == reported_root


def test_cli_defaults_to_served_alias_without_checkpoint_pin(plan, server, capsys):
    from data_processing.arabic_reference.__main__ import main

    url, state = server
    state["root"] = "Qwen/Qwen3.8-27B-FP8"
    assert main(["doctor", "--plan", str(plan), "--url", url]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["served_model"] == "corrector"
    assert result["declared_checkpoint"] is None
    assert result["server_model"]["root"] == state["root"]


@pytest.mark.parametrize("before,after", [("ام", "أم"), ("اتوقع", "أتوقع"), ("ادريس", "إدريس")])
def test_declared_ideal_spelling_improvement_is_allowed_and_reviewed(before, after):
    from data_processing.arabic_reference.review import validate_review

    row = proposal_row()
    row.update(text=before, binding="b", policy_sha256="p", audio_sha256="a")
    value = proposal_value(row)
    value.update(
        proposed_verified_text=before,
        proposed_ideal_text=after + ".",
        changes=[
            {
                "kind": "ideal_orthography",
                "before": before,
                "after": after,
                "reason": "Hamza spelling refinement; still needs review",
            },
            {
                "kind": "formatting",
                "before": after,
                "after": after + ".",
                "reason": "Sentence punctuation",
            },
        ],
    )
    assert parse_proposal(json.dumps(value), "stop", row) == value
    # It must not be silently reclassified as ordinary punctuation/diacritics.
    value["changes"][0]["kind"] = "formatting"
    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert parsed["proposed_ideal_text"] == after + "."
    assert any(a["type"] == "metadata_warning" for a in adjustments)
    review = human_review(row)
    review.update(verified_text=before, ideal_text=after + ".")
    with pytest.raises(ValueError, match="spelling explanation"):
        validate_review(review, row)
    review["ideal_spelling_reason"] = (
        "Reviewed the word and approved its conventional hamza spelling"
    )
    validate_review(review, row)


@pytest.mark.parametrize(
    "before,after",
    [("لا تذهب", "تذهب"), ("2400", "2250"), ("علي", "على"), ("رحمه", "رحمة"), ("أم", "إم")],
)
def test_spelling_explanation_cannot_replace_content_attestation(plan, before, after):
    from data_processing.arabic_reference.review import validate_review

    row = load_plan(plan)[1][0]
    review = human_review(row)
    review.update(
        verified_text=before,
        ideal_text=after,
        ideal_spelling_reason="Looks better",
        ideal_content_preserved=False,
    )
    with pytest.raises(ValueError, match="lexical review"):
        validate_review(review, row)


def invalid_reply(body):
    value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
    value["uncertainties"] = []
    return value


def test_validation_feedback_repairs_once_and_preserves_every_response(plan, tmp_path, server):
    url, state = server
    state["reply"] = (
        lambda body: invalid_reply(body)
        if len(body["messages"]) == 2
        else proposal_value(payload=json.loads(body["messages"][1]["content"]))
    )
    result = propose(plan, tmp_path / "repair", url=url, model="corrector", emit=lambda _: None)
    assert result["counts"] == {"proposed": 6} and state["calls"] == 12
    assert result["validation_retries"] == 6
    assert result["usage"]["prompt_tokens"] == 1200
    records = list(read_jsonl(tmp_path / "repair/proposals.jsonl"))
    for r in records:
        assert len(r["attempts"]) == 2
        assert "listening uncertainty" in r["attempts"][0]["validation_error"]
        assert "Validation error:" in r["attempts"][0]["repair_feedback"]
        assert r["raw_response"] == r["attempts"][1]["raw_response"]
        assert r["verified_text"] is None and r["ideal_text"] is None
    assert any(len(request["messages"]) == 4 for request in state["requests"])


def test_persistent_validation_error_stops_after_one_repair(plan, tmp_path, server):
    url, state = server
    state["reply"] = invalid_reply
    result = propose(plan, tmp_path / "bad", url=url, model="corrector", emit=lambda _: None)
    assert result["counts"] == {"failed": 6} and state["calls"] == 12
    assert len(result["failed_examples"]) == 6 and result["failure_counts"]
    assert all(len(r["attempts"]) == 2 for r in read_jsonl(tmp_path / "bad/proposals.jsonl"))


def test_smoke_controls_produce_drafts_without_becoming_gold(plan, tmp_path, server):
    url, _ = server
    result = propose(
        plan, tmp_path / "smoke", url=url, model="corrector", smoke=True, emit=lambda _: None
    )
    assert result["smoke_checks"]["passed"]
    assert len(result["smoke_checks"]["agreement_controls"]) == 6
    assert result["verified"] == 0


def test_valid_abstentions_are_not_retried_but_fail_agreement_control_gate(
    plan, tmp_path, server, capsys
):
    from data_processing.arabic_reference.__main__ import main

    url, state = server

    def abstain(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        value.update(
            base_candidate=None, proposed_verified_text=None, proposed_ideal_text=None, changes=[]
        )
        return value

    state["reply"] = abstain
    result = main(
        [
            "propose",
            "--plan",
            str(plan),
            "--output",
            str(tmp_path / "smoke"),
            "--url",
            url,
            "--smoke",
        ]
    )
    assert result == 1
    saved = json.loads((tmp_path / "smoke/summary.json").read_text())
    assert saved["counts"] == {"unresolved": 6} and not saved["smoke_checks"]["passed"]
    assert state["calls"] == 6 and saved["validation_retries"] == 0


def test_smoke_controls_deterministic_and_missing_controls_explicit(plan):
    metadata, rows = load_plan(plan)
    _, controls = select_smoke(metadata, rows)
    assert controls == select_smoke(metadata, list(reversed(rows)))[1]
    for row in rows:
        row["candidates"][0]["text"] = "different words"
    selected, controls = select_smoke(metadata, rows)
    assert controls == {}
    result = check_smoke(selected, [], controls)
    assert not result["passed"] and set(result["missing_control_datasets"]) == set(DATASETS)


def test_ideal_spelling_does_not_replace_substrings_of_other_words():
    row = proposal_row()
    row["text"] = "ام في عام"
    value = proposal_value(row)
    value.update(
        proposed_verified_text="ام في عام",
        proposed_ideal_text="أم في عام.",
        changes=[
            {
                "kind": "ideal_orthography",
                "before": "ام",
                "after": "أم",
                "reason": "Hamza spelling",
            },
            {
                "kind": "formatting",
                "before": "أم في عام",
                "after": "أم في عام.",
                "reason": "Sentence punctuation",
            },
        ],
    )
    assert parse_proposal(json.dumps(value), "stop", row) == value


@pytest.mark.parametrize(
    "before,after,edits",
    [
        (
            "وقف طويلا تكلمه في حاجتها",
            "وقف طويلًا تكلّمه في حاجتها.",
            [("طويلا", "طويلًا"), ("تكلمه", "تكلّمه")],
        ),
        ("في ذمتك", "في ذمّتك.", [("ذمتك", "ذمّتك")]),
    ],
)
def test_mislabelled_diacritics_preserve_transcript_and_record_adjustments(before, after, edits):
    row = proposal_row()
    row["text"] = before
    value = proposal_value(row)
    value.update(
        proposed_verified_text=before,
        proposed_ideal_text=after,
        changes=[
            {
                "kind": "ideal_orthography",
                "before": a,
                "after": b,
                "reason": "Incorrect model claim: this adds hamza",
            }
            for a, b in edits
        ]
        + [
            {
                "kind": "formatting",
                "before": after[:-1],
                "after": after,
                "reason": "Terminal period",
            }
        ],
    )
    raw = json.dumps(value)
    adjustments = []
    parsed = parse_proposal(raw, "stop", row, adjustments=adjustments)
    assert parsed["proposed_verified_text"] == before
    assert parsed["proposed_ideal_text"] == after
    assert all(c["kind"] == "formatting" for c in parsed["changes"])
    assert [n["change_index"] for n in adjustments] == list(range(len(edits)))
    assert all(
        n["from_kind"] == "ideal_orthography" and n["to_kind"] == "formatting" for n in adjustments
    )
    # The original response and its unverified explanation are retained, not rewritten.
    assert json.loads(raw) == value
    assert parsed["changes"][0]["reason"] == value["changes"][0]["reason"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("لا تذهب", "تذهب"),
        ("3.09", "309"),
        ("-5", "5"),
        ("رحمه", "رحمة"),
        ("على", "علي"),
        ("أم", "ام"),
        ("بارت تايم", "part-time"),
    ],
)
def test_edit_label_cannot_hide_actual_editorial_differences(before, after):
    row = proposal_row()
    row["text"] = before
    value = proposal_value(row)
    value.update(
        proposed_verified_text=before,
        proposed_ideal_text=after,
        changes=[
            {
                "kind": "ideal_orthography",
                "before": before,
                "after": after,
                "reason": "Claimed formatting",
            }
        ],
    )
    from data_processing.arabic_reference.prompts import proposal_checks

    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert proposal_checks(parsed, row, adjustments)["wording_to_ideal"]
    assert parsed["changes"][0]["kind"] == "ideal_orthography"
    assert not any(a["type"] == "edit_category" for a in adjustments)


def test_declared_formatting_edits_can_use_prior_formatting_intermediate():
    row = proposal_row()
    value = proposal_value(row)
    value["changes"] = [
        {"kind": "formatting", "before": "مرحبا", "after": "مرحبًا", "reason": "Vowel mark"},
        {"kind": "formatting", "before": "مرحبًا بكم", "after": "مرحبًا بكم.", "reason": "Period"},
    ]
    assert parse_proposal(json.dumps(value), "stop", row) == value
    value["changes"][1]["before"] = "absent intermediate"
    adjustments = []
    parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert any(
        a["type"] == "metadata_warning" and "not anchored" in a["reason"] for a in adjustments
    )


@pytest.mark.parametrize("mode", ["edits", "notes", "both"])
def test_runtime_audits_metadata_normalization_without_retry_or_verification(
    plan, tmp_path, server, mode
):
    url, state = server
    edit_count = int(mode != "notes")
    note_count = 2 if mode != "edits" else 0

    def reply(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        if edit_count:
            value["changes"][0]["kind"] = "ideal_orthography"
        if note_count:
            value["issue_notes"].update(duration="Check clipping", dialect="Check dialect")
        return value

    state["reply"] = reply
    output = tmp_path / "normalized"
    result = propose(plan, output, url=url, model="corrector", smoke=True, emit=lambda _: None)
    assert state["calls"] == 6 and result["validation_retries"] == 0
    assert result["normalized_records"] == 6
    assert result["normalized_edits"] == 6 * edit_count
    assert result["relocated_issue_notes"] == 6 * note_count
    assert result["smoke_checks"]["passed"] and result["verified"] == 0
    for record in read_jsonl(output / "proposals.jsonl"):
        assert record["validation_adjustments"] == record["attempts"][0]["validation_adjustments"]
        raw = json.loads(record["raw_response"])
        assert raw["changes"][0]["kind"] == ("ideal_orthography" if edit_count else "formatting")
        assert record["proposal"]["changes"][0]["kind"] == "formatting"
        assert (
            record["proposal"]["proposed_ideal_text"] == raw["proposed_ideal_text"] == "مرحبًا بكم."
        )
        assert record["proposal"]["proposed_verified_text"] == raw["proposed_verified_text"]
        assert record["verified_text"] is None and record["ideal_text"] is None
        if note_count:
            assert raw["issue_notes"]["duration"] == "Check clipping"
            assert "duration" not in record["proposal"]["issue_notes"]
            assert any(
                "[duration]: Check clipping" in u for u in record["proposal"]["uncertainties"]
            )
    resumed = propose(plan, output, url=url, model="corrector", smoke=True, emit=lambda _: None)
    assert resumed["resumed"] == 6 and state["calls"] == 6
    assert resumed["normalized_edits"] == result["normalized_edits"]
    assert resumed["relocated_issue_notes"] == result["relocated_issue_notes"]
    pack = tmp_path / "listening"
    review_pack(plan, pack, proposals=output, metadata_only=True)
    jobs = json.loads((pack / "jobs.json").read_text())["jobs"]
    assert all(len(job["validation_adjustments"]) == edit_count + note_count for job in jobs)
    assert (
        "Metadata adjustments and explicit rendering changes" in (pack / "index.html").read_text()
    )
    if note_count:
        assert all(
            any("[dialect]: Check dialect" in u for u in job["proposal"]["uncertainties"])
            for job in jobs
        )


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("unresolved", [False, True])
def test_extra_issue_notes_become_audited_uncertainties(required, unresolved):
    row = proposal_row()
    if required:
        row["flags"].append("duplicate_audio")
    value = proposal_value(row)
    if unresolved:
        value.update(
            base_candidate=None, proposed_verified_text=None, proposed_ideal_text=None, changes=[]
        )
    extras = {
        "duration": "Check the full clip; short duration alone does not establish missing words.",
        "initial_word_disagreement": "Check initial اي against the candidate that omits it.",
    }
    original_notes = dict(value["issue_notes"])
    value["issue_notes"].update(extras)
    raw = json.dumps(value, ensure_ascii=False)
    adjustments = []
    parsed = parse_proposal(raw, "stop", row, adjustments=adjustments)
    assert parsed["issue_notes"] == original_notes
    for key in (
        "base_candidate",
        "proposed_verified_text",
        "proposed_ideal_text",
        "changes",
        "reason",
    ):
        assert parsed[key] == value[key]
    assert parsed["uncertainties"][: len(value["uncertainties"])] == value["uncertainties"]
    assert len(adjustments) == len(extras)
    for adjustment in adjustments:
        key = adjustment["issue_key"]
        assert adjustment["type"] == "issue_note_relocation"
        assert adjustment["note"] == extras[key]
        assert (
            parsed["uncertainties"][adjustment["uncertainty_index"]]
            == f"Additional unverified model comment [{key}]: {extras[key]}"
        )
    assert json.loads(raw) == value  # Raw model reply is unchanged.
    again = []
    assert parse_proposal(json.dumps(parsed), "stop", row, adjustments=again) == parsed
    assert again == []  # Canonical output never duplicates a relocated note.


@pytest.mark.parametrize(
    "notes,error",
    [
        ([], "must be an object"),
        (None, "must be an object"),
        ({"duration": ""}, "nonempty strings"),
        ({"duration": "  "}, "nonempty strings"),
        ({"duration": {"note": "Nested note"}}, "nonempty strings"),
        ({"duration": None}, "nonempty strings"),
        ({"": "Question"}, "keys must be nonempty"),
    ],
)
def test_note_relocation_does_not_accept_malformed_notes(notes, error):
    value = proposal_value()
    value["issue_notes"] = notes
    adjustments = []
    with pytest.raises(ValueError, match=error):
        parse_proposal(json.dumps(value), "stop", proposal_row(), adjustments=adjustments)
    assert adjustments == []


def test_extra_note_cannot_replace_required_note_or_mask_bad_transcript():
    row = proposal_row()
    row["flags"].append("duplicate_audio")
    value = proposal_value(row)
    value["issue_notes"] = {"duration": "Check duration"}
    adjustments = []
    with pytest.raises(ValueError, match="missing keys: duplicate_audio"):
        parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert adjustments == []
    value["issue_notes"]["duplicate_audio"] = "Check duplicate references"
    value["changes"][0]["before"] = "Absent evidence"
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert parsed["proposed_verified_text"] == value["proposed_verified_text"]
    assert [a["type"] for a in adjustments] == ["issue_note_relocation", "metadata_warning"]
    assert "not anchored" in adjustments[1]["reason"]


def test_human_issue_resolutions_still_require_exact_keys(plan):
    from data_processing.arabic_reference.review import validate_review

    row = load_plan(plan)[1][0]
    review = human_review(row)
    review["issue_resolutions"]["duration"] = "Additional model note is not a required flag"
    with pytest.raises(ValueError, match="Unexpected review issue note keys: duration"):
        validate_review(review, row)


@pytest.mark.parametrize("include_retention_note", [False, True])
def test_commonvoice_unchanged_draft_drops_only_redundant_edit_metadata(include_retention_note):
    row = proposal_row()
    row["text"] = "قال توم أنّه شكّ أنّ ماري ستفعل ذلك."
    value = proposal_value(row)
    value.update(proposed_verified_text=row["text"], proposed_ideal_text=row["text"])
    value["changes"] = [
        {
            "kind": "ideal_orthography",
            "before": word,
            "after": word,
            "reason": "Hamza already present",
        }
        for word in ("أنّه", "أنّ")
    ]
    if include_retention_note:
        value["changes"].append(
            {
                "kind": "formatting",
                "before": row["text"],
                "after": row["text"],
                "reason": "Retained diacritics and punctuation",
            }
        )
    raw = json.dumps(value, ensure_ascii=False)
    adjustments = []
    parsed = parse_proposal(raw, "stop", row, adjustments=adjustments)
    assert parsed == {**value, "changes": []}
    assert len(adjustments) == len(value["changes"])
    for index, adjustment in enumerate(adjustments):
        assert adjustment["type"] == "no_op_edit_removed"
        assert adjustment["change_index"] == index
        assert adjustment["change"] == value["changes"][index]
    assert json.loads(raw) == value
    again = []
    assert parse_proposal(json.dumps(parsed), "stop", row, adjustments=again) == parsed
    assert again == []


@pytest.mark.parametrize(
    "kind", ["formatting", "ideal_orthography", "orthography", "content", "script", "unclear"]
)
def test_anchored_noop_is_not_actual_edit_evidence(kind):
    row = proposal_row()
    value = proposal_value(row)
    value.update(proposed_verified_text=row["text"], proposed_ideal_text=row["text"])
    value["changes"] = [{"kind": kind, "before": "بكم", "after": "بكم", "reason": "Retained text"}]
    assert parse_proposal(json.dumps(value), "stop", row)["changes"] == []


@pytest.mark.parametrize("span", ["", " ", "not present"])
def test_noop_must_have_nonempty_anchored_spans(span):
    row = proposal_row()
    value = proposal_value(row)
    value.update(proposed_verified_text=row["text"], proposed_ideal_text=row["text"])
    value["changes"] = [
        {"kind": "formatting", "before": span, "after": span, "reason": "Claimed retention"}
    ]
    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert parsed["changes"] == value["changes"]
    assert adjustments[0]["type"] == "metadata_warning"
    assert "not anchored" in adjustments[0]["reason"]


@pytest.mark.parametrize(
    "base,wording,ideal,kind,span,error",
    [
        ("مرحبا بكم", "مرحبا بكم", "مرحبًا بكم.", "formatting", "بكم", "Rendering differs"),
        ("لا تذهب", "تذهب", "تذهب", "content", "تذهب", "Unexplained word/spelling/script"),
        (
            "انا هنا",
            "انا هنا",
            "أنا هنا",
            "ideal_orthography",
            "هنا",
            "Ideal spelling/content differences",
        ),
    ],
)
def test_noop_cannot_cover_an_actual_unexplained_change(base, wording, ideal, kind, span, error):
    row = proposal_row()
    row["text"] = base
    value = proposal_value(row)
    value.update(proposed_verified_text=wording, proposed_ideal_text=ideal)
    value["changes"] = [{"kind": kind, "before": span, "after": span, "reason": "Retained text"}]
    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert parsed["changes"] == []
    assert adjustments[0]["type"] == "no_op_edit_removed"
    assert any(a["type"] == "metadata_warning" and error in a["reason"] for a in adjustments)


def test_noop_removal_keeps_raw_indices_for_actual_edit_adjustments():
    row = proposal_row()
    value = proposal_value(row)
    value["changes"][0]["kind"] = "ideal_orthography"
    value["changes"].insert(
        0, {"kind": "content", "before": "بكم", "after": "بكم", "reason": "Retained"}
    )
    adjustments = []
    parsed = parse_proposal(json.dumps(value), "stop", row, adjustments=adjustments)
    assert len(parsed["changes"]) == 1 and parsed["changes"][0]["kind"] == "formatting"
    assert [a["change_index"] for a in adjustments] == [0, 1]
    assert [a["type"] for a in adjustments] == ["no_op_edit_removed", "edit_category"]
    assert parsed["proposed_ideal_text"] == value["proposed_ideal_text"]


def test_runtime_audits_noop_removal_without_regeneration(plan, tmp_path, server):
    url, state = server

    def reply(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        value["proposed_ideal_text"] = value["proposed_verified_text"]
        value["changes"] = [
            {
                "kind": "ideal_orthography",
                "before": "بكم",
                "after": "بكم",
                "reason": "Retained spelling",
            },
            {
                "kind": "formatting",
                "before": "مرحبا بكم",
                "after": "مرحبا بكم",
                "reason": "Retained formatting",
            },
        ]
        return value

    state["reply"] = reply
    output = tmp_path / "noop"
    result = propose(plan, output, url=url, model="corrector", smoke=True, emit=lambda _: None)
    assert state["calls"] == 6 and result["validation_retries"] == 0
    assert result["normalized_records"] == 6 and result["removed_no_op_edits"] == 12
    assert result["normalized_edits"] == 0 and result["relocated_issue_notes"] == 0
    assert result["smoke_checks"]["passed"] and result["verified"] == 0
    for record in read_jsonl(output / "proposals.jsonl"):
        raw = json.loads(record["raw_response"])
        assert len(raw["changes"]) == 2 and record["proposal"]["changes"] == []
        assert record["validation_adjustments"] == record["attempts"][0]["validation_adjustments"]
        assert record["proposal"]["proposed_verified_text"] == raw["proposed_verified_text"]
        assert record["proposal"]["proposed_ideal_text"] == raw["proposed_ideal_text"]
        assert record["verified_text"] is None and record["ideal_text"] is None
    resumed = propose(plan, output, url=url, model="corrector", smoke=True, emit=lambda _: None)
    assert resumed["resumed"] == 6 and state["calls"] == 6
    assert resumed["removed_no_op_edits"] == 12
    pack = tmp_path / "listening"
    review_pack(plan, pack, proposals=output, metadata_only=True)
    jobs = json.loads((pack / "jobs.json").read_text())["jobs"]
    assert all(
        len(job["validation_adjustments"]) == 2 and job["proposal"]["changes"] == [] for job in jobs
    )


@pytest.mark.parametrize("span,missing", [("مرحبا", "after"), ("مرحبًا", "before")])
def test_noop_requires_presence_in_both_stated_texts(span, missing):
    value = proposal_value()
    value["changes"].insert(
        0, {"kind": "formatting", "before": span, "after": span, "reason": "Claimed retention"}
    )
    adjustments = []
    parse_proposal(json.dumps(value), "stop", proposal_row(), adjustments=adjustments)
    assert adjustments[0]["type"] == "metadata_warning"
    assert f"change[0] {missing} span is absent" in adjustments[0]["reason"]


@pytest.mark.parametrize(
    "before,after",
    [
        ("و ندمت", "وندمت"),
        ("ياابو سالم", "يا أبو سالم"),
        ("الحلقه القادمه", "الحلقة القادمة"),
        ("لغه برمجه", "لغة برمجة"),
        ("المسؤلية", "المسؤولية"),
        ("لابوسريع", "لأبو سريع"),
        ("اخر كلام", "آخر كلام"),
        ("مكانً مناسبً", "مكانًا مناسبًا"),
    ],
)
def test_contextual_editorial_drafts_are_audited_and_require_human_review(before, after):
    from data_processing.arabic_reference.prompts import proposal_checks
    from data_processing.arabic_reference.review import validate_review

    row = proposal_row()
    row.update(text=before, binding="b", policy_sha256="p", audio_sha256="a")
    value = proposal_value(row)
    value.update(
        proposed_verified_text=before,
        proposed_ideal_text=after,
        changes=[
            {
                "kind": "ideal_orthography",
                "before": before,
                "after": after,
                "reason": "Contextual editorial proposal; needs review",
            }
        ],
    )
    raw = json.dumps(value, ensure_ascii=False)
    changes = []
    parsed = parse_proposal(raw, "stop", row, adjustments=changes)
    assert parsed == value and not changes
    checks = proposal_checks(parsed, row, changes)
    assert checks["wording_to_ideal"] and checks["basis_to_wording"] == []
    review = human_review(row)
    review.update(verified_text=before, ideal_text=after, ideal_content_preserved=False)
    with pytest.raises(ValueError, match="attestation"):
        validate_review(review, row)
    review["ideal_content_preserved"] = True
    with pytest.raises(ValueError, match="spelling explanation"):
        validate_review(review, row)
    review["ideal_spelling_reason"] = (
        "Reviewer approved contextual orthography with the same dialect and wording"
    )
    validate_review(review, row)


@pytest.mark.parametrize(
    "before,after",
    [
        ("3.09", "309"),
        ("-5", "5"),
        ("2400", "2250"),
        ("25%", "25"),
        ("listener@example.org", "listener@example.com"),
        ("https://example.org/a", "https://example.org/b"),
        ("2400", "ألفان وأربعمائة"),
    ],
)
def test_editorial_attestation_does_not_override_protected_values(before, after):
    from data_processing.arabic_reference.review import validate_review

    row = proposal_row()
    row.update(binding="b", policy_sha256="p", audio_sha256="a")
    review = human_review(row)
    review.update(verified_text=before, ideal_text=after, ideal_spelling_reason="Looks better")
    with pytest.raises(ValueError, match="preserve numbers and identifiers"):
        validate_review(review, row)


def test_semantic_rewrite_remains_visible_in_saved_records_and_listening_pack(
    plan, tmp_path, server
):
    from data_processing.arabic_reference.review import review_pack

    url, state = server

    def reply(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        value.update(
            proposed_verified_text="فالدنيا معاه",
            proposed_ideal_text="فالدار الآخرة معاه",
            changes=[
                {
                    "kind": "formatting",
                    "before": "فالدنيا",
                    "after": "فالدار الآخرة",
                    "reason": "Incorrectly claims punctuation only",
                }
            ],
        )
        return value

    state["reply"] = reply
    root = tmp_path / "editorial"
    result = propose(plan, root, url=url, model="corrector", emit=lambda _: None)
    assert result["counts"] == {"proposed": 6} and result["verified"] == 0
    assert result["ideal_editorial_change_records"] == 6
    assert result["wording_change_records"] == 6 and result["metadata_warning_records"] == 6
    records = list(read_jsonl(root / "proposals.jsonl"))
    for r in records:
        assert r["verified_text"] is None and r["ideal_text"] is None
        assert r["proposal_checks"]["wording_to_ideal"] == [
            {"before": "فالدنيا", "after": "فالدار الآخرة"}
        ]
        assert (
            json.loads(r["raw_response"])["changes"][0]["reason"]
            == "Incorrectly claims punctuation only"
        )
        assert r["proposal_checks"]["metadata_warnings"]
    assert propose(plan, root, url=url, model="corrector", emit=lambda _: None)["resumed"] == 6
    pack = tmp_path / "editorial-review"
    review_pack(plan, pack, proposals=root, metadata_only=True)
    jobs = json.loads((pack / "jobs.json").read_text())["jobs"]
    assert all(j["proposal_checks"]["wording_to_ideal"] for j in jobs)
    assert "Independently computed differences" in (pack / "index.html").read_text()


def test_editorial_attestation_is_required_on_import_and_scoring(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    row = load_plan(plan)[1][0]
    review = human_review(row)
    review.update(
        verified_text="الحلقه", ideal_text="الحلقة", ideal_spelling_reason="Contextual spelling"
    )
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    missing = dict(review)
    missing.pop("ideal_content_preserved")
    write_jsonl(path, [missing])
    with pytest.raises(ValueError, match="schema"):
        import_reviews(plan, [path], reviewed)
    write_jsonl(path, [review])
    import_reviews(plan, [path], reviewed)
    rows = list(read_jsonl(reviewed))
    target = next(r for r in rows if r["id"] == row["id"])
    assert target["reviewed_ideal_differences"] == [{"before": "الحلقه", "after": "الحلقة"}]
    target["review"]["ideal_content_preserved"] = False
    target["reviewed_sha256"] = digest({k: v for k, v in target.items() if k != "reviewed_sha256"})
    write_jsonl(reviewed, rows)
    with pytest.raises(ValueError, match="attestation"):
        score(plan, reviewed, tmp_path / "score.json")


@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_scoring_rechecks_computed_editorial_differences(plan, tmp_path, mutation):
    pytest.importorskip("rapidfuzz")
    row = load_plan(plan)[1][0]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(row)])
    import_reviews(plan, [path], reviewed)
    rows = list(read_jsonl(reviewed))
    target = next(r for r in rows if r["id"] == row["id"])
    if mutation == "missing":
        target.pop("reviewed_ideal_differences")
    else:
        target["reviewed_ideal_differences"] = [{"before": "fabricated", "after": "evidence"}]
    target["reviewed_sha256"] = digest({k: v for k, v in target.items() if k != "reviewed_sha256"})
    write_jsonl(reviewed, rows)
    with pytest.raises(ValueError, match="editorial differences"):
        score(plan, reviewed, tmp_path / "score.json")


def test_full_suite_selection_retains_every_source_row_and_smoke_controls(inputs, tmp_path):
    root = tmp_path / "full"
    metadata = prepare(inputs, root, full_suite=True)
    _, rows = load_plan(root)
    assert len(rows) == 18 and metadata["pilot_clips"] == 0
    assert metadata["selection_scope"] == "full_suite" and metadata["per_dataset"] is None
    assert all(r["cohorts"] == ["full"] for r in rows)
    assert prepare(inputs, root, full_suite=True) == metadata
    selected, controls = select_smoke(metadata, rows)
    assert len(controls) == 6 and set(controls.values()) <= {r["id"] for r in selected}
    with pytest.raises(ValueError, match="differs"):
        prepare(inputs, root, per_dataset=1)


def test_full_suite_review_scores_have_separate_normalized_and_strict_tracks(inputs, tmp_path):
    pytest.importorskip("rapidfuzz")
    root = tmp_path / "full"
    prepare(inputs, root, full_suite=True)
    rows = load_plan(root)[1]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(r) for r in rows])
    import_reviews(root, [path], reviewed)
    result = score(root, reviewed, tmp_path / "scores.json")
    assert result["scores"]["full/one/ideal_elm/macro"]["full_suite"] is True
    assert result["scores"]["full/one/ideal_strict/macro"]["wer"] == 0
    assert result["scores"]["full/one/original_strict/macro"]["wer"] > 0
    assert result["coverage"]["full"]["verified"] == dict.fromkeys(DATASETS, 3)


def test_dataset_export_preserves_qasr_and_separates_drafts_from_final(inputs, tmp_path, server):
    pq = pytest.importorskip("pyarrow.parquet")
    from data_processing.arabic_reference.release import export_dataset, validate_bundle

    root = tmp_path / "qasr-plan"
    prepare([("qasr", inputs[0][1]), ("gqasr", inputs[1][1])], root, per_dataset=1)
    url, _ = server
    proposals = tmp_path / "proposals"
    propose(root, proposals, url=url, model="corrector", emit=lambda _: None)
    output = tmp_path / "release"
    manifest = export_dataset(root, output, proposals=proposals)
    assert manifest["visibility"] == "private" and manifest["repo_type"] == "dataset"
    assert validate_bundle(output)["bundle_sha256"] == manifest["bundle_sha256"]
    for ds in DATASETS:
        table = pq.read_table(output / "data" / ds / "test.parquet")
        rows = table.to_pylist()
        assert len(rows) == 1
        r = rows[0]
        assert r["text"] == "مرحبا بكم" and r["asr_text"] == r["gqasr_text"] == "مرحبًا بكم."
        assert r["ideal_text_draft"] == "مرحبًا بكم." and r["verified_text_draft"] == "مرحبا بكم"
        assert (
            r["ideal_text"] is None and r["verified_text"] is None and r["audio_reviewed"] is False
        )
        assert r["review_status"] == "pending" and r["proposal_status"] == "proposed"
        assert r["split"] == "test" and r["purpose"] == "evaluation_only"
        assert str(tmp_path) not in json.dumps(r)
        assert str(table.schema.field("ideal_text").type) == "string"
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["corrector"]["served_model"] == "corrector"
    assert provenance["corrector"]["prompt_version"] == "arabic-reference-proposer-v4.1"
    assert provenance["audio_included"] is False
    assert (
        export_dataset(root, output, proposals=proposals)["bundle_sha256"]
        == manifest["bundle_sha256"]
    )


def test_dataset_export_keeps_failed_and_pending_rows_visible(plan, tmp_path, server):
    pq = pytest.importorskip("pyarrow.parquet")
    from data_processing.arabic_reference.release import export_dataset

    url, state = server
    state["fail"] = True
    proposals = tmp_path / "failed"
    propose(plan, proposals, url=url, model="corrector", emit=lambda _: None)
    output = tmp_path / "release"
    export_dataset(plan, output, proposals=proposals)
    rows = [r for p in output.glob("data/*/*.parquet") for r in pq.read_table(p).to_pylist()]
    assert len(rows) == 6 and all(r["ideal_text"] is None for r in rows)
    assert all(r["proposal_status"] == "failed" and r["ideal_text_draft"] is None for r in rows)
    pending = tmp_path / "pending"
    export_dataset(plan, pending)
    pending_rows = [
        r for p in pending.glob("data/*/*.parquet") for r in pq.read_table(p).to_pylist()
    ]
    assert all(
        r["proposal_status"] == "pending" and r["ideal_text_draft"] is None for r in pending_rows
    )


def test_dataset_export_validates_reviews_and_never_fills_unreviewed_finals(plan, tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    from data_processing.arabic_reference.release import export_dataset

    row = load_plan(plan)[1][0]
    path, reviewed = tmp_path / "review.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(row)])
    import_reviews(plan, [path], reviewed)
    output = tmp_path / "release"
    export_dataset(plan, output, reviewed=reviewed)
    values = [r for p in output.glob("data/*/*.parquet") for r in pq.read_table(p).to_pylist()]
    assert sum(r["ideal_text"] is not None for r in values) == 1
    assert sum(r["audio_reviewed"] for r in values) == 1
    forged = list(read_jsonl(reviewed))
    target = next(r for r in forged if r["review_status"] == "pending")
    target["ideal_text"] = "Unreviewed text"
    target["reviewed_sha256"] = digest({k: v for k, v in target.items() if k != "reviewed_sha256"})
    write_jsonl(reviewed, forged)
    with pytest.raises(ValueError, match="Pending rows"):
        export_dataset(plan, tmp_path / "forged", reviewed=reviewed)


@pytest.mark.parametrize("mutation", ["extra", "changed", "symlink"])
def test_frozen_upload_bundle_rejects_extra_or_mutated_files(plan, tmp_path, mutation):
    pytest.importorskip("pyarrow")
    from data_processing.arabic_reference.release import export_dataset, validate_bundle

    output = tmp_path / "release"
    export_dataset(plan, output)
    if mutation == "extra":
        (output / "private-log.txt").write_text("Not an approved upload")
    elif mutation == "changed":
        (output / "README.md").write_text("Changed release")
    else:
        (output / "outside").symlink_to(tmp_path)
    with pytest.raises(ValueError, match=r"Unexpected|changed|symlinks"):
        validate_bundle(output)


def test_release_outputs_are_immutable_and_dataset_selection_is_explicit(plan, tmp_path):
    pytest.importorskip("pyarrow")
    from data_processing.arabic_reference.release import export_dataset

    output = tmp_path / "release"
    result = export_dataset(plan, output, datasets=["sada"])
    assert result["datasets"] == ["sada"] and len(list(output.glob("data/*/*.parquet"))) == 1
    with pytest.raises(ValueError, match="different contents"):
        export_dataset(plan, output, datasets=["commonvoice"])
    with pytest.raises(ValueError, match="known dataset"):
        export_dataset(plan, tmp_path / "bad", datasets=["nonexistent"])
    with pytest.raises(ValueError, match="filename-safe"):
        export_dataset(plan, tmp_path / "bad", release_id="../bad")


class PrivateHubStub:
    def __init__(self, exists=True, private=True):
        self.exists, self.private = exists, private
        self.calls = []

    def repo_info(self, repo_id, repo_type):
        from types import SimpleNamespace

        from huggingface_hub.errors import RepositoryNotFoundError

        if not self.exists:
            # Hub versions differ on whether response is optional. Keep the
            # fixture transport-independent while supplying its HTTP attributes.
            response = SimpleNamespace(
                status_code=404,
                headers={},
                request=SimpleNamespace(method="GET", url=f"https://huggingface.co/{repo_id}"),
            )
            raise RepositoryNotFoundError("Fixture repository is absent", response=response)
        return SimpleNamespace(private=self.private, sha="existing-head")

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))
        self.exists = True
        self.private = kwargs["private"]

    def create_commit(self, repo_id, **kwargs):
        from types import SimpleNamespace

        self.calls.append(("create_commit", repo_id, kwargs))
        return SimpleNamespace(
            oid="new-head", commit_url="https://huggingface.co/datasets/test/bench/commit/new-head"
        )


def test_private_publisher_dry_run_never_calls_hub_and_uploads_only_frozen_files(plan, tmp_path):
    pytest.importorskip("pyarrow")
    from data_processing.arabic_reference.release import export_dataset, publish_bundle

    output = tmp_path / "release"
    export_dataset(plan, output)
    api = PrivateHubStub(exists=False)
    dry = publish_bundle(output, "test/bench", api=api)
    assert dry["executed"] is False and api.calls == []
    result = publish_bundle(output, "test/bench", execute=True, api=api)
    assert result["executed"] and result["commit_sha"] == "new-head"
    assert api.calls[0][2]["private"] is True and api.calls[0][2]["exist_ok"] is False
    commit = api.calls[1][2]
    assert commit["parent_commit"] == "existing-head"
    assert {op.path_in_repo for op in commit["operations"]} == set(dry["files"])
    assert all(isinstance(op.path_or_fileobj, bytes) for op in commit["operations"])


@pytest.mark.parametrize(
    "private,expected,match",
    [
        (False, "existing-head", "public"),
        (True, None, "expected-head"),
        (True, "stale-head", "expected-head"),
    ],
)
def test_private_publisher_refuses_public_or_unreviewed_existing_destination(
    plan, tmp_path, private, expected, match
):
    pytest.importorskip("pyarrow")
    from data_processing.arabic_reference.release import export_dataset, publish_bundle

    output = tmp_path / "release"
    export_dataset(plan, output)
    api = PrivateHubStub(private=private)
    with pytest.raises(ValueError, match=match):
        publish_bundle(output, "test/bench", execute=True, expected_head=expected, api=api)
    assert api.calls == []


def test_leaderboard_uses_original_scores_until_reviews_exist(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    from data_processing.arabic_reference.leaderboard import build_leaderboard
    from data_processing.arabic_reference.release import publish_bundle, validate_bundle

    output = tmp_path / "space"
    build_leaderboard(plan, output)
    payload = json.loads((output / "leaderboard.json").read_text())
    assert payload["verified_clips"] == 0 and payload["independently_audited"] is False
    assert all(
        p["reference_status"] == "original" and not p["full_suite"] for p in payload["panels"]
    )
    assert all(s["track"].startswith("original_") for p in payload["panels"] for s in p["scores"])
    assert validate_bundle(output)["repo_type"] == "space"
    api = PrivateHubStub(exists=False)
    publish_bundle(output, "test/bench", execute=True, api=api)
    assert api.calls[0][2]["space_sdk"] == "static" and api.calls[0][2]["private"]


def test_leaderboard_reviewed_tracks_share_exact_subset_and_missing_macro_is_null(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    from data_processing.arabic_reference.leaderboard import build_leaderboard

    row = load_plan(plan)[1][0]
    path, reviewed = tmp_path / "reviews.jsonl", tmp_path / "reviewed.jsonl"
    write_jsonl(path, [human_review(row)])
    import_reviews(plan, [path], reviewed)
    output = tmp_path / "space"
    build_leaderboard(plan, output, reviewed=reviewed)
    data = json.loads((output / "leaderboard.json").read_text())
    panel = next(p for p in data["panels"] if p["id"] == "reviewed-pilot")
    assert panel["scored_clips"] == 1 and panel["selected_clips"] == 6
    assert len({s["track"] for s in panel["scores"]}) == 6
    assert all(s["utterances"] == 1 for s in panel["scores"])
    assert all(s["wer"] is None for s in panel["scores"] if s["dataset"] == "macro")


@pytest.mark.parametrize(
    "mutation", ["valid", "missing", "duplicate", "audio", "reference", "model_name"]
)
def test_leaderboard_additional_submission_requires_exact_frozen_membership(
    inputs, tmp_path, mutation
):
    pytest.importorskip("rapidfuzz")
    from data_processing.arabic_reference.leaderboard import build_leaderboard

    root = tmp_path / "full"
    prepare(inputs, root, full_suite=True)
    data = list(read_jsonl(inputs[0][1]))
    if mutation == "missing":
        data.pop()
    if mutation == "duplicate":
        data.append(data[0])
    if mutation == "audio":
        data[0]["audio_sha256"] = "0" * 64
    if mutation == "reference":
        data[0]["text"] = "Different original reference"
    path = tmp_path / "submission.jsonl"
    write_jsonl(path, data)
    output = tmp_path / "space"
    name = "one" if mutation == "model_name" else "new-model"
    if mutation != "valid":
        with pytest.raises(ValueError):
            build_leaderboard(root, output, predictions=[(name, path)])
    else:
        build_leaderboard(root, output, predictions=[(name, path)])
        payload = json.loads((output / "leaderboard.json").read_text())
        assert payload["panels"][0]["full_suite"] is True
        assert {s["model"] for s in payload["panels"][0]["scores"]} == {"one", "two", "new-model"}
        assert payload["prediction_sources"][-1]["predictions_sha256"] == sha256(path)


def test_space_title_roundtrips_cli_metadata_and_html_safely(plan, tmp_path):
    pytest.importorskip("rapidfuzz")
    import html

    import yaml

    from data_processing.arabic_reference.__main__ import main
    from data_processing.arabic_reference.release import validate_bundle

    title = 'لوحة: "ASR" & </script><img src=x> __DATA_JSON__ __SPACE_TITLE__'
    output = tmp_path / "named-space"
    main(["leaderboard", "--plan", str(plan), "--output", str(output), "--title", title])
    card = (output / "README.md").read_text()
    assert yaml.safe_load(card.split("---", 2)[1])["title"] == title
    page = (output / "index.html").read_text()
    assert f"<title>{html.escape(title)}</title>" in page
    assert f"<h1>{html.escape(title)}</h1>" in page
    assert "<img src=x>" not in page and "<img src=x>" not in card.split("---", 2)[2]
    assert json.loads((output / "leaderboard.json").read_text())["title"] == title
    assert validate_bundle(output)["title"] == title


@pytest.mark.parametrize("title", ["", "   ", "A\nB", "A\tB", "A\u2028B", "A" * 121])
def test_space_title_rejects_invalid_metadata(plan, tmp_path, title):
    pytest.importorskip("rapidfuzz")
    from data_processing.arabic_reference.leaderboard import build_leaderboard

    output = tmp_path / "space"
    with pytest.raises(ValueError, match="Space title"):
        build_leaderboard(plan, output, title=title)
    assert not output.exists()


def test_launcher_keeps_space_repo_title_and_dataset_independent(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "python"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON": str(binary),
        "ARABIC_REF_ROOT": str(tmp_path / "out"),
        "HF_DATASET_REPO": "test/data",
        "HF_SPACE_REPO": "test/custom-board",
        "HF_SPACE_TITLE": "لوحة ASR: My Board",
    }
    for stage, flag, expected in [
        ("leaderboard", "--title", env["HF_SPACE_TITLE"]),
        ("publish-leaderboard", "--repo-id", env["HF_SPACE_REPO"]),
        ("publish-dataset", "--repo-id", env["HF_DATASET_REPO"]),
    ]:
        proc = subprocess.run(
            ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), stage],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        args = json.loads(proc.stdout)
        assert args[args.index(flag) + 1] == expected


@pytest.mark.parametrize(
    "before,after",
    [
        ("قال صلى الله عليه وسلم", "قال ﷺ"),
        ("قال صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ.", "قال ﷺ."),
        ("صلى\nالله عليه وسلم", "ﷺ"),
        ("ﷺ", "ﷺ"),
        ("صلى الله عليه وآله وسلم", "صلى الله عليه وآله وسلم"),
        ("صلى الله عليه", "صلى الله عليه"),
        ("قال النبي", "قال النبي"),
        ("https://example.org/cafe\u0301", "https://example.org/cafe\u0301"),
        ("ان انا الى اكتب اسم ابن", "ان انا الى اكتب اسم ابن"),
        ("أن إن أكتب اكتب أحمد أحمدُ", "أن إن أكتب اكتب أحمد أحمدُ"),
    ],
)
def test_explicit_editorial_style_is_bounded_and_idempotent(before, after):
    from data_processing.arabic_reference.editorial import ideal_style

    assert ideal_style(before) == after
    assert ideal_style(after) == after


def test_rendering_audit_exposes_losses_hidden_by_wording_normalization():
    from data_processing.arabic_reference.editorial import rendering_audit

    before = "قال أنّه شكّ أنّ ماري ستفعل ذلك."
    after = "قال أنه شكّ أن ماري ستفعل ذلك."
    assert format_signature(before) == format_signature(after)
    audit = rendering_audit(before, after)
    assert audit["same_words_after_explicit_style"] and audit["removed_shadda"] == 2
    # Mark relocation is detected even when the total number of marks stays equal.
    assert rendering_audit("الشّركة", "الشركّة")["removed_shadda"] == 1
    assert rendering_audit("صَلَّى اللَّهُ عَلَيْهِ وَسَلَّمَ", "ﷺ")["removed_shadda"] == 0
    # Different words cannot be aligned as if only punctuation had changed.
    assert not rendering_audit("الخصف", "الخسف")["same_words_after_explicit_style"]


@pytest.mark.parametrize(
    "reference,left,right,flags,allowed",
    [
        ("قال انه شك", "قال انه شكّ.", "قال انه شكّ.", [], True),
        ("قال إنه شكّ", "قال إنه شك.", "قال إنه شك.", [], False),
        ("ان الجو جميل", "إن الجو جميل.", "إن الجو جميل.", [], False),
        ("اكتب رسالة", "أكتب رسالة.", "أكتب رسالة.", [], False),
        ("الخصف", "الخسف", "الخسف", [], False),
        ("في ذمتك", "في ذمّتك.", "في ذمتك.", [], False),
        ("غيرواضح", "غيرواضح", "غيرواضح", [], False),
        ("مرحبا", "مرحبا", "مرحبا", ["duplicate_audio"], False),
        ("قال صلى الله عليه وسلم", "قال ﷺ.", "قال صلّى الله عليه وسلّم.", [], True),
    ],
)
def test_direct_cleanup_requires_shared_rendering_and_keeps_ambiguous_words_for_llm(
    reference, left, right, flags, allowed
):
    from data_processing.arabic_reference.cleanup import direct_proposal

    row = proposal_row()
    row.update(
        text=reference,
        flags=flags,
        candidates=[
            {"name": "first", "text": left, "generation_limit_hit": False},
            {"name": "second", "text": right, "generation_limit_hit": False},
        ],
    )
    proposal = direct_proposal(row)
    assert (proposal is not None) == allowed
    if proposal:
        parsed = parse_proposal(json.dumps(proposal), "stop", row)
        assert parsed["proposed_ideal_text"] == proposal["proposed_ideal_text"]


def test_full_scan_accounts_for_every_row_and_protects_snapshot(inputs, tmp_path):
    from data_processing.arabic_reference.cleanup import scan

    root = tmp_path / "full"
    prepare(inputs, root, full_suite=True)
    output = tmp_path / "scan"
    report = scan(root, output)
    rows = list(read_jsonl(output / "corrected-drafts.jsonl"))
    direct = list(read_jsonl(output / "direct-proposals.jsonl"))
    queue = list(read_jsonl(output / "contextual-queue.jsonl"))
    assert report["selected"] == len(rows) == 18
    assert len(queue) == 18
    assert {r["id"] for r in direct} <= {r["id"] for r in queue}
    assert all(r["contextual_correction_pending"] for r in rows)
    assert all(r["verified_text"] is None and r["ideal_text"] is None for r in rows)
    assert scan(root, output) == report
    with (output / "corrected-drafts.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="changed"):
        scan(root, output)


def test_full_corrector_reviews_every_row_after_initial_cleanup_and_resumes(
    inputs, tmp_path, server
):
    url, state = server
    rows = list(read_jsonl(inputs[1][1]))
    rows[0]["pred_text"] = "مرحبا يا أصدقاء"
    write_jsonl(inputs[1][1], rows)
    root = tmp_path / "full"
    prepare(inputs, root, full_suite=True)
    output = tmp_path / "run"
    kwargs = {
        "url": url,
        "model": "corrector",
        "initial_cleanup": True,
        "concurrency": 3,
        "emit": lambda _: None,
    }
    result = propose(root, output, **kwargs)
    assert result["counts"] == {"proposed": 18}
    assert result["correction_methods"] == {"llm_after_initial_cleanup": 18}
    assert result["initial_cleanup_methods"] == {
        "reference_style_baseline": 1,
        "agreement_rendering": 17,
    }
    assert state["calls"] == 18
    for request in state["requests"]:
        payload = json.loads(request["messages"][1]["content"])
        assert len(payload["candidates"]) == 3
        assert payload["initial_editorial_draft"]["verified"] is False
        assert payload["initial_editorial_draft"]["method"] == "initial_editorial_cleanup"
        assert payload["initial_editorial_draft"]["ideal"]["text"]
    assert propose(root, output, **kwargs)["resumed"] == 18 and state["calls"] == 18
    for record in read_jsonl(output / "proposals.jsonl"):
        assert record["verified_text"] is None and record["ideal_text"] is None
        assert len(record["attempts"]) == 1 and record["usage"]["prompt_tokens"] == 100
        assert record["initial_draft"]["requires_llm_review"] is True


def test_style_application_is_explicit_and_does_not_rewrite_wording():
    from data_processing.arabic_reference.editorial import apply_ideal_style
    from data_processing.arabic_reference.prompts import proposal_checks

    row = proposal_row()
    row["text"] = "قال صلى الله عليه وسلم"
    value = proposal_value(row)
    value.update(proposed_verified_text=row["text"], proposed_ideal_text=row["text"], changes=[])
    adjustments = []
    apply_ideal_style(value, adjustments)
    assert value["proposed_verified_text"] == row["text"]
    assert value["proposed_ideal_text"] == "قال ﷺ"
    assert adjustments[0]["type"] == "editorial_style"
    checks = proposal_checks(value, row, adjustments)
    assert checks["editorial_style_changes"] == adjustments
    assert checks["ideal_follows_explicit_style"]


def test_editorial_checks_flag_content_only_in_ideal_and_candidate_mark_loss():
    from data_processing.arabic_reference.prompts import proposal_checks

    row = proposal_row()
    row["text"] = "الخصف"
    value = proposal_value(row)
    value.update(
        proposed_verified_text="الخصف",
        proposed_ideal_text="الخسف",
        changes=[
            {"kind": "content", "before": "الخصف", "after": "الخسف", "reason": "Disputed word"}
        ],
    )
    assert "declared_content_change_only_in_ideal" in proposal_checks(value, row)["editorial_flags"]
    row["text"] = "الشّركة"
    value = proposal_value(row)
    value.update(proposed_verified_text="الشركة", proposed_ideal_text="الشركة", changes=[])
    checks = proposal_checks(value, row)
    assert "candidate_shadda_loss" in checks["editorial_flags"]


def test_corrector_applies_salawat_style_after_llm_and_retains_raw_reply(inputs, tmp_path, server):
    url, state = server
    phrase = "قال صلّى الله عليه وسلّم"
    for _, path in inputs:
        rows = list(read_jsonl(path))
        for row in rows:
            row.update(text=phrase, pred_text=phrase)
        write_jsonl(path, rows)
    root = tmp_path / "plan"
    prepare(inputs, root, per_dataset=1)

    def reply(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        value.update(proposed_verified_text=phrase, proposed_ideal_text=phrase, changes=[])
        return value

    state["reply"] = reply
    output = tmp_path / "corrected"
    result = propose(
        root, output, url=url, model="corrector", initial_cleanup=True, emit=lambda _: None
    )
    assert state["calls"] == 6
    assert result["explicit_style_records"] == 6
    for record in read_jsonl(output / "proposals.jsonl"):
        assert json.loads(record["raw_response"])["proposed_ideal_text"] == phrase
        assert record["proposal"]["proposed_verified_text"] == phrase
        assert record["proposal"]["proposed_ideal_text"] == "قال ﷺ"
        assert record["proposal_checks"]["editorial_style_changes"]
        assert record["verified_text"] is None and record["ideal_text"] is None


@pytest.fixture
def mixed_recovery_source(plan, tmp_path, server):
    from data_processing.arabic_reference.runtime import record_path

    url, state = server

    def mixed(body):
        if state["calls"] <= 2:
            return '{"reason":"unterminated'
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        if state["calls"] == 3:
            value.update(
                base_candidate=None,
                proposed_verified_text=None,
                proposed_ideal_text=None,
                changes=[],
            )
        return value

    state["reply"] = mixed
    source = tmp_path / "original"
    summary = propose(
        plan,
        source,
        url=url,
        model="corrector",
        concurrency=1,
        initial_cleanup=True,
        emit=lambda _: None,
    )
    assert summary["counts"] == {"failed": 1, "unresolved": 1, "proposed": 4}
    rows = load_plan(plan)[1]
    original = {r["id"]: record_path(source, r["id"]).read_bytes() for r in rows}
    state.pop("reply")
    state["requests"].clear()
    return source, original


def test_recovery_only_requests_failures_and_exports_complete_unchanged_rows(
    plan, tmp_path, server, mixed_recovery_source
):
    from data_processing.arabic_reference.release import load_release_proposals
    from data_processing.arabic_reference.runtime import load_proposals, record_path

    url, state = server
    source, original = mixed_recovery_source
    output = tmp_path / "recovered"
    kwargs = {
        "url": url,
        "model": "corrector",
        "repair_from": source,
        "max_tokens": 8192,
        "emit": lambda _: None,
    }
    summary = propose(plan, output, **kwargs)
    assert summary["counts"] == {"proposed": 5, "unresolved": 1}
    assert summary["recovery"]["targeted_failures"] == 1
    assert summary["recovery"]["carried_forward"] == 5
    assert len(state["requests"]) == 1
    assert state["requests"][0]["max_tokens"] == 8192
    assert "initial_editorial_draft" in json.loads(state["requests"][0]["messages"][1]["content"])
    metadata, rows = load_plan(plan)
    records, identity = load_release_proposals(output, metadata, rows)
    assert len(records) == len(load_proposals(output, metadata, rows)) == 6
    for row in rows:
        key = row["id"]
        assert record_path(source, key).read_bytes() == original[key]
        if key in identity["inherited_records"]:
            assert record_path(output, key).read_bytes() == original[key]
        else:
            assert record_path(output / "source-failures", key).read_bytes() == original[key]
            assert (
                records[key]["recovery_source"]["record_sha256"]
                == json.loads(original[key])["record_sha256"]
            )
        assert records[key]["verified_text"] is records[key]["ideal_text"] is None
    assert propose(plan, output, **kwargs)["resumed"] == 6
    assert len(state["requests"]) == 1
    with pytest.raises(ValueError, match="changed"):
        propose(plan, output, **{**kwargs, "max_tokens": 6144})


@pytest.mark.parametrize("failure", ["length", "json"])
def test_recovery_retries_fresh_context_and_retains_each_attempt(
    plan, tmp_path, server, mixed_recovery_source, failure
):
    url, state = server
    source, _ = mixed_recovery_source

    def reply(body):
        first = len(state["requests"]) == 1
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        state["fail"] = first and failure == "length"
        if first and failure == "json":
            return '{"reason":"bad JSON'
        return value

    state["reply"] = reply
    output = tmp_path / "recovered"
    summary = propose(
        plan, output, url=url, model="corrector", repair_from=source, emit=lambda _: None
    )
    assert summary["counts"] == {"proposed": 5, "unresolved": 1}
    assert len(state["requests"]) == 2
    first, second = state["requests"]
    assert first["temperature"] == 0 and second["temperature"] == 0.2
    assert all(m["role"] != "assistant" for m in second["messages"])
    assert len(second["messages"][-1]["content"]) < 700
    records = list(read_jsonl(output / "proposals.jsonl"))
    repaired = next(r for r in records if "recovery_source" in r)
    assert len(repaired["attempts"]) == 2
    assert repaired["attempts"][0]["validation_error"]
    assert repaired["attempts"][1]["request_settings"]["fresh_context"]
    assert repaired["usage"]["completion_tokens"] == 100


def test_recovery_is_bounded_and_archives_previous_recovery_failures(
    plan, tmp_path, server, mixed_recovery_source
):
    from data_processing.arabic_reference.runtime import record_path

    url, state = server
    source, _ = mixed_recovery_source
    output = tmp_path / "recovered"
    kwargs = {"url": url, "model": "corrector", "repair_from": source, "emit": lambda _: None}
    state["fail"] = True
    summary = propose(plan, output, **kwargs)
    assert summary["counts"] == {"failed": 1, "unresolved": 1, "proposed": 4}
    assert len(state["requests"]) == 3
    failed = next(r for r in read_jsonl(output / "proposals.jsonl") if r["status"] == "failed")
    before = record_path(output, failed["id"]).read_bytes()
    state["fail"] = False
    summary = propose(plan, output, **kwargs)
    assert summary["counts"] == {"proposed": 5, "unresolved": 1}
    archived = (
        output
        / "attempt-history"
        / failed["id"].replace(":", "-")
        / (failed["record_sha256"] + ".json")
    )
    assert json.loads(archived.read_bytes()) == json.loads(before)
    assert len(state["requests"]) == 4
    # A second-generation recovery remains portable and accepts the inherited origins.
    next_output = tmp_path / "recovered-again"
    assert (
        propose(plan, next_output, **{**kwargs, "repair_from": output})["recovery"][
            "targeted_failures"
        ]
        == 0
    )
    assert len(state["requests"]) == 4


def test_recovery_rejects_changed_or_missing_source_and_unsafe_destination(
    plan, tmp_path, server, mixed_recovery_source
):
    from data_processing.arabic_reference.runtime import record_path

    url, state = server
    source, _ = mixed_recovery_source
    kwargs = {"url": url, "model": "corrector", "repair_from": source, "emit": lambda _: None}
    for dest in (source, source / "nested", source.parent):
        with pytest.raises(ValueError, match="separate directory"):
            propose(plan, dest, **kwargs)
    path = record_path(source, load_plan(plan)[1][0]["id"])
    old = path.read_bytes()
    changed = json.loads(old)
    changed["error"] = "tampered"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Changed checkpoint"):
        propose(plan, tmp_path / "changed", **kwargs)
    path.unlink()
    with pytest.raises(ValueError, match="incomplete"):
        propose(plan, tmp_path / "missing", **kwargs)
    path.write_bytes(old)
    assert len(state["requests"]) == 0


@pytest.mark.parametrize("relabel_origin", [False, True])
def test_recovery_detects_tampered_inherited_records_even_after_rehash(
    plan, tmp_path, server, mixed_recovery_source, relabel_origin
):
    from data_processing.arabic_reference.core import write_json
    from data_processing.arabic_reference.release import load_release_proposals
    from data_processing.arabic_reference.runtime import record_path

    url, _ = server
    source, _ = mixed_recovery_source
    output = tmp_path / "recovered"
    propose(plan, output, url=url, model="corrector", repair_from=source, emit=lambda _: None)
    metadata, rows = load_plan(plan)
    identity = json.loads((output / "run.json").read_text())
    key = next(iter(identity["inherited_records"]))
    path = record_path(output, key)
    record = json.loads(path.read_text())
    record.pop("record_sha256")
    record["raw_response"] = "tampered"
    if relabel_origin:
        record["run_sha256"] = digest(identity)
    record["record_sha256"] = digest(record)
    write_json(path, record)
    with pytest.raises(ValueError, match="Changed checkpoint"):
        load_release_proposals(output, metadata, rows)


def test_recovery_launcher_defaults_and_downstream_selection(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / "python"
    binary.write_text(f"#!{sys.executable}\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n")
    binary.chmod(0o755)
    env = {
        **os.environ,
        "PYTHON": str(binary),
        "ARABIC_REF_ROOT": str(tmp_path),
        "ARABIC_REF_PROPOSALS": str(tmp_path / "custom-source"),
        "REF_MAX_TOKENS": "4096",
    }
    for key in ("REF_CONCURRENCY", "REF_REPAIR_MAX_TOKENS", "ARABIC_REF_REPAIR_OUTPUT"):
        env.pop(key, None)
    (tmp_path / "custom-source").mkdir()
    (tmp_path / "custom-source/run.json").write_text("{}")
    for command in ("repair", "review-pack", "export-dataset", "status"):
        result = subprocess.run(
            ["bash", str(root / "scripts/corpus/run_arabic_reference.sh"), command],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        args = json.loads(result.stdout)
        source_arg = "--source" if command == "repair" else "--proposals"
        assert args[args.index(source_arg) + 1] == str(tmp_path / "custom-source")
        if command == "repair":
            assert args[args.index("--max-tokens") + 1] == "8192"
            assert args[args.index("--concurrency") + 1] == "90"
            assert args[args.index("--output") + 1] == str(tmp_path / "proposals-repaired-v4.2")


def test_recovered_dataset_preserves_both_run_origins_and_null_final_columns(
    plan, tmp_path, server, mixed_recovery_source
):
    pq = pytest.importorskip("pyarrow.parquet")
    from data_processing.arabic_reference.release import export_dataset, validate_bundle

    url, _ = server
    source, _ = mixed_recovery_source
    output = tmp_path / "recovered"
    propose(plan, output, url=url, model="corrector", repair_from=source, emit=lambda _: None)
    release = tmp_path / "release"
    manifest = export_dataset(plan, release, proposals=output)
    assert validate_bundle(release)["bundle_sha256"] == manifest["bundle_sha256"]
    rows = [r for p in release.glob("data/*/*.parquet") for r in pq.read_table(p).to_pylist()]
    assert len(rows) == 6
    assert Counter(r["proposal_status"] for r in rows) == {"proposed": 5, "unresolved": 1}
    assert all(r["verified_text"] is r["ideal_text"] is None for r in rows)
    provenance = json.loads((release / "provenance.json").read_text())
    lineage = provenance["corrector_runs"]
    assert [r["records_in_release"] for r in lineage] == [1, 5]
    assert {r["proposal_run_sha256"] for r in rows} == {r["run_sha256"] for r in lineage}
    assert lineage[0]["source_run_sha256"] == lineage[1]["run_sha256"]
    assert str(tmp_path) not in json.dumps(provenance)


def test_verbose_metadata_is_a_warning_without_changing_transcripts():
    from data_processing.arabic_reference.recovery import recovery_metadata_warnings

    value = proposal_value()
    value["reason"] = "A complete but excessively detailed explanation. " * 20
    before = json.dumps(value, ensure_ascii=False)
    warnings = recovery_metadata_warnings(value)
    assert warnings[0]["code"] == "verbose_explanation"
    assert warnings[0]["exceeded"][0]["field"] == "reason"
    assert json.dumps(value, ensure_ascii=False) == before


def test_recovery_keeps_complete_verbose_reply_without_wasting_a_retry(
    plan, tmp_path, server, mixed_recovery_source
):
    url, state = server
    source, _ = mixed_recovery_source

    def reply(body):
        value = proposal_value(payload=json.loads(body["messages"][1]["content"]))
        value["reason"] = "Detailed explanation. " * 30
        return value

    state["reply"] = reply
    output = tmp_path / "recovered"
    result = propose(
        plan, output, url=url, model="corrector", repair_from=source, emit=lambda _: None
    )
    assert result["counts"] == {"proposed": 5, "unresolved": 1}
    assert len(state["requests"]) == 1
    record = next(r for r in read_jsonl(output / "proposals.jsonl") if "recovery_source" in r)
    assert record["proposal"]["reason"] == "Detailed explanation. " * 30
    assert any(
        w.get("code") == "verbose_explanation"
        for w in record["proposal_checks"]["metadata_warnings"]
    )


@pytest.fixture
def saved_reply_source(plan, mixed_recovery_source):
    from data_processing.arabic_reference.core import write_json
    from data_processing.arabic_reference.runtime import record_path

    source, _ = mixed_recovery_source
    records = list(read_jsonl(source / "proposals.jsonl"))
    failed = next(r for r in records if r["status"] == "failed")
    row = next(r for r in load_plan(plan)[1] if r["id"] == failed["id"])
    value = proposal_value(row=row)
    value["reason"] = "Complete saved explanation. " * 20
    raw = json.dumps(value, ensure_ascii=False)
    failed["attempts"] = [
        {
            "number": 1,
            "raw_response": raw,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 50},
            "validation_error": "Recovery explanations exceed limits; give concise final decisions",
        },
        {
            "number": 2,
            "raw_response": '{"reason":"unterminated',
            "finish_reason": "stop",
            "usage": {"completion_tokens": 50},
            "validation_error": "Unterminated string",
        },
    ]
    failed["usage"] = {"completion_tokens": 100}
    failed["raw_response"] = failed["attempts"][-1]["raw_response"]
    failed["error"] = "JSONDecodeError: Unterminated string"
    failed.pop("record_sha256")
    failed["record_sha256"] = digest(failed)
    write_json(record_path(source, failed["id"]), failed)
    write_jsonl(source / "proposals.jsonl", records)
    return source, records, raw


def test_offline_revalidation_recovers_earlier_complete_reply_without_network(
    plan, tmp_path, saved_reply_source, monkeypatch
):
    import shutil

    from data_processing.arabic_reference import runtime
    from data_processing.arabic_reference.release import export_dataset, load_release_proposals
    from data_processing.arabic_reference.revalidate import revalidate_saved

    source, originals, raw = saved_reply_source
    original_hash = sha256(source / "proposals.jsonl")

    def no_network(*args, **kwargs):
        raise AssertionError("Offline revalidation must never contact a model server")

    monkeypatch.setattr(runtime, "doctor", no_network)
    monkeypatch.setattr(runtime, "client_for", no_network)
    # The complete hash-checked aggregate is portable even during a partial records copy.
    shutil.rmtree(source / "records")
    output = tmp_path / "offline"
    summary = revalidate_saved(plan, source, output, emit=lambda _: None)
    assert summary["counts"] == {"proposed": 5, "unresolved": 1}
    assert summary["revalidation_counts"] == {"proposed": 1}
    assert summary["llm_requests"] == summary["verified"] == 0
    assert sha256(source / "proposals.jsonl") == original_hash
    metadata, rows = load_plan(plan)
    records, identity = load_release_proposals(output, metadata, rows)
    assert identity["initial_cleanup"] == "agreement-rendering-v1"
    assert len(records) == len(originals)
    for original in originals:
        record = records[original["id"]]
        assert record["ideal_text"] is record["verified_text"] is None
        if original["status"] != "failed":
            assert record == original
        else:
            assert record["attempts"] == original["attempts"]
            assert record["raw_response"] == raw
            assert (
                record["proposal"]["proposed_ideal_text"] == json.loads(raw)["proposed_ideal_text"]
            )
            assert record["saved_response_revalidation"]["selected_attempt"] == 1
            assert record["saved_response_revalidation"]["attempt_results"][0]["accepted"] is False
            assert "error" not in record
    assert (
        revalidate_saved(plan, source, output, emit=lambda _: None)["counts"] == summary["counts"]
    )
    pytest.importorskip("pyarrow")
    release = tmp_path / "release"
    export_dataset(plan, release, proposals=output)
    provenance = json.loads((release / "provenance.json").read_text())
    assert provenance["corrector"]["processing_mode"] == "saved_response_revalidation"
    assert provenance["corrector"]["llm_requests"] == 0
    assert provenance["corrector"]["generation"] is None
    assert [r["records_in_release"] for r in provenance["corrector_runs"]] == [1, 5]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "changed", "binding"])
def test_offline_revalidation_rejects_incomplete_or_changed_aggregate(
    plan, tmp_path, saved_reply_source, mutation
):
    from data_processing.arabic_reference.revalidate import revalidate_saved

    source, records, _ = saved_reply_source
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records.append(records[0])
    elif mutation == "changed":
        records[0]["raw_response"] = "tampered"
    else:
        records[0].pop("record_sha256")
        records[0]["binding"] = "changed-binding"
        records[0]["record_sha256"] = digest(records[0])
    write_jsonl(source / "proposals.jsonl", records)
    with pytest.raises(ValueError, match=r"incomplete|Changed|duplicate"):
        revalidate_saved(plan, source, tmp_path / "offline", emit=lambda _: None)


def test_offline_revalidation_does_not_salvage_malformed_or_length_limited_replies(
    plan, tmp_path, saved_reply_source
):
    from data_processing.arabic_reference.revalidate import revalidate_saved

    source, records, _ = saved_reply_source
    failed = next(r for r in records if r["status"] == "failed")
    # Even syntactically complete text is rejected if the server reported truncation.
    failed["attempts"][0]["finish_reason"] = "length"
    failed.pop("record_sha256")
    failed["record_sha256"] = digest(failed)
    write_jsonl(source / "proposals.jsonl", records)
    summary = revalidate_saved(plan, source, tmp_path / "offline", emit=lambda _: None)
    assert summary["counts"] == {"failed": 1, "unresolved": 1, "proposed": 4}
    record = next(
        r for r in read_jsonl(tmp_path / "offline/proposals.jsonl") if r["status"] == "failed"
    )
    assert record["saved_response_revalidation"]["selected_attempt"] is None
    assert "proposal" not in record


def test_offline_revalidation_cli_and_inherited_run_chain(
    plan, tmp_path, saved_reply_source, capsys
):
    from data_processing.arabic_reference.__main__ import main
    from data_processing.arabic_reference.revalidate import revalidate_saved

    source, _, _ = saved_reply_source
    output = tmp_path / "offline"
    assert (
        main(["revalidate", "--plan", str(plan), "--source", str(source), "--output", str(output)])
        == 0
    )
    summary = revalidate_saved(plan, output, tmp_path / "offline-again", emit=lambda _: None)
    assert summary["targeted_failures"] == 0 and summary["carried_forward"] == 6
    assert summary["counts"] == {"proposed": 5, "unresolved": 1}


@pytest.fixture
def speculative_release(plan, tmp_path):
    from data_processing.arabic_reference.core import write_json
    from data_processing.arabic_reference.runtime import record_path

    metadata, rows = load_plan(plan)
    source = tmp_path / "source"
    identity = {"manifest_digest": metadata["manifest_digest"], "ids": [r["id"] for r in rows]}
    write_json(source / "run.json", identity)
    for row in rows:
        proposal = proposal_value(row)
        if row == rows[0]:
            proposal.update(
                base_candidate=None,
                proposed_verified_text=None,
                proposed_ideal_text=None,
                changes=[],
            )
        record = {
            "id": row["id"],
            "binding": row["binding"],
            "run_sha256": digest(identity),
            "status": "unresolved" if row == rows[0] else "proposed",
            "proposal": proposal,
            "verified_text": None,
            "ideal_text": None,
        }
        record["record_sha256"] = digest(record)
        write_json(record_path(source, row["id"]), record)
    row = rows[0]
    record = json.loads(record_path(source, row["id"]).read_text())
    draft = {
        "id": row["id"],
        "binding": row["binding"],
        "source_run_sha256": digest(identity),
        "source_record_sha256": record["record_sha256"],
        "text": row["text"],
        "asr_text": None,
        "gqasr_text": None,
        "proposal_status": "speculative_draft",
        "user_requested_speculation": True,
        "correction_method": "assistant_speculative_adjudication",
        "audio_listened": False,
        "audio_review_required": True,
        "include_in_verified_scores": False,
        "verified_text": None,
        "ideal_text": None,
        "verified_text_draft": "مرحبا بكم",
        "ideal_text_draft": "مرحبًا بكم.",
        "confidence": "low",
        "judgment": "User-requested guess without audio",
        "speculative_spans": [{"chosen": "مرحبا", "reason": "Uncertain"}],
        "proposal": proposal_value(row),
        "validation_adjustments": [],
    }
    sidecar = tmp_path / "speculative" / "drafts.jsonl"
    return plan, source, sidecar, draft


def save_speculative_fixture(sidecar, draft, *, duplicate=False, corrupt=False):
    from data_processing.arabic_reference.core import write_json

    draft["speculative_draft_sha256"] = digest(
        {k: v for k, v in draft.items() if k != "speculative_draft_sha256"}
    )
    if corrupt:
        draft["speculative_draft_sha256"] = "0" * 64
    write_jsonl(sidecar, [draft, draft] if duplicate else [draft])
    write_json(
        sidecar.parent / "manifest.json",
        {
            "source_run_sha256": draft["source_run_sha256"],
            "drafts_sha256": sha256(sidecar),
            "audio_listened": False,
            "verified": 0,
            "records": 1,
            "ids": [draft["id"]],
        },
    )


def test_speculative_export_preserves_originals_and_labels_six_case_style_overlay(
    speculative_release, tmp_path
):
    pq = pytest.importorskip("pyarrow.parquet")
    from data_processing.arabic_reference.release import export_dataset, validate_bundle
    from data_processing.arabic_reference.runtime import record_path

    plan, source, sidecar, draft = speculative_release
    save_speculative_fixture(sidecar, draft)
    original = sha256(record_path(source, draft["id"]))
    result = export_dataset(plan, tmp_path / "bundle", proposals=source, speculative_drafts=sidecar)
    assert validate_bundle(tmp_path / "bundle")["bundle_sha256"] == result["bundle_sha256"]
    rows = [
        r
        for p in (tmp_path / "bundle").glob("data/*/*.parquet")
        for r in pq.read_table(p).to_pylist()
    ]
    target = next(r for r in rows if r["id"] == draft["id"])
    assert len(rows) == 6 and sum(r["speculative"] for r in rows) == 1
    assert target["ideal_text_draft"] == draft["ideal_text_draft"]
    assert target["proposal_status"] == "speculative_draft"
    assert target["correction_method"] == "assistant_speculative_adjudication"
    assert target["proposal_record_sha256"] == draft["source_record_sha256"]
    assert target["speculative_draft_sha256"] == draft["speculative_draft_sha256"]
    assert all(r["ideal_text"] is None and not r["audio_reviewed"] for r in rows)
    assert all(r["proposal_status"] == "proposed" for r in rows if not r["speculative"])
    assert sha256(record_path(source, draft["id"])) == original
    assert str(tmp_path) not in json.dumps(rows)
    provenance = json.loads((tmp_path / "bundle" / "provenance.json").read_text())
    assert provenance["speculative_drafts"]["records_in_release"] == 1
    assert provenance["speculative_drafts"]["audio_listened"] is False
    # Merely having an adjacent sidecar never activates it on later exports.
    export_dataset(plan, tmp_path / "without-overlay", proposals=source)
    untouched = [
        r
        for p in (tmp_path / "without-overlay").glob("data/*/*.parquet")
        for r in pq.read_table(p).to_pylist()
    ]
    assert not any(r["speculative"] for r in untouched)
    assert next(r for r in untouched if r["id"] == draft["id"])["ideal_text_draft"] is None


@pytest.mark.parametrize(
    "mutation",
    [
        "binding",
        "source_record",
        "source_run",
        "candidate",
        "verified",
        "listened",
        "score",
        "text",
        "duplicate",
        "hash",
        "file_hash",
        "coverage",
        "missing_source",
    ],
)
def test_speculative_release_rejects_mismatched_or_promoted_guesses(
    speculative_release, tmp_path, mutation
):
    pytest.importorskip("pyarrow")
    from data_processing.arabic_reference.release import export_dataset

    plan, source, sidecar, draft = speculative_release
    if mutation == "binding":
        draft["binding"] = "wrong"
    if mutation == "source_record":
        draft["source_record_sha256"] = "wrong"
    if mutation == "source_run":
        draft["source_run_sha256"] = "wrong"
    if mutation == "candidate":
        draft["asr_text"] = "غير صحيح"
    if mutation == "verified":
        draft["ideal_text"] = draft["ideal_text_draft"]
    if mutation == "listened":
        draft["audio_listened"] = True
    if mutation == "score":
        draft["include_in_verified_scores"] = True
    if mutation == "text":
        draft["ideal_text_draft"] = "مختلف"
    save_speculative_fixture(
        sidecar, draft, duplicate=mutation == "duplicate", corrupt=mutation == "hash"
    )
    if mutation == "file_hash":
        sidecar.write_text(sidecar.read_text() + "\n")
    if mutation == "coverage":
        p = sidecar.parent / "manifest.json"
        metadata = json.loads(p.read_text())
        metadata["records"] = 2
        p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        export_dataset(
            plan,
            tmp_path / "invalid",
            proposals=None if mutation == "missing_source" else source,
            speculative_drafts=sidecar,
        )
    assert not (tmp_path / "invalid").exists()


def test_audio_bundle_preserves_float_samples_texts_and_hf_configurations(inputs, tmp_path):
    import hashlib
    import io

    pq = pytest.importorskip("pyarrow.parquet")
    sf = pytest.importorskip("soundfile")
    hf = pytest.importorskip("datasets")
    from data_processing.arabic_reference.audio_release import package_audio
    from data_processing.arabic_reference.release import export_dataset, validate_bundle

    # Multiple shards and amplitudes outside [-1, 1] catch lossy PCM16 conversions.
    values = list(read_jsonl(inputs[0][1]))
    wave = np.tile(np.array([-1.25, 0.12345679, 1.5, 0], dtype=np.float32), 35000)
    for row in values:
        np.save(row["audio"], wave)
        row["audio_sha256"] = sha256(row["audio"])
        row["actual_duration"] = len(wave) / 16000
    for _, path in inputs:
        write_jsonl(path, values)
    plan = tmp_path / "full"
    prepare(inputs, plan, full_suite=True)
    annotations = tmp_path / "annotations"
    export_dataset(plan, annotations)
    original = validate_bundle(annotations)
    output = tmp_path / "audio-bundle"
    result = package_audio(
        annotations, tmp_path / "audio", output, workers=2, shard_mib=1, emit=lambda _: None
    )
    assert result["audio_included"] is True
    assert result["annotations_bundle_sha256"] == original["bundle_sha256"]
    assert validate_bundle(annotations) == original
    assert len(list(output.glob("data/*/*.parquet"))) == 18
    rows = [r for p in output.glob("data/*/*.parquet") for r in pq.read_table(p).to_pylist()]
    assert len(rows) == 18
    for row in rows:
        decoded, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        assert rate == 16000 and np.array_equal(decoded, wave)
        assert hashlib.sha256(row["audio"]["bytes"]).hexdigest() == row["audio_bytes_sha256"]
        assert row["text"] == "مرحبا بكم" and row["ideal_text"] is None
        assert row["release_id"] == "v4.5-audio-001"
        assert not row["audio_reviewed"] and not row["speculative"]
        assert "/" not in row["audio"]["path"]
    # Exercise the same local README configs and Audio metadata the Hub loader uses.
    ds = hf.load_dataset(str(output), "all", split="test", cache_dir=str(tmp_path / "hf-cache"))
    assert len(ds) == 18 and isinstance(ds.features["audio"], hf.Audio)
    assert ds.features["audio"].sampling_rate == 16000
    data = ds.cast_column("audio", hf.Audio(decode=False))[0]["audio"]
    assert data["bytes"]
    one = hf.load_dataset(str(output), "sada", split="test", cache_dir=str(tmp_path / "hf-cache"))
    assert len(one) == 3
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["audio_encoding"]["exact_float32_samples"] is True
    assert (
        package_audio(
            annotations, tmp_path / "audio", output, workers=1, shard_mib=1, emit=lambda _: None
        )["bundle_sha256"]
        == result["bundle_sha256"]
    )
    with pytest.raises(ValueError, match="different inputs/version"):
        package_audio(annotations, tmp_path / "audio", output, release_id="different")


@pytest.mark.parametrize("mutation", ["hash", "missing", "stereo", "nan", "float64", "duration"])
def test_audio_release_rejects_wrong_recordings_without_leaving_partial_bundle(
    plan, tmp_path, mutation
):
    pytest.importorskip("datasets")
    from data_processing.arabic_reference.audio_release import package_audio
    from data_processing.arabic_reference.release import (
        export_dataset,
        finish_bundle,
        validate_bundle,
    )

    annotations = tmp_path / "annotations"
    export_dataset(plan, annotations)
    row = load_plan(plan)[1][0]
    audio = tmp_path / "audio" / row["dataset"] / f"{row['ordinal']:06d}.npy"
    if mutation == "missing":
        audio.unlink()
    elif mutation == "hash":
        np.save(audio, np.zeros(1600, dtype=np.float32))
    else:
        # Rebind a deliberately malformed test cache to exercise validation after hashing.
        import pyarrow as pa
        import pyarrow.parquet as pq

        wave = np.load(audio)
        if mutation == "stereo":
            wave = np.stack([wave, wave], axis=1)
        if mutation == "nan":
            wave[0] = np.nan
        if mutation == "float64":
            wave = wave.astype(np.float64)
        np.save(audio, wave)
        manifest = validate_bundle(annotations)
        table_path = annotations / "data" / row["dataset"] / "test.parquet"
        table = pq.read_table(table_path)
        values = table.to_pylist()
        values[0]["audio_sha256"] = sha256(audio)
        if mutation == "duration":
            values[0]["duration_seconds"] += 1
        pq.write_table(pa.Table.from_pylist(values, schema=table.schema), table_path)
        (annotations / "release.json").unlink()
        rebound = tmp_path / "rebound"
        finish_bundle(
            annotations,
            rebound,
            {k: v for k, v in manifest.items() if k not in {"files", "bundle_sha256"}},
        )
        annotations = rebound
    with pytest.raises((ValueError, FileNotFoundError)):
        package_audio(
            annotations, tmp_path / "audio", tmp_path / "output", workers=1, emit=lambda _: None
        )
    assert not (tmp_path / "output").exists()


def test_large_audio_upload_uses_verified_disk_snapshot(tmp_path):
    from data_processing.arabic_reference.release import finish_bundle, publish_bundle

    stage = tmp_path / "stage"
    stage.mkdir()
    body = b"x" * (9 * 1024**2)
    (stage / "large.parquet").write_bytes(body)
    bundle = tmp_path / "bundle"
    finish_bundle(stage, bundle, {"repo_type": "dataset", "release_id": "audio-test"})
    snapshots = []

    class InspectHub(PrivateHubStub):
        def create_commit(self, repo_id, **kwargs):
            from pathlib import Path

            operation = next(o for o in kwargs["operations"] if o.path_in_repo == "large.parquet")
            assert isinstance(operation.path_or_fileobj, str)
            snapshot = Path(operation.path_or_fileobj)
            assert snapshot != bundle / "large.parquet"
            snapshots.append(snapshot)
            (bundle / "large.parquet").write_bytes(b"Changed after snapshot")
            assert snapshot.read_bytes() == body
            return super().create_commit(repo_id, **kwargs)

    result = publish_bundle(
        bundle, "test/bench", execute=True, expected_head="existing-head", api=InspectHub()
    )
    assert result["executed"] is True and snapshots and not snapshots[0].exists()


@pytest.fixture
def quality_audio_release(plan, tmp_path):
    pytest.importorskip("datasets")
    from data_processing.arabic_reference.audio_release import package_audio
    from data_processing.arabic_reference.release import export_dataset

    annotations, output = tmp_path / "annotations", tmp_path / "audio-release"
    export_dataset(plan, annotations)
    package_audio(annotations, tmp_path / "audio", output, workers=1, emit=lambda _: None)
    return annotations, output


def test_audio_card_uses_exact_shards_and_reports_real_quality(quality_audio_release):
    import pyarrow.parquet as pq
    import yaml

    from data_processing.arabic_reference.release import publish_bundle

    _, output = quality_audio_release
    quality = json.loads((output / "quality_report.json").read_text())
    assert quality["status"] == "passed" and quality["decoded_audio_rows"] == 6
    assert quality["all_annotation_columns_preserved"] is True
    assert quality["transcript_accuracy_certified"] is False
    assert quality["audio_reviewed_transcripts"] == 0
    card = (output / "README.md").read_text()
    meta = yaml.safe_load(card.split("---", 2)[1])
    paths = meta["configs"][0]["data_files"][0]["path"]
    assert len(paths) == 6 and all("*" not in p for p in paths)
    for path in paths:
        assert (output / path).is_file()
        assert pq.ParquetFile(output / path).schema_arrow.names[0] == "audio"
    assert "**0 audio-reviewed transcripts**" in card
    assert "أ / إ / آ / ؤ / ئ" in card and "ﷺ" in card
    assert publish_bundle(output, "shahink/asr_bench")["executed"] is False
    with pytest.raises(ValueError, match="destination differs"):
        publish_bundle(output, "someone/different")


@pytest.mark.parametrize("mutation", ["text", "audio_bytes", "samples", "duration", "missing"])
def test_written_shard_audit_catches_post_encoding_damage(quality_audio_release, mutation):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from data_processing.arabic_reference.release_quality import audit_audio_shards

    annotations, output = quality_audio_release
    provenance = json.loads((output / "provenance.json").read_text())
    paths = provenance["data_files"]
    ds = next(iter(paths))
    path = output / paths[ds][0]
    table = pq.read_table(path)
    rows = table.to_pylist()
    if mutation == "text":
        rows[0]["text"] = "Changed original"
    if mutation == "audio_bytes":
        rows[0]["audio"]["bytes"] = b"corrupt"
    if mutation == "samples":
        rows[0]["audio_samples_sha256"] = "0" * 64
    if mutation == "duration":
        rows[0]["duration_seconds"] = 1000
    if mutation == "missing":
        rows = []
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    with pytest.raises(ValueError):
        audit_audio_shards(
            output, annotations, paths, provenance["release_id"], emit=lambda _: None
        )


def test_configurable_audio_dataset_title_repo_and_selection(plan, tmp_path):
    import yaml

    from data_processing.arabic_reference.audio_release import package_audio
    from data_processing.arabic_reference.release import export_dataset

    annotations, output = tmp_path / "annotations", tmp_path / "custom"
    export_dataset(plan, annotations)
    package_audio(
        annotations,
        tmp_path / "audio",
        output,
        datasets=["sada"],
        title="My Arabic Speech Collection",
        repo_id="team/custom-audio",
        workers=1,
        emit=lambda _: None,
    )
    card = (output / "README.md").read_text()
    meta = yaml.safe_load(card.split("---", 2)[1])
    assert meta["pretty_name"] == "My Arabic Speech Collection"
    assert [c["config_name"] for c in meta["configs"]] == ["all", "sada"]
    assert '"team/custom-audio"' in card and '"shahink/asr_bench"' not in card
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["selection_scope"] == "dataset_subset"
    assert set(provenance["counts"]) == {"sada"}
    assert provenance["quality"]["rows"] == 1
    with pytest.raises(ValueError, match="different inputs/version"):
        package_audio(annotations, tmp_path / "audio", output, datasets=["sada"], title="Changed")


@pytest.mark.parametrize(
    "mutation", [None, "private", "revision", "missing", "size", "lfs_hash", "git_hash"]
)
def test_remote_release_verification_checks_every_pinned_file(plan, tmp_path, mutation):
    import hashlib
    from types import SimpleNamespace

    from data_processing.arabic_reference.cluster_release import verify_published_bundle
    from data_processing.arabic_reference.release import export_dataset

    bundle = tmp_path / "annotations"
    export_dataset(plan, bundle)
    siblings = []
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        body = path.read_bytes()
        lfs = (
            SimpleNamespace(sha256=hashlib.sha256(body).hexdigest())
            if path.suffix == ".parquet"
            else None
        )
        siblings.append(
            SimpleNamespace(
                rfilename=path.relative_to(bundle).as_posix(),
                size=len(body),
                lfs=lfs,
                blob_id=hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(),
            )
        )
    info = SimpleNamespace(private=True, sha="published-commit", siblings=siblings)
    if mutation == "private":
        info.private = False
    if mutation == "revision":
        info.sha = "different"
    if mutation == "missing":
        info.siblings = siblings[:-1]
    if mutation == "size":
        siblings[0].size += 1
    if mutation == "lfs_hash":
        next(s for s in siblings if s.lfs).lfs.sha256 = "wrong"
    if mutation == "git_hash":
        next(s for s in siblings if not s.lfs).blob_id = "wrong"

    class Hub:
        def repo_info(self, repo_id, **kwargs):
            assert kwargs == {
                "repo_type": "dataset",
                "revision": "published-commit",
                "files_metadata": True,
            }
            return info

    if mutation:
        with pytest.raises(ValueError):
            verify_published_bundle(bundle, "team/audio", "published-commit", Hub())
    else:
        result = verify_published_bundle(bundle, "team/audio", "published-commit", Hub())
        assert result["status"] == "passed" and result["files_verified"] == len(siblings)


def test_cluster_command_packages_selected_datasets_with_configured_name(plan, tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("datasets")
    from data_processing.arabic_reference.release import export_dataset, validate_bundle

    annotations, output = tmp_path / "annotations", tmp_path / "packaged"
    export_dataset(plan, annotations)
    repo = Path(__file__).parents[1]
    env = {
        **os.environ,
        "PYTHON": sys.executable,
        "ARABIC_REF_ROOT": str(tmp_path),
        "ARABIC_REF_ANNOTATIONS_BUNDLE": str(annotations),
        "ARABIC_REF_AUDIO_ROOT": str(tmp_path / "audio"),
        "ARABIC_REF_DATASET_BUNDLE": str(output),
        "ARABIC_REF_RELEASE_ID": "configured-release",
        "HF_OWNER": "test-team",
        "HF_DATASET_NAME": "custom-speech",
        "HF_DATASET_TITLE": "Arabic Speech · Test Edition",
        "REF_DATASETS": "sada commonvoice",
        "REF_AUDIO_WORKERS": "2",
    }
    env.pop("HF_DATASET_REPO", None)
    result = subprocess.run(
        ["bash", str(repo / "scripts/corpus/publish_arabic_dataset.sh"), "package"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    manifest = validate_bundle(output)
    assert manifest["datasets"] == ["sada", "commonvoice"]
    assert manifest["quality_status"] == "passed"
    assert manifest["package_config"]["dataset_repo"] == "test-team/custom-speech"
    assert manifest["package_config"]["dataset_title"] == env["HF_DATASET_TITLE"]
    assert not list(tmp_path.glob("publication-*.json"))
