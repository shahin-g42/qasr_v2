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
            assert body["temperature"] == 0 and not body["chat_template_kwargs"]["enable_thinking"]
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


@pytest.mark.parametrize("stage", ["doctor", "smoke", "run"])
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
    assert "Proposal metadata was normalized" in (pack / "index.html").read_text()
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
