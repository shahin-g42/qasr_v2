"""Offline fixtures are test evidence only, never production human sign-off."""

from __future__ import annotations

import copy

import pytest

from jev_contracts import ValidationError
from jev_data.arabic import normalized_view, numeral_to_ascii
from jev_data.dedup import validate_splits
from jev_data.registry import Registry


def registry():
    return Registry.from_dict({"schema_version": 1, "sources": [{
        "id": "commissioned", "status": "approved", "url": "internal://fixture-source",
        "revision": "fixture-v1", "sha256": "a" * 64, "license_ref": "fixture-rights",
        "attribution": "Test authors", "intended_uses": [
            "training", "evaluation", "redistribution", "derivatives"],
        "training": True, "redistribution": True, "derivatives": True, "consent": True,
        "approval_ref": "test-only-approval", "data_class": "commissioned",
        "original_language": "ar", "original_split": "unassigned",
        "collected_at": "2026-01-01", "related_sources": [],
    }]})


def row(identifier="r1", split="train"):
    meta = {"source_pins": registry().pins(["commissioned"])}
    if split == "calibration":
        meta["calibration"] = {"schema_version": 1, "subset": "temperature_fit"}
    return {
        "schema_version": 2, "id": identifier, "family_id": "family-" + identifier,
        "split": split, "task": "numeric_policy", "language": "ar", "variety": "MSA",
        "state": "ليس أَلِف مثل الف. قيمة الطلب ١٠٠ ريال.",
        "question": {"type": "choice", "instructions": "هل يكفي الدليل؟", "options": [
            {"id": "yes", "description": "نعم"}, {"id": "no", "description": "لا"}]},
        "target": {"kind": "hard", "option_id": "no"},
        "provenance": {"origin": "native_arabic_authored", "primary_route": "human",
                       "license_ref": "fixture-rights", "source_ids": ["commissioned"],
                       "derivation_ids": [], "evidence_ref": "test://evidence",
                       "native_origin": True, "human_authored": True,
                       "localized": False, "synthetic": False, "metadata": meta},
        "quality": {"tier": "bronze", "audit_ref": "pending://human"},
    }


def test_registry_strict_rights_and_unknown_sources():
    registry().validate_row(row())
    for key, value in [("consent", "true"), ("training", 1), ("revision", "main")]:
        data = registry().to_dict()
        data["sources"][0][key] = value
        with pytest.raises(ValidationError):
            Registry.from_dict(data)
    data = registry().to_dict()
    data["sources"][0]["status"] = "pending"
    with pytest.raises(ValidationError):
        Registry.from_dict(data).validate_row(row())
    with pytest.raises(ValidationError):
        registry().source("unknown-public-source")
    data = registry().to_dict()
    data["sources"][0]["redistribution"] = False
    data["sources"][0]["intended_uses"].remove("redistribution")
    with pytest.raises(ValidationError):
        Registry.from_dict(data).validate_row(row(), release=True)


@pytest.mark.parametrize("kind", ["parent_ids", "translation_ids", "document_ids", "group_ids"])
def test_lineage_across_inputs(kind):
    left, right = row("a"), row("b", "development")
    if kind in {"parent_ids", "translation_ids"}:
        right["provenance"]["metadata"]["lineage"] = {"schema_version": 1, kind: ["a"]}
    else:
        for item in (left, right):
            item["provenance"]["metadata"]["lineage"] = {"schema_version": 1, kind: ["shared"]}
    with pytest.raises(ValidationError, match="cross protected"):
        validate_splits([left, right])


def test_calibration_and_explicit_union_disjointness():
    left, right = row("a", "calibration"), row("b", "calibration")
    right["provenance"]["metadata"]["calibration"]["subset"] = "risk_verification"
    with pytest.raises(ValidationError):
        validate_splits([left, right], [["a", "b"]])
    right = copy.deepcopy(left)
    with pytest.raises(ValidationError, match="duplicate"):
        validate_splits([left, right])


def test_arabic_fidelity_and_numerals():
    value = row()["state"]
    assert "أَلِف" in normalized_view(value)
    assert "ليس" in normalized_view(value)
    assert "١٠٠" in normalized_view(value)
    assert numeral_to_ascii("−١٢٫٥") == "-12.5"
    assert numeral_to_ascii("۱۲") == "12"
    for bad in ["١,٠٠٠", "١e٣", "١٢ ريال", "NaN", "1/2"]:
        with pytest.raises(ValidationError):
            numeral_to_ascii(bad)


def test_duplicate_report_buckets_by_language_task_and_variety():
    from jev_data.dedup import duplicate_report

    base = ("تقرير الحالة للطلب رقم ٤٢١ المقدم من العميل يتضمن قيمة المبلغ "
            "الإجمالي المطلوب لصرفه هذا الشهر.")
    tweaked = base.replace("٤٢١", "٤٢٢")

    def near_pairs(mutate):
        left, right = row("a"), row("b")
        left["state"], right["state"] = base, tweaked
        mutate(right)
        return duplicate_report([left, right])["near"]

    # Same (language, task, variety) bucket: the near pair needs review.
    assert near_pairs(lambda right: None) == [["a", "b"]]
    # Cross-variety pairs are same-family by construction; cross-task surfaces
    # are structurally different, so neither is ever scheduled for comparison.
    assert near_pairs(lambda right: right.update(variety="Egyptian")) == []
    assert near_pairs(lambda right: right.update(task="budget_check")) == []


def test_exact_duplicates_are_detected_across_buckets():
    from jev_data.dedup import duplicate_report

    left, right = row("a"), row("b")
    right["variety"] = "Egyptian"
    report = duplicate_report([left, right])
    assert report["exact"] == [["a", "b"]]
    assert report["near"] == []


def world(index=1, operator=">"):
    return {"schema_version": 1, "kind": "comparison", "family_id": f"family-{index}",
            "template_id": f"independent-fixture-policy-{index}", "entity": f"طلب رقم {index}",
            "quantity": {"value": "١٠٠", "unit": "SAR"},
            "threshold": {"value": "100", "unit": "SAR"}, "operator": operator}


def candidate(index=1):
    from jev_data.generate import oracle_candidate

    return oracle_candidate(world(index), identifier=f"oracle-{index}", source_id="commissioned",
                            registry=registry())


def human_vote(item, job, reviewer_id, label=None):
    from jev_contracts import renderable_input
    from jev_data.io import digest

    return {"job_id": job, "reviewer": {"id": reviewer_id, "version": "test-v1",
                                      "native_varieties": ["MSA"], "qualifications": []},
            "rubric_version": "test-rubric-v1", "input_hash": digest(renderable_input(item)),
            "label": label or item["target"]["option_id"], "independent": True,
            "unambiguous": True, "surface_faithful": True, "evidence": "Offline test fixture",
            "elapsed_seconds": 60, "flags": []}


def human_review(items):
    from jev_data.annotation import export_jobs, import_reviews

    bundle = export_jobs(items, rubric_version="test-rubric-v1")
    by_id = {item["id"]: item for item in items}
    submissions = [{"job_id": job, "votes": [
        human_vote(by_id[mapping["row_id"]], job, "reviewer-a"),
        human_vote(by_id[mapping["row_id"]], job, "reviewer-b"),
    ], "adjudication": None} for job, mapping in bundle["private_map"].items()]
    return import_reviews(items, bundle, submissions)


def test_exact_oracle_independent_boundaries_units_and_surface():
    from jev_data.oracle import check_oracle, independent_solve, solve

    for operator, expected in [("<", "ineligible"), ("<=", "eligible"),
                               (">", "ineligible"), (">=", "eligible"),
                               ("==", "eligible"), ("!=", "ineligible")]:
        symbolic = world(operator=operator)
        assert solve(symbolic) == independent_solve(symbolic) == expected
    for value in range(-10, 11):
        for operator in ["<", "<=", ">", ">=", "==", "!="]:
            symbolic = world(operator=operator)
            symbolic["quantity"] = {"value": str(value), "unit": "m"}
            symbolic["threshold"] = {"value": "0", "unit": "cm"}
            assert solve(symbolic) == independent_solve(symbolic)
    symbolic = world(operator="==")
    symbolic["quantity"] = {"value": "١٠٠٠٠", "unit": "halala"}
    assert solve(symbolic) == "eligible"
    symbolic["quantity"] = None
    assert solve(symbolic) == "insufficient"
    symbolic["quantity"] = {"value": "100", "unit": "m"}
    with pytest.raises(ValidationError):
        solve(symbolic)
    item = candidate()
    check_oracle(item)
    item["state"] = item["state"].replace("أكبر من", "أكبر من أو يساوي")
    with pytest.raises(ValidationError, match="surface"):
        check_oracle(item)


def test_human_gold_requires_distinct_literal_blind_votes():
    from jev_data.annotation import export_jobs, import_reviews, validate_gold

    item = candidate()
    assert item["quality"]["tier"] == "bronze"
    gold = human_review([item])[0]
    validate_gold(gold)
    assert gold["provenance"]["human_authored"] is False
    assert len(gold["quality"]["metadata"]["annotation"]["votes"]) == 2
    bundle = export_jobs([item], rubric_version="test-rubric-v1")
    assert set(bundle["jobs"][0]["input"]) == {"state", "question"}
    assert item["id"] not in bundle["jobs"][0]["job_id"]
    job = bundle["jobs"][0]["job_id"]
    votes = [human_vote(item, job, "same"), human_vote(item, job, "same")]
    with pytest.raises(ValidationError, match="duplicate human"):
        import_reviews([item], bundle, [{"job_id": job, "votes": votes, "adjudication": None}])
    votes[1]["reviewer"]["id"] = "other"
    votes[1]["independent"] = "true"
    with pytest.raises(ValidationError, match="literal boolean"):
        import_reviews([item], bundle, [{"job_id": job, "votes": votes, "adjudication": None}])
    votes[1]["independent"] = True
    votes[1]["label"] = "eligible"
    quarantined = import_reviews([item], bundle, [{"job_id": job, "votes": votes,
                                                  "adjudication": None}])[0]
    assert quarantined["quality"]["tier"] == "bronze"
    fake = copy.deepcopy(item)
    fake["quality"]["tier"] = "gold"
    with pytest.raises(ValidationError, match="actual human"):
        validate_gold(fake)


def teacher_config(family="family-a"):
    from jev_data.teachers import TeacherConfig

    return TeacherConfig.from_dict({
        "name": family, "family": family, "model": "test-model", "revision": "test-revision",
        "endpoint": "http://localhost:9000/v1", "api_key_env": None, "timeout_seconds": 1,
        "transport_retries": 1, "max_content_attempts": 3, "max_requests": 10,
        "concurrency": 2, "max_tokens": 200, "temperature": 0,
        "allowed_data_classes": ["commissioned"], "qualification_ref": "test-only-qualification",
    })


def completion(value, finish="stop"):
    import json

    return {"choices": [{"finish_reason": finish,
                         "message": {"role": "assistant", "content": json.dumps(value)}}]}


def test_teacher_blind_dual_family_cache_and_malformed(tmp_path):
    import asyncio

    import httpx

    from jev_data.io import read_json
    from jev_data.teachers import TeacherClient
    from jev_data.verify import blind_payload, validate_silver, verify

    item = candidate()
    item["provenance"]["metadata"]["secret"] = {"schema_version": 1, "rationale": "do not render"}
    sent = []
    vote = {"label": "ineligible", "independent": True, "unambiguous": True, "evidence": "boundary"}

    def handler(request):
        import json

        request_data = json.loads(request.content)
        user_content = request_data["messages"][1]["content"]
        payload = json.loads(user_content[user_content.index("{"):])
        sent.append(payload)
        assert set(payload) == {"state", "question"}
        return httpx.Response(200, json=completion(vote))

    clients = [TeacherClient(teacher_config(family), tmp_path / family, allow_network=True,
                             transport=httpx.MockTransport(handler)) for family in ["a", "b"]]
    result = asyncio.run(verify(item, registry(), clients))
    validate_silver(result)
    assert result["quality"]["tier"] == "silver"
    assert all(payload == blind_payload(item) for payload in sent)
    asyncio.run(verify(item, registry(), clients))
    assert len(sent) == 2
    assert sum(client.cache_hits for client in clients) == 2
    with pytest.raises(ValidationError, match="two distinct"):
        asyncio.run(verify(item, registry(), [clients[0], clients[0]]))
    vote["independent"] = "true"
    bad = [TeacherClient(teacher_config(family), tmp_path / (family + "bad"), allow_network=True,
                         transport=httpx.MockTransport(handler)) for family in ["a", "b"]]
    result = asyncio.run(verify(item, registry(), bad))
    assert result["quality"]["tier"] == "bronze"
    assert sum(client.requests for client in bad) == 6
    cache = next((tmp_path / "a").glob("*.json"))
    assert read_json(cache)["response_hash"]


def test_teacher_single_family_mode_recorded_and_quarantine(tmp_path):
    import asyncio

    import httpx

    from jev_data.teachers import TeacherClient
    from jev_data.verify import validate_silver, verify

    item = candidate()
    vote = {"label": "ineligible", "independent": True, "unambiguous": True, "evidence": "boundary"}

    def handler(request):
        return httpx.Response(200, json=completion(vote))

    solo = TeacherClient(teacher_config("solo"), tmp_path / "solo", allow_network=True,
                         transport=httpx.MockTransport(handler))
    result = asyncio.run(verify(item, registry(), [solo], mode="single_family"))
    validate_silver(result)
    assert result["quality"]["tier"] == "silver"
    evidence = result["quality"]["metadata"]["verification"]
    assert evidence["mode"] == "single_family"
    assert len(evidence["judgments"]) == 1
    with pytest.raises(ValidationError, match="exactly one"):
        asyncio.run(verify(candidate(), registry(), [solo, solo], mode="single_family"))
    with pytest.raises(ValidationError, match="two distinct"):
        asyncio.run(verify(candidate(), registry(), [solo]))
    vote["label"] = "eligible"
    disagree = TeacherClient(teacher_config("solo-b"), tmp_path / "solo-b", allow_network=True,
                             transport=httpx.MockTransport(handler))
    quarantined = asyncio.run(verify(candidate(), registry(), [disagree], mode="single_family"))
    assert quarantined["quality"]["tier"] == "bronze"
    assert quarantined["quality"]["metadata"]["disposition"]["reasons"] == [
        "teacher_author_disagreement"]


def test_teacher_unconfigured_and_bad_content(tmp_path):
    import asyncio

    from jev_data.teachers import TeacherClient, TeacherConfig, content

    with pytest.raises(ValidationError):
        TeacherConfig.from_dict(
            {**teacher_config().identity(), "api_key_env": None, "endpoint": None})
    client = TeacherClient(teacher_config(), tmp_path)
    with pytest.raises(ValidationError, match="explicitly enabled"):
        asyncio.run(client.complete({}, prompt="test", data_class="commissioned"))
    for raw in [{}, completion({}, "length"), {"choices": []}]:
        with pytest.raises(ValidationError):
            content(raw)


def quality_evidence(items, plan):
    from jev_data.io import digest

    required = {key for sample in plan["samples"].values()
                for ids in sample["families"].values() for key in ids}
    return [{"id": item["id"], "row_hash": digest(item), "plan_hash": digest(plan),
             "reviewer_id": "independent-auditor", "reviewer_version": "test-v1",
             "native_variety": item["variety"], "independent": True, "critical_error": False,
             "evidence": "Offline quality test fixture, not real annotation evidence",
             "elapsed_seconds": 60} for item in items if item["id"] in required]


def release_approval(items, report):
    from jev_data.io import dataset_identity, digest

    return {"schema_version": 1, "approved": True, "approver": "test-only-release-owner",
            "approval_ref": "test-only-signoff", "dataset_hash": dataset_identity(items)["sha256"],
            "audit_hash": digest(report), "registry_hash": digest(registry().to_dict()),
            "coverage": {"planned_ids": [item["id"] for item in items],
                         "accepted_ids": [item["id"] for item in items], "quarantined": [],
                         "excluded": [], "pending_ids": [], "accepted_subset_approved": False}}


@pytest.fixture(scope="module")
def audited_fixture():
    from jev_data.curate import audit, audit_plan
    from jev_data.generate import oracle_candidate
    from jev_data.io import digest

    candidates = []
    for index in range(200):
        symbolic = world(index)
        symbolic["entity"] += "، مرجع الحالة " + digest([index, "fixture"])
        candidates.append(oracle_candidate(symbolic, identifier=f"r-{index}",
                                           source_id="commissioned", registry=registry()))
    reviewed = human_review(candidates)
    plan = audit_plan(reviewed, registry(), seed="preregistered-fixture")
    report = audit(reviewed, registry(), plan=plan, reviews=quality_evidence(reviewed, plan))
    return reviewed, report


def test_e2e_oracle_review_audit_freeze_load(tmp_path, audited_fixture):
    from jev_contracts import verify_manifest
    from jev_data.freeze import freeze, load_release

    reviewed, report = audited_fixture
    assert report["passed"], report["curation"]["errors"][:3]
    assert report["results"]["overall"]["wilson_upper_95"] < 0.02
    approval = release_approval(reviewed, report)
    root = tmp_path / "release"
    manifest = freeze(root, reviewed, registry(), report, approval=approval)
    assert manifest["schema_version"] == 1
    assert manifest["metadata"] == {"kind": "dataset_release", "approved": True,
                                    "schema_version": 2, "splits": {"train": "train.jsonl"},
                                    "audit": "audit.json", "registry": "registry.json"}
    assert verify_manifest(root) == manifest
    assert load_release(root, "train") == reviewed
    with pytest.raises(FileExistsError):
        freeze(root, reviewed, registry(), report, approval=approval)
    with pytest.raises(ValidationError, match="sealed test"):
        load_release(root, "test")
    (root / "train.jsonl").write_text("{}\n")
    with pytest.raises(ValidationError):
        load_release(root, "train")


def test_audit_small_missing_and_reused_evidence(tmp_path, audited_fixture):
    from jev_data.curate import audit, audit_plan, validate_audit
    from jev_data.freeze import freeze

    reviewed = human_review([candidate()])
    plan = audit_plan(reviewed, registry(), seed="small-fixture")
    missing = audit(reviewed, registry(), plan=plan, reviews=[])
    assert not missing["passed"] and missing["missing_review_ids"]
    tiny = audit(reviewed, registry(), plan=plan, reviews=quality_evidence(reviewed, plan))
    assert not tiny["passed"]
    with pytest.raises(ValidationError, match="has not passed"):
        freeze(tmp_path / "tiny", reviewed, registry(), tiny,
               approval=release_approval(reviewed, tiny))
    full, report = audited_fixture
    altered = copy.deepcopy(full)
    altered[0]["state"] += " Changed"
    with pytest.raises(ValidationError, match="different data"):
        validate_audit(altered, registry(), report)
    fake = copy.deepcopy(report)
    fake["reviews"][0]["critical_error"] = "false"
    with pytest.raises(ValidationError, match="literal boolean"):
        validate_audit(full, registry(), fake)


def test_curate_near_resolution_and_exact_overlap():
    from jev_data.curate import curate
    from jev_data.io import digest

    items = human_review([candidate(1), candidate(2)])
    report = curate(items, registry())
    assert not report["passed"]
    assert any(item["reason"] == "near_duplicate_requires_review" for item in report["errors"])
    resolution = {"ids": [item["id"] for item in items],
                  "row_hashes": {item["id"]: digest(item) for item in items},
                  "reviewer_id": "test-reviewer", "independent": True,
                  "decision": "distinct", "reason": "Test-only separate policy sources"}
    assert curate(items, registry(), resolutions=[resolution])["passed"]
    resolution["row_hashes"][items[0]["id"]] = "wrong"
    with pytest.raises(ValidationError, match="different rows"):
        curate(items, registry(), resolutions=[resolution])


def test_curate_report_is_input_order_canonical():
    from jev_data.curate import curate
    from jev_data.generate import oracle_candidate

    percent_world = {**world(3), "quantity": {"value": "٤٠", "unit": "percent"},
                     "threshold": {"value": "٤١", "unit": "percent"}}
    items = human_review([candidate(1), candidate(2),
                          oracle_candidate(percent_world, identifier="oracle-3",
                                           source_id="commissioned", registry=registry())])
    forward = curate(items, registry())
    backward = curate(list(reversed(items)), registry())
    assert forward == backward
    assert any(item["flag"] == "negation_present" for item in forward["flags"])


def test_state_ownership_retry_quarantine_and_immutable(tmp_path):
    from jev_data.state import ShardState

    with pytest.raises(ValidationError):
        ShardState(tmp_path / "unsafe", "owner")
    with ShardState(tmp_path / "shard", "owner-a", local_storage=True) as state:
        with pytest.raises(ValueError, match="already owned"):
            ShardState(tmp_path / "shard", "owner-b", local_storage=True)
        first = state.plan("opaque/id", {"input": 1})
        assert first == state.plan("opaque/id", {"input": 1})
        with pytest.raises(ValidationError):
            state.plan("opaque/id", {"input": 2})
        state.transition("opaque/id", "quarantined", reason="malformed response")
        assert state.counts()["counts"]["quarantined"] == 1
        state.retry("opaque/id")
        state.transition("opaque/id", "authored")
        state.transition("opaque/id", "verified")
        manifest = state.publish("opaque/id", {"accepted": 1})
        assert state.publish("opaque/id", {"accepted": 1}) == manifest
        with pytest.raises(ValidationError, match="immutable"):
            state.publish("opaque/id", {"accepted": 2})
        with pytest.raises(ValidationError):
            state.retry("opaque/id")
        assert state.counts()["pending"] == 0
    with ShardState(tmp_path / "shard", "owner-b", local_storage=True) as state:
        assert state.counts()["counts"]["accepted"] == 1
        assert state.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        for attempt in range(3):
            if attempt == 0:
                state.plan("failed", {"input": 2})
            else:
                state.retry("failed")
            state.transition("failed", "quarantined", reason="bad content")
        with pytest.raises(ValidationError, match="budget"):
            state.retry("failed")


def test_all_pending_configs_and_cli_help(capsys):
    from pathlib import Path
    from urllib.parse import urlsplit

    from jev_data.cli import main
    from jev_data.curate import policy_config
    from jev_data.io import read_config

    root = Path(__file__).resolve().parents[2]
    config = root / "configs/data"
    sources = Registry.load(config / "sources.yaml")
    assert len(sources.sources) == 31
    # The shipped registry is all-pending except the two commissioned oracle
    # template inventories, which are approved in place with the tasks.yaml hash.
    approved = sorted(s.id for s in sources.sources.values() if s.status == "approved")
    assert approved == ["oracle-templates", "oracle-templates-v2"]
    assert all(source.status == "pending"
               for key, source in sources.sources.items()
               if key not in {"oracle-templates", "oracle-templates-v2"})
    policy_config(read_config(config / "quality.yaml"))
    # The shipped teacher config reflects the single-family deployment: the
    # author and verifier roles are wired to one localhost server, and every
    # endpoint must stay same-node. Verification uses single-family mode.
    teachers = read_config(config / "teachers.yaml")["teachers"]
    assert {"author", "verifier_a"} <= set(teachers)
    for teacher in teachers.values():
        endpoint = urlsplit(teacher["endpoint"])
        assert endpoint.scheme == "http"
        assert endpoint.hostname in {"localhost", "127.0.0.1", "::1"}
    assert main(["sources", "--registry", str(config / "sources.yaml")]) == 0
    assert "pending" in capsys.readouterr().out
    for command in ["sources", "ingest", "plan", "annotate", "generate", "verify", "curate",
                    "freeze", "audit", "retry", "acquire"]:
        with pytest.raises(SystemExit) as exc:
            main([command, "--help"])
        assert exc.value.code == 0


def test_local_ingest_pins_and_immutable_source(tmp_path):
    import hashlib

    from jev_contracts import verify_manifest
    from jev_data.ingest import ingest
    from jev_data.io import jsonl_bytes, read_rows

    item = row()
    item["provenance"]["metadata"].pop("source_pins")
    raw = jsonl_bytes([item])
    source_path = tmp_path / "raw.jsonl"
    source_path.write_bytes(raw)
    data = registry().to_dict()
    data["sources"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    sources = Registry.from_dict(data)
    result = ingest(source_path, sources, "commissioned", output=tmp_path / "ingested")
    assert source_path.read_bytes() == raw
    assert result[0]["state"] == item["state"]
    assert read_rows(tmp_path / "ingested/validated") == result
    verify_manifest(tmp_path / "ingested")
    source_path.write_bytes(raw + b" ")
    with pytest.raises(ValidationError, match="checksum"):
        ingest(source_path, sources, "commissioned")


def test_plan_determinism_and_calibration_subgroups():
    from pathlib import Path

    from jev_data.io import read_config
    from jev_data.planning import plan

    cards = read_config(
        Path(__file__).resolve().parents[2] / "configs/data/tasks.yaml")["task_cards"]
    items = [candidate(index) for index in range(15)]
    groups = [[items[0]["id"], items[1]["id"]]]
    first = plan(items, seed="frozen", cards=cards, weights={"calibration": 1}, union_groups=groups)
    second = plan(list(reversed(items)), seed="frozen", cards=cards,
                  weights={"calibration": 1}, union_groups=groups)
    assert first == second
    validate_splits(first["rows"])
    assert first["rows"][0]["provenance"]["metadata"]["calibration"]["subset"]
    assert all(item["option_count"] == 3 for item in first["slots"])


def test_test_review_requires_three_plus_adjudication():
    from jev_data.annotation import export_jobs, import_reviews, validate_gold
    from jev_data.io import digest

    item = row("fresh-test", "test")
    with pytest.raises(ValidationError):
        export_jobs([item], rubric_version="test-rubric-v1", allow_test=True)
    bundle = export_jobs([item], rubric_version="test-rubric-v1", allow_test=True,
                         purpose="evaluation")
    job = bundle["jobs"][0]["job_id"]
    votes = [human_vote(item, job, name) for name in ["a", "b", "c"]]
    submission = {"job_id": job, "votes": votes, "adjudication": None}
    result = import_reviews([item], bundle, [submission], allow_test=True, purpose="evaluation")
    assert result[0]["quality"]["tier"] == "bronze"
    submission["adjudication"] = {
        "vote": human_vote(item, job, "adjudicator"), "resolved": True,
        "reason": "All evidence checked independently", "votes_hash": digest(votes),
    }
    result = import_reviews([item], bundle, [submission], allow_test=True, purpose="evaluation")
    validate_gold(result[0])
    submission["votes"] = votes[:2]
    submission["adjudication"]["votes_hash"] = digest(votes[:2])
    result = import_reviews([item], bundle, [submission], allow_test=True, purpose="evaluation")
    assert result[0]["quality"]["tier"] == "bronze"


def test_active_learning_train_only_finite_and_diversity():
    from jev_data.active_learning import acquire
    from jev_data.io import digest

    items = human_review([candidate(index) for index in range(8)])
    predictions = [{"id": item["id"], "row_hash": digest(item), "model_revision": "test-r1",
                    "probabilities": {"eligible": 0.98, "ineligible": 0.01, "insufficient": 0.01}}
                   for item in items]
    result = acquire(items, predictions, registry(), count=4, seed="acquire")
    assert len(result["selected"]) == 4 and result["approved"] is False
    assert any("random_diversity_control" in item["flags"] for item in result["selected"])
    assert any("high_confidence_error" in item["flags"] for item in result["selected"])
    predictions[0]["probabilities"]["eligible"] = float("nan")
    with pytest.raises(ValidationError):
        acquire(items, predictions, registry(), count=4, seed="acquire")
    with pytest.raises(ValidationError, match="cannot mine"):
        acquire([row("protected", "calibration")], [], registry(), count=1, seed="acquire")


def test_cli_oracle_annotation_audit_and_failclosed_freeze(tmp_path):
    from jev_contracts import atomic_write_json
    from jev_data.cli import main
    from jev_data.io import read_json, read_rows, read_verified_json

    sources, symbolic = tmp_path / "registry.json", tmp_path / "world.json"
    atomic_write_json(sources, registry().to_dict())
    atomic_write_json(symbolic, world())
    generated = tmp_path / "generated"
    assert main(["generate", "--mode", "oracle", "--input", str(symbolic), "--registry",
                 str(sources), "--source", "commissioned", "--identifier", "cli-row",
                 "--output", str(generated)]) == 0
    jobs = tmp_path / "jobs"
    assert main(["annotate", "export", "--input", str(generated), "--registry", str(sources),
                 "--rubric-version", "test-rubric-v1", "--output", str(jobs)]) == 0
    item = read_rows(generated)[0]
    job = read_verified_json(jobs, "private.json")["jobs"][0]["job_id"]
    submissions = tmp_path / "votes.json"
    atomic_write_json(submissions, [{"job_id": job, "votes": [human_vote(item, job, name)
                                                             for name in ["a", "b"]],
                                     "adjudication": None}])
    reviewed = tmp_path / "reviewed"
    assert main(["annotate", "import", "--input", str(generated), "--registry", str(sources),
                 "--jobs", str(jobs), "--reviews", str(submissions),
                 "--output", str(reviewed)]) == 0
    plan_path = tmp_path / "audit-plan.json"
    assert main(["audit", "--phase", "plan", "--input", str(reviewed), "--registry", str(sources),
                 "--seed", "preregistered", "--output", str(plan_path)]) == 0
    evidence_path = tmp_path / "audit-reviews.json"
    atomic_write_json(evidence_path, quality_evidence(read_rows(reviewed), read_json(plan_path)))
    report_path = tmp_path / "audit.json"
    assert main(["audit", "--input", str(reviewed), "--registry", str(sources),
                 "--audit-plan", str(plan_path), "--reviews", str(evidence_path),
                 "--output", str(report_path)]) == 2
    assert read_json(report_path)["passed"] is False
    approval_path = tmp_path / "approval.json"
    atomic_write_json(approval_path, release_approval(read_rows(reviewed), read_json(report_path)))
    assert main(["freeze", "--input", str(reviewed), "--registry", str(sources),
                 "--audit", str(report_path), "--approval", str(approval_path),
                 "--output", str(tmp_path / "release")]) == 2
    assert not (tmp_path / "release").exists()


def test_teacher_transport_retry_budget_and_cache_corruption(tmp_path):
    import asyncio

    import httpx

    from jev_data.teachers import TeacherClient

    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(503 if len(calls) == 1 else 200, json=completion({"ok": True}))

    client = TeacherClient(teacher_config(), tmp_path, allow_network=True,
                           transport=httpx.MockTransport(transport))
    result = asyncio.run(client.complete({}, prompt="test", data_class="commissioned"))
    assert result and client.requests == 2 and client.transport_failures == 1
    cache = next(tmp_path.glob("*.json"))
    cache.write_text('{"schema_version":1,"request_hash":"wrong","response_hash":"wrong","response":{}}')
    with pytest.raises(ValidationError, match="identity"):
        asyncio.run(client.complete({}, prompt="test", data_class="commissioned"))
