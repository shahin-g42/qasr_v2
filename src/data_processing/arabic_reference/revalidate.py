"""Recover complete saved replies offline; never reconstruct truncated JSON or speech."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from .core import digest, load_plan, read_jsonl, sha256, write_json, write_jsonl
from .editorial import apply_ideal_style
from .prompts import parse_proposal, proposal_checks
from .recovery import RECOVERY_VERSION, recovery_metadata_warnings
from .runtime import exclusive, record_path

REVALIDATION_VERSION = "arabic-saved-replies-v1"


def checked_aggregate(path, rows, identity):
    """Validate the complete portable aggregate, including inherited record origins."""
    by_id = {r["id"]: r for r in rows}
    expected_ids = set(identity["ids"])
    if len(expected_ids) != len(identity["ids"]) or expected_ids - by_id.keys():
        raise ValueError("Source run has duplicate or unknown IDs")
    run_hash = digest(identity)
    inherited = identity.get("inherited_records", {})
    seen = set()
    for record in read_jsonl(path):
        key = record["id"]
        stored = record.pop("record_sha256")
        if (
            key in seen
            or key not in expected_ids
            or digest(record) != stored
            or record["binding"] != by_id[key]["binding"]
        ):
            raise ValueError(f"Changed or duplicate source record: {key}")
        record["record_sha256"] = stored
        origin = {k: record[k] for k in ("run_sha256", "record_sha256", "status")}
        if key in inherited:
            if origin != inherited[key] or record["status"] not in {"proposed", "unresolved"}:
                raise ValueError(f"Changed inherited source record: {key}")
        elif record["run_sha256"] != run_hash:
            raise ValueError(f"Unknown source run: {key}")
        if record["status"] not in {"proposed", "unresolved", "failed"}:
            raise ValueError("Unknown source status")
        if record.get("verified_text") is not None or record.get("ideal_text") is not None:
            raise ValueError("A corrector source cannot supply final reference columns")
        seen.add(key)
        yield record
    if seen != expected_ids:
        raise ValueError("Source aggregate is incomplete")


def replay_failure(original, row, run_hash):
    """Choose the latest complete usable reply; retain every earlier failure verbatim."""
    record = dict(original)
    record.pop("record_sha256")
    record["run_sha256"] = run_hash
    audit = {
        "source_run_sha256": original["run_sha256"],
        "source_record_sha256": original["record_sha256"],
        "llm_requests": 0,
        "selected_attempt": None,
        "attempt_results": [],
    }
    record["saved_response_revalidation"] = audit
    for attempt in reversed(original.get("attempts", [])):
        try:
            adjustments = []
            proposal = parse_proposal(
                attempt.get("raw_response", ""),
                attempt.get("finish_reason"),
                row,
                adjustments=adjustments,
            )
            adjustments.extend(recovery_metadata_warnings(proposal))
            apply_ideal_style(proposal, adjustments)
            checks = proposal_checks(proposal, row, adjustments)
        except (ValueError, TypeError, KeyError) as exc:
            audit["attempt_results"].append(
                {
                    "number": attempt.get("number"),
                    "accepted": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        audit.update(
            {
                "selected_attempt": attempt["number"],
                "raw_response_sha256": digest(attempt["raw_response"]),
            }
        )
        audit["attempt_results"].append({"number": attempt["number"], "accepted": True})
        record.update(
            {
                "proposal": proposal,
                "proposal_checks": checks,
                "validation_adjustments": adjustments,
                "raw_response": attempt["raw_response"],
                "finish_reason": attempt["finish_reason"],
                "response_model": attempt.get("response_model"),
                "status": "proposed"
                if proposal["proposed_verified_text"] is not None
                else "unresolved",
            }
        )
        record.pop("error", None)
        break
    record["record_sha256"] = digest(record)
    return record


def revalidate_saved(plan_root, source, output, *, emit=print):
    """Publish a complete new local snapshot without a model endpoint or audio claims."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Revalidation output must be a separate directory, outside its source")
    plan, all_rows = load_plan(plan_root)
    source_identity = json.loads((source / "run.json").read_text())
    if source_identity["manifest_digest"] != plan["manifest_digest"]:
        raise ValueError("Revalidation source belongs to a different plan")
    aggregate = source / "proposals.jsonl"
    aggregate_hash = sha256(aggregate)
    index = {
        r["id"]: {k: r[k] for k in ("run_sha256", "record_sha256", "status")}
        for r in checked_aggregate(aggregate, all_rows, source_identity)
    }
    inherited = {k: v for k, v in index.items() if v["status"] != "failed"}
    rows = {r["id"]: r for r in all_rows}
    identity = {
        "schema_version": 1,
        "manifest_digest": plan["manifest_digest"],
        "policy_sha256": plan["policy_sha256"],
        "initial_cleanup": source_identity.get("initial_cleanup"),
        "editorial_style": source_identity.get("editorial_style"),
        "ids": source_identity["ids"],
        "revalidation_version": REVALIDATION_VERSION,
        "processing_mode": "saved_response_revalidation",
        "llm_requests": 0,
        "recovery_version": RECOVERY_VERSION,
        "recovery_prompt_sha256": source_identity.get("recovery_prompt_sha256"),
        "recovery_temperatures": [],
        "generation": None,
        "inherited_records": inherited,
        "retry_ids": [k for k, v in index.items() if v["status"] == "failed"],
        "repair_source": {
            "run_sha256": digest(source_identity),
            "identity": source_identity,
            "aggregate_sha256": aggregate_hash,
            "failed_records": {k: v for k, v in index.items() if v["status"] == "failed"},
        },
        "code": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    run_hash = digest(identity)
    summary = {
        "run_sha256": run_hash,
        "output": str(output),
        "selected": len(index),
        "llm_requests": 0,
        "verified": 0,
        "audio_review_required": True,
        "carried_forward": len(inherited),
        "targeted_failures": len(index) - len(inherited),
    }
    counts, new_counts, flags, usage = Counter(), Counter(), Counter(), Counter()
    recovered, failures = [], []
    flagged = metadata_warnings = completed = 0
    start = time.monotonic()
    with exclusive(output):
        metadata_path = output / "run.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != identity:
            raise ValueError("Run inputs or code changed; choose a new revalidation directory")
        write_json(metadata_path, identity)

        def save(path, record):
            if path.exists():
                if json.loads(path.read_text()) != record:
                    raise ValueError(f"Changed output record: {record['id']}")
            else:
                write_json(path, record)

        def process():
            nonlocal flagged, metadata_warnings, completed
            for original in checked_aggregate(aggregate, all_rows, source_identity):
                key = original["id"]
                record = original
                if original["status"] == "failed":
                    save(record_path(output / "source-failures", key), original)
                    record = replay_failure(original, rows[key], run_hash)
                    new_counts[record["status"]] += 1
                    if record["status"] != "failed":
                        recovered.append(
                            {
                                "id": key,
                                "selected_attempt": record["saved_response_revalidation"][
                                    "selected_attempt"
                                ],
                                "editorial_flags": record["proposal_checks"]["editorial_flags"],
                            }
                        )
                    else:
                        failures.append({"id": key, "error": record.get("error")})
                save(record_path(output, key), record)
                counts[record["status"]] += 1
                checks = record.get("proposal_checks", {})
                flags.update(checks.get("editorial_flags", []))
                flagged += bool(checks.get("editorial_flags"))
                metadata_warnings += bool(checks.get("metadata_warnings"))
                usage.update({k: v for k, v in record.get("usage", {}).items() if type(v) is int})
                completed += 1
                if completed % 2000 == 0 or completed == len(index):
                    emit(
                        json.dumps(
                            {
                                "event": "reference_revalidation_progress",
                                "completed": completed,
                                "total": len(index),
                            }
                        )
                    )
                yield record

        write_jsonl(output / "proposals.jsonl", process())
        if sha256(aggregate) != aggregate_hash:
            raise ValueError("Source aggregate changed during revalidation")
        summary.update(
            {
                "counts": dict(counts),
                "revalidation_counts": dict(new_counts),
                "editorial_flag_records": flagged,
                "editorial_flag_counts": dict(flags),
                "metadata_warning_records": metadata_warnings,
                "usage": dict(usage),
                "new_usage": {},
                "recovered": recovered,
                "failed_examples": failures,
                "wall_seconds_this_invocation": round(time.monotonic() - start, 3),
                "note": "Complete saved replies retained as unverified drafts; prose-budget overruns are review warnings. Historical attempts/usage are preserved. No LLM inference or audio verification occurred.",
            }
        )
        write_json(output / "summary.json", summary)
    return summary
