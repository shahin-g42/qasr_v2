"""Bounded, resumable requests to the existing Yodas3 corrector endpoint."""

from __future__ import annotations

import fcntl
import json
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from data_processing.asr_clean.endpoints import JsonClient

from .core import (
    DATASETS,
    digest,
    format_signature,
    load_plan,
    review_issues,
    sha256,
    write_json,
    write_jsonl,
)
from .prompts import PROMPT_VERSION, REPAIR_INSTRUCTION, SYSTEM, messages, parse_proposal


@contextmanager
def exclusive(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"Another writer owns {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def client_for(url, timeout=180):
    url = url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    return JsonClient(url, timeout=timeout, retries=2)


def doctor(url, model, checkpoint=None, timeout=180):
    data = client_for(url, timeout).request("GET", "/models")
    matches = [entry for entry in data.get("data", []) if entry.get("id") == model]
    if len(matches) != 1:
        raise ValueError(
            f"Expected served model {model!r}; available: {[x.get('id') for x in data.get('data', [])]}"
        )
    found = matches[0]
    actual = found.get("root")
    # Routing uses the served alias. A checkpoint restriction is opt-in;
    # always retain the advertised root so resume can detect a model change.
    if (
        checkpoint
        and actual
        and "/" in actual
        and not actual.startswith("/")
        and actual != checkpoint
    ):
        raise ValueError(f"Server advertises {actual}, expected {checkpoint}")
    if checkpoint:
        identity_check = (
            "matched"
            if actual == checkpoint
            else "declaration_only; record serving revision before release"
        )
    else:
        identity_check = (
            "reported_by_server"
            if actual and actual != model
            else "alias_only; checkpoint unverified"
        )
    return {
        "served_model": model,
        "declared_checkpoint": checkpoint,
        "server_model": {k: found.get(k) for k in ("id", "root", "max_model_len")},
        "identity_check": identity_check,
        "audio_capability": "not_used; proposals are text-only",
    }


def record_path(root, identifier):
    return Path(root) / "records" / (identifier.replace(":", "-") + ".json")


def load_record(root, row, run_hash):
    path = record_path(root, row["id"])
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    stored_hash = record.pop("record_sha256")
    if (
        digest(record) != stored_hash
        or record["binding"] != row["binding"]
        or record["run_sha256"] != run_hash
    ):
        raise ValueError(f"Changed checkpoint record: {row['id']}")
    record["record_sha256"] = stored_hash
    return record


def select_smoke(plan, rows):
    chosen = {next(r["id"] for r in rows if r["dataset"] == ds) for ds in plan["input_counts"]}
    chosen.add(max(rows, key=lambda r: len(json.dumps(messages(r))))["id"])
    chosen.add(max(rows, key=lambda r: len(r["flags"]))["id"])
    controls = {}
    for ds in DATASETS:
        eligible = [
            r
            for r in rows
            if r["dataset"] == ds
            and "pilot" in r["cohorts"]
            and not review_issues(r)
            and len(
                {format_signature(t) for t in [r["text"], *(c["text"] for c in r["candidates"])]}
            )
            == 1
        ]
        if eligible:
            selected = min(
                eligible, key=lambda r: digest([plan["seed"], "wording-control-v1", r["id"]])
            )
            controls[ds] = selected["id"]
            chosen.add(selected["id"])
    return [r for r in rows if r["id"] in chosen], controls


def check_smoke(rows, records, controls):
    by_id = {r["id"]: r for r in rows}
    results = {}
    for record in records:
        if record["id"] not in controls.values():
            continue
        draft = (record.get("proposal") or {}).get("proposed_verified_text")
        results[record["id"]] = {
            "status": record["status"],
            "wording_preserved": isinstance(draft, str)
            and format_signature(draft) == format_signature(by_id[record["id"]]["text"]),
        }
    missing = [ds for ds in DATASETS if ds not in controls]
    return {
        "passed": not missing
        and not any(r["status"] == "failed" for r in records)
        and len(results) == len(DATASETS)
        and all(r["status"] == "proposed" and r["wording_preserved"] for r in results.values()),
        "missing_control_datasets": missing,
        "agreement_controls": results,
        "note": "Checks draft production and wording preservation, not audio accuracy or full-run throughput. Review the difficult-case proposals before scaling.",
    }


def combined_usage(attempts):
    result = Counter()
    for attempt in attempts:
        usage = attempt.get("usage")
        if isinstance(usage, dict):
            result.update({k: v for k, v in usage.items() if type(v) is int})
    return dict(result)


def propose(
    plan_root,
    output,
    *,
    url,
    model,
    checkpoint=None,
    revision="unrecorded",
    concurrency=16,
    max_tokens=2048,
    timeout=180,
    smoke=False,
    emit=print,
):
    if not 1 <= concurrency <= 256 or max_tokens < 128 or timeout <= 0:
        raise ValueError("Invalid request settings")
    plan, all_rows = load_plan(plan_root)
    rows = all_rows
    controls = {}
    if smoke:
        rows, controls = select_smoke(plan, rows)
    output = Path(output)
    with exclusive(output):
        server = doctor(url, model, checkpoint, timeout)
        identity = {
            "schema_version": 1,
            "manifest_digest": plan["manifest_digest"],
            "policy_sha256": plan["policy_sha256"],
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": digest(SYSTEM),
            "server": server,
            "checkpoint_revision": revision,
            "url": url.rstrip("/"),
            "generation": {"temperature": 0, "max_tokens": max_tokens, "enable_thinking": False},
            "code": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
            "ids": [r["id"] for r in rows],
            "smoke": smoke,
            "smoke_controls": controls,
            "validation_retries": 1,
            "repair_prompt_sha256": digest(REPAIR_INSTRUCTION),
        }
        run_hash = digest(identity)
        metadata_path = output / "run.json"
        if metadata_path.exists() and json.loads(metadata_path.read_text()) != identity:
            raise ValueError(
                "Run inputs, code, model, or policy changed; choose a new run directory"
            )
        write_json(metadata_path, identity)
        client = client_for(url, timeout)
        pending_rows = []
        for row in rows:
            old = load_record(output, row, run_hash)
            if not old or old["status"] == "failed":
                pending_rows.append(row)
        completed = len(rows) - len(pending_rows)
        start = time.monotonic()
        emit(
            json.dumps(
                {
                    "event": "reference_start",
                    "output": str(output.resolve()),
                    "selected": len(rows),
                    "resumed": completed,
                    "concurrency": concurrency,
                    "smoke_controls": controls,
                }
            )
        )

        def one(row):
            record = {
                "id": row["id"],
                "binding": row["binding"],
                "run_sha256": run_hash,
                "review_status": "proposed",
                "verified_text": None,
                "ideal_text": None,
                "audio_review_required": True,
                "audio_supplied_to_corrector": False,
                "attempts": [],
            }
            conversation = messages(row)
            try:
                for attempt_number in range(2):
                    attempt = {"number": attempt_number + 1}
                    record["attempts"].append(attempt)
                    try:
                        response = client.request(
                            "POST",
                            "/chat/completions",
                            {
                                "model": model,
                                "messages": conversation,
                                "temperature": 0,
                                "max_tokens": max_tokens,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                        )
                        choice = response["choices"][0]
                        details = {
                            "raw_response": choice["message"].get("content") or "",
                            "finish_reason": choice.get("finish_reason"),
                            "usage": response.get("usage", {}),
                            "response_model": response.get("model"),
                        }
                        attempt.update(details)
                        record.update(details)
                    except Exception as exc:
                        attempt["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    try:
                        record["proposal"] = parse_proposal(
                            record["raw_response"], record["finish_reason"], row
                        )
                    except ValueError as exc:
                        attempt["validation_error"] = str(exc)
                        if attempt_number == 0 and record["finish_reason"] == "stop":
                            feedback = REPAIR_INSTRUCTION + str(exc)
                            attempt["repair_feedback"] = feedback
                            conversation = [
                                *conversation,
                                {"role": "assistant", "content": record["raw_response"]},
                                {"role": "user", "content": feedback},
                            ]
                            continue
                        raise
                    break
                record["status"] = (
                    "proposed"
                    if record["proposal"]["proposed_verified_text"] is not None
                    else "unresolved"
                )
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            record["usage"] = combined_usage(record["attempts"])
            record["record_sha256"] = digest(record)
            write_json(record_path(output, row["id"]), record)
            return record["status"]

        iterator = iter(pending_rows)
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            inflight = {}

            def fill():
                while len(inflight) < concurrency:
                    row = next(iterator, None)
                    if row is None:
                        break
                    inflight[pool.submit(one, row)] = row["id"]

            fill()
            last = 0.0
            while inflight:
                done, _ = wait(inflight, timeout=5, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()  # Surface disk failures; never claim an unwritten record.
                    del inflight[future]
                    completed += 1
                fill()
                now = time.monotonic()
                if now - last >= 5 or not inflight:
                    emit(
                        json.dumps(
                            {
                                "event": "reference_progress",
                                "completed": completed,
                                "total": len(rows),
                                "elapsed_seconds": round(now - start, 1),
                            }
                        )
                    )
                    last = now
        records = [load_record(output, row, run_hash) for row in rows]
        counts = Counter(r["status"] for r in records)
        summary = {
            "run_sha256": run_hash,
            "output": str(output.resolve()),
            "selected": len(rows),
            "counts": dict(counts),
            "resumed": len(rows) - len(pending_rows),
            "verified": 0,
            "audio_review_required": True,
            "smoke": smoke,
            "usage": dict(
                sum(
                    (
                        Counter({k: v for k, v in r.get("usage", {}).items() if type(v) is int})
                        for r in records
                    ),
                    Counter(),
                )
            ),
            "wall_seconds_this_invocation": round(time.monotonic() - start, 3),
        }
        summary["validation_retries"] = sum(len(r.get("attempts", [])) > 1 for r in records)
        summary["failure_counts"] = dict(
            Counter(r["error"] for r in records if r["status"] == "failed")
        )
        summary["failed_examples"] = [
            {k: r.get(k) for k in ("id", "error", "finish_reason")}
            for r in records
            if r["status"] == "failed"
        ][:10]
        if smoke:
            summary["smoke_checks"] = check_smoke(rows, records, controls)
        write_jsonl(output / "proposals.jsonl", records)
        write_json(output / "summary.json", summary)
        return summary


def load_proposals(root, plan, rows):
    if root is None:
        return {}
    identity = json.loads((Path(root) / "run.json").read_text())
    if identity["manifest_digest"] != plan["manifest_digest"]:
        raise ValueError("Proposals belong to a different pilot")
    result = {}
    for row in rows:
        record = load_record(root, row, digest(identity))
        if record and record["status"] != "failed":
            result[row["id"]] = record
    return result
