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

from .core import digest, load_plan, sha256, write_json, write_jsonl
from .prompts import PROMPT_VERSION, SYSTEM, messages, parse_proposal


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


def doctor(url, model, checkpoint, timeout=180):
    data = client_for(url, timeout).request("GET", "/models")
    matches = [entry for entry in data.get("data", []) if entry.get("id") == model]
    if len(matches) != 1:
        raise ValueError(
            f"Expected served model {model!r}; available: {[x.get('id') for x in data.get('data', [])]}"
        )
    found = matches[0]
    actual = found.get("root")
    # Absolute cache paths cannot be compared to Hub names. Preserve both for audit.
    if actual and "/" in actual and not actual.startswith("/") and actual != checkpoint:
        raise ValueError(f"Server advertises {actual}, expected {checkpoint}")
    return {
        "served_model": model,
        "declared_checkpoint": checkpoint,
        "server_model": {k: found.get(k) for k in ("id", "root", "max_model_len")},
        "identity_check": "matched"
        if actual == checkpoint
        else "declaration_only; record serving revision before release",
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


def propose(
    plan_root,
    output,
    *,
    url,
    model,
    checkpoint,
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
    if smoke:
        # Cover all datasets, plus the longest prompt and the most heavily flagged case.
        chosen = {next(r["id"] for r in rows if r["dataset"] == ds) for ds in plan["input_counts"]}
        chosen.add(max(rows, key=lambda r: len(json.dumps(messages(r))))["id"])
        chosen.add(max(rows, key=lambda r: len(r["flags"]))["id"])
        rows = [r for r in rows if r["id"] in chosen]
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
            }
            try:
                response = client.request(
                    "POST",
                    "/chat/completions",
                    {
                        "model": model,
                        "messages": messages(row),
                        "temperature": 0,
                        "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                choice = response["choices"][0]
                record.update(
                    {
                        "raw_response": choice["message"].get("content") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "usage": response.get("usage", {}),
                        "response_model": response.get("model"),
                    }
                )
                record["proposal"] = parse_proposal(
                    record["raw_response"], record["finish_reason"], row
                )
                record["status"] = (
                    "proposed"
                    if record["proposal"]["proposed_verified_text"] is not None
                    else "unresolved"
                )
            except Exception as exc:
                record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
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
