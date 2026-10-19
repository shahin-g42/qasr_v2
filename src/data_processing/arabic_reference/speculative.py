"""Explicit speculative overlays; never promote them into listening-reviewed references."""

from __future__ import annotations

import json
from pathlib import Path

from .core import digest, read_jsonl, sha256
from .prompts import parse_proposal


def load_speculative_drafts(path, rows, records, identity):
    if path is None:
        return {}, None
    if identity is None:
        raise ValueError("Speculative drafts require their source proposals")
    path = Path(path)
    manifest = json.loads((path.parent / "manifest.json").read_text())
    run_hash = digest(identity)
    if (
        manifest["source_run_sha256"] != run_hash
        or manifest["drafts_sha256"] != sha256(path)
        or manifest["audio_listened"] is not False
        or manifest["verified"] != 0
    ):
        raise ValueError("Speculative manifest differs from its source or draft file")
    by_id = {r["id"]: r for r in rows}
    drafts = {}
    for draft in read_jsonl(path):
        key = draft["id"]
        stored = draft["speculative_draft_sha256"]
        if (
            key in drafts
            or key not in by_id
            or stored != digest({k: v for k, v in draft.items() if k != "speculative_draft_sha256"})
        ):
            raise ValueError("Changed, duplicate or unknown speculative draft")
        row, record = by_id[key], records.get(key, {})
        candidates = {c["name"]: c["text"] for c in row["candidates"]}
        if (
            record.get("status") != "unresolved"
            or draft["binding"] != row["binding"]
            or draft["source_run_sha256"] != run_hash
            or draft["source_record_sha256"] != record.get("record_sha256")
            or draft["text"] != row["text"]
            or draft["asr_text"] != candidates.get("qasr")
            or draft["gqasr_text"] != candidates.get("gqasr")
        ):
            raise ValueError(f"Speculative source mismatch: {key}")
        if (
            draft["proposal_status"] != "speculative_draft"
            or draft["correction_method"] != "assistant_speculative_adjudication"
            or draft["user_requested_speculation"] is not True
            or draft["audio_listened"] is not False
            or draft["audio_review_required"] is not True
            or draft["include_in_verified_scores"] is not False
            or draft["verified_text"] is not None
            or draft["ideal_text"] is not None
        ):
            raise ValueError("Speculative drafts must remain explicitly unverified")
        proposal = parse_proposal(json.dumps(draft["proposal"], ensure_ascii=False), "stop", row)
        if (
            not draft["ideal_text_draft"]
            or not draft["verified_text_draft"]
            or proposal["proposed_ideal_text"] != draft["ideal_text_draft"]
            or proposal["proposed_verified_text"] != draft["verified_text_draft"]
            or not draft["speculative_spans"]
            or not draft["judgment"]
            or draft["confidence"] not in {"very_low", "low", "medium", "high"}
        ):
            raise ValueError("Speculative draft text or uncertainty evidence is inconsistent")
        drafts[key] = {**draft, "proposal": proposal}
    if (
        len(drafts) != manifest["records"]
        or len(set(manifest["ids"])) != len(manifest["ids"])
        or set(drafts) != set(manifest["ids"])
    ):
        raise ValueError("Speculative manifest coverage mismatch")
    return drafts, {
        "method": "assistant_speculative_adjudication",
        "source_run_sha256": run_hash,
        "drafts_sha256": manifest["drafts_sha256"],
        "records": len(drafts),
        "ids": sorted(drafts),
        "audio_listened": False,
        "user_requested_speculation": True,
    }
