"""Failed-only recovery: bounded explanations and immutable source provenance."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .core import digest

RECOVERY_VERSION = "arabic-reference-recovery-v1"
RECOVERY_INSTRUCTION = """Recovery output requirements (the linguistic policy still applies):
Return only the final JSON decision, never a debate, self-correction monologue, or repeated
alternatives. Preserve full transcript text; shorten explanations, never the utterance.
Each reason and issue note must be one short sentence, at most 360 characters. Use 1 to 8
listening uncertainties, each at most 480 characters. All explanatory text combined must
fit 6000 characters. List each actual edit once, at most 128 edits; group punctuation-only
changes when useful. Empty changes is correct when no edit is needed. Do not repeatedly
reconsider marks: preserve valid supplied shadda, including sun-letter assimilation and
root gemination; an unmarked candidate is not a reason to strip them. If an interpretation
remains uncertain, preserve the candidate form and state that uncertainty once. Keep all
contextual hamza, ya/maqsura, ta marbuta/pronoun, dialect, value and salawat rules. Use valid
JSON escaping inside strings. Do not invent speech or set null merely to avoid an error.
"""


def validate_recovery_metadata(value):
    """Reject runaway explanations without trimming or accepting partial transcripts."""
    reasons = [value["reason"], *value["issue_notes"].values()]
    reasons.extend(change["reason"] for change in value["changes"])
    uncertainties = value["uncertainties"]
    if (
        len(value["changes"]) > 128
        or len(uncertainties) > 8
        or any(len(s) > 360 for s in reasons)
        or any(len(s) > 480 for s in uncertainties)
        or sum(map(len, reasons + uncertainties)) > 6000
    ):
        raise ValueError("Recovery explanations exceed limits; give concise final decisions")
    edits = [(c["kind"], c["before"], c["after"]) for c in value["changes"]]
    if any(n > 3 for n in Counter(edits).values()):
        raise ValueError("Recovery repeats the same edit; consolidate its explanation")


def source_index(source, plan, all_rows):
    from .runtime import load_record

    identity = json.loads((Path(source) / "run.json").read_text())
    if identity["manifest_digest"] != plan["manifest_digest"]:
        raise ValueError("Recovery source belongs to a different plan")
    ids = identity["ids"]
    wanted = set(ids)
    rows = [row for row in all_rows if row["id"] in wanted]
    if len(ids) != len(wanted) or len(rows) != len(ids):
        raise ValueError("Recovery source has duplicate or unknown IDs")
    run_hash = digest(identity)
    index = {}
    for row in rows:
        record = load_record(source, row, run_hash, inherited=identity.get("inherited_records", {}))
        if record is None:
            raise ValueError(f"Recovery source is incomplete: {row['id']}")
        if record["status"] not in {"proposed", "unresolved", "failed"}:
            raise ValueError("Unknown source proposal status")
        if record.get("verified_text") is not None or record.get("ideal_text") is not None:
            raise ValueError("A corrector source cannot supply final reference columns")
        index[row["id"]] = {k: record[k] for k in ("run_sha256", "record_sha256", "status")}
    return identity, rows, index
