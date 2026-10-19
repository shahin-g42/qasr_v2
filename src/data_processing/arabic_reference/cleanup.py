"""Full-dataset scan and deterministic, candidate-supported editorial drafts."""

from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from .core import (
    format_signature,
    load_plan,
    protected_values,
    review_issues,
    sha256,
    write_json,
    write_jsonl,
)
from .editorial import STYLE_VERSION, ideal_style, rendering_audit

CLEANUP_VERSION = "agreement-rendering-v1"


def direct_proposal(row):
    """Reuse a shared rendering only when all wording agrees and no risk is flagged."""
    from .prompts import blinded

    candidates = row["candidates"]
    if len({c["name"] for c in candidates}) < 2 or review_issues(row):
        return None
    renderings = {ideal_style(c["text"]) for c in candidates}
    if len(renderings) != 1:
        return None
    rendering = next(iter(renderings))
    if not rendering or len(rendering) > 1800:
        return None
    if format_signature(ideal_style(row["text"])) != format_signature(rendering):
        return None
    if protected_values(row["text"]) != protected_values(rendering):
        return None
    audit = rendering_audit(row["text"], rendering)
    if not audit["same_words_after_explicit_style"] or audit["removed_marks"]:
        return None
    basis = next(
        (c for c in blinded(row) if not c["truncated"] and ideal_style(c["text"]) == rendering),
        None,
    )
    if basis is None:
        return None
    return {
        "base_candidate": basis["label"],
        "proposed_verified_text": basis["text"],
        "proposed_ideal_text": rendering,
        "changes": []
        if basis["text"] == rendering
        else [
            {
                "kind": "ideal_orthography"
                if "ﷺ" in rendering and "ﷺ" not in basis["text"]
                else "formatting",
                "before": basis["text"],
                "after": rendering,
                "reason": "Explicit NFC/whitespace/ﷺ rendering convention; preserve the shared candidate wording.",
            }
        ],
        "issue_notes": {},
        "uncertainties": [
            "Candidate agreement and retained marks do not establish the spoken words; audio accuracy remains unverified."
        ],
        "reason": "All source wording agrees after explicit rendering conventions; ASR candidates share this rendering and no original diacritics or protected values are lost. No contextual spelling replacement was inferred.",
    }


def initial_draft(row):
    direct = direct_proposal(row)
    return {
        "method": "agreement_rendering" if direct else "reference_style_baseline",
        "wording": direct["proposed_verified_text"] if direct else row["text"],
        "ideal": direct["proposed_ideal_text"] if direct else ideal_style(row["text"]),
        "verified": False,
        "requires_llm_review": True,
    }


def scan(plan_root, output):
    """Process every row, write useful direct drafts and an explicit contextual queue."""
    plan, rows = load_plan(plan_root)
    output = Path(output)
    identity = {
        "manifest_digest": plan["manifest_digest"],
        "cleanup_version": CLEANUP_VERSION,
        "style_version": STYLE_VERSION,
        "implementation": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if output.exists():
        previous = json.loads((output / "scan.json").read_text())
        if previous["identity"] != identity or any(
            sha256(output / name) != value for name, value in previous["files"].items()
        ):
            raise ValueError("Scan inputs/code changed; choose a new output directory")
        return previous
    categories, methods, flags = Counter(), Counter(), Counter()
    by_dataset = defaultdict(Counter)
    edits, edit_ids = Counter(), defaultdict(list)
    drafts, queue, direct = [], [], []
    for row in rows:
        texts = [row["text"], *(c["text"] for c in row["candidates"])]
        signatures = [format_signature(t) for t in texts]
        category = (
            "all_same_words"
            if len(set(signatures)) == 1
            else "asr_same_words"
            if len(set(signatures[1:])) == 1
            else "reference_matches_one_asr"
            if signatures[0] in signatures[1:]
            else "different_candidates"
        )
        categories[category] += 1
        by_dataset[row["dataset"]][category] += 1
        issues = review_issues(row)
        flags.update(issues.keys())
        proposal = direct_proposal(row)
        method = "agreement_rendering" if proposal else "reference_style_baseline"
        methods[method] += 1
        by_dataset[row["dataset"]][method] += 1
        wording = proposal["proposed_verified_text"] if proposal else row["text"]
        ideal = proposal["proposed_ideal_text"] if proposal else ideal_style(row["text"])
        record = {
            "id": row["id"],
            "dataset": row["dataset"],
            "binding": row["binding"],
            "audio_sha256": row["audio_sha256"],
            "text": row["text"],
            "asr_text": next((c["text"] for c in row["candidates"] if c["name"] == "qasr"), None),
            "gqasr_text": next(
                (c["text"] for c in row["candidates"] if c["name"] == "gqasr"), None
            ),
            "verified_text_draft": wording,
            "ideal_text_draft": ideal,
            "verified_text": None,
            "ideal_text": None,
            "audio_reviewed": False,
            "correction_method": method,
            "contextual_correction_pending": True,
            "agreement": category,
            "review_issues": list(issues),
            "reference_rendering_audit": rendering_audit(row["text"], ideal),
            "explicit_style_changed": ideal_style(row["text"]) != row["text"],
        }
        drafts.append(record)
        if proposal:
            direct.append({"id": row["id"], "binding": row["binding"], "proposal": proposal})
        queue.append(
            {
                "id": row["id"],
                "binding": row["binding"],
                "agreement": category,
                "review_issues": list(issues),
            }
        )
        # Corpus tendencies are diagnostic evidence, never a global replacement dictionary.
        if (
            category == "asr_same_words"
            and not issues
            and max(len(s.split()) for s in signatures) < 200
        ):
            a, b = signatures[0].split(), signatures[1].split()
            for tag, i, j, k, end in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
                if tag != "equal" and max(j - i, end - k) <= 4:
                    pair = (" ".join(a[i:j]), " ".join(b[k:end]))
                    edits[pair] += 1
                    if row["id"] not in edit_ids[pair] and len(edit_ids[pair]) < 3:
                        edit_ids[pair].append(row["id"])
    report = {
        "schema_version": 1,
        "identity": identity,
        "selected": len(rows),
        "selection_scope": plan.get("selection_scope", "pilot_plus_diagnostics"),
        "agreement_counts": dict(categories),
        "correction_methods": dict(methods),
        "direct_drafts": len(direct),
        "contextual_queue": len(queue),
        "contextual_wording_priority": methods["reference_style_baseline"],
        "original_to_draft_changed": sum(r["text"] != r["ideal_text_draft"] for r in drafts),
        "explicit_style_changed_rows": sum(r["explicit_style_changed"] for r in drafts),
        "salawat_symbol_rows": sum("ﷺ" in r["ideal_text_draft"] for r in drafts),
        "audio_verified": 0,
        "datasets": dict(by_dataset),
        "review_issue_counts": dict(flags),
        "common_candidate_edits": [
            {"before": a, "after": b, "occurrences": n, "examples": edit_ids[(a, b)]}
            for (a, b), n in edits.most_common(100)
        ],
        "notes": [
            "Every source row is retained and requires the subsequent LLM contextual pass, including agreement-rendering drafts. Initial drafts are not completed corrections.",
            "Direct drafts require all-candidate wording agreement, identical ASR rendering, no review-risk flags and preservation of existing marks and protected values.",
            "No blanket hamza, ta marbuta, alif maqsura or clitic replacement is applied. Corpus frequencies are not proof of the intended word.",
            "The user-requested ﷺ convention changes representation only; it never inserts a missing honorific.",
            "All drafts remain unverified. Model-assisted reference development must be disclosed in benchmark comparisons.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-scan-", dir=output.parent) as temporary:
        stage = Path(temporary) / "scan"
        stage.mkdir()
        write_jsonl(stage / "corrected-drafts.jsonl", drafts)
        write_jsonl(stage / "direct-proposals.jsonl", direct)
        write_jsonl(stage / "contextual-queue.jsonl", queue)
        report["files"] = {p.name: sha256(p) for p in stage.iterdir()}
        write_json(stage / "scan.json", report)
        stage.rename(output)
    return report
