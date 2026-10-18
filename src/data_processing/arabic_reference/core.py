"""Immutable pilot selection and provenance. No model or audio dependencies."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATASETS = ("sada", "commonvoice", "masc_clean", "masc_noisy", "mgb2", "casablanca")
POLICY = {
    "version": "arabic-reference-pilot-v3",
    "purpose": "evaluation_only",
    "audio_review_required": True,
    "evidence": "Audio decides spoken content. References, model outputs, language headers and dialect metadata can all be wrong. Fluency, model agreement and lower WER are not proof.",
    "content": "Preserve dialect, code switching, negation, names, values, hesitations, audible repetition and incomplete speech. Never translate, paraphrase, repair grammar or complete quotations/scripture from memory.",
    "orthography": "Do not globally equate ه/ة, ى/ي, أ/إ/ا or dialect variants. Distinguish spelling refinement from different spoken wording. Declared additions of hamza to bare alif may refine ideal_text after review. Other spelling, clitic-spacing and script decisions require lexical review; never apply blanket substitutions.",
    "foreign_speech": "Arabic-letter phonetic English and Latin-script English are competing hypotheses. Use listening to identify the spoken language and words; script, spelling and hyphenation are editorial decisions, not sounds. Use conventional spelling for confidently identified foreign words, retain Arabic for established Arabic loanwords, and leave uncertain names/spelling unresolved. Neither script is automatically better.",
    "number_policy": "Preserve number words and digits; do not convert between them. Preserve numeric signs, decimal/thousands separators, units, dates, times, email addresses and URLs. Any content correction requires audio evidence.",
    "formatting": "ideal_text is the polished, faithful rendering of verified wording. Allow punctuation, whitespace layout, case, conservative Arabic vowel marks, and explicitly documented/reviewed bare-alif to hamzated-alif spelling refinements. Preserve word boundaries, letters, numbers and punctuation inside numbers/identifiers. Do not split/join words or change apostrophes in foreign words as formatting.",
    "diacritics": "Add only justified Arabic vowel marks; leave ambiguous vowels unmarked. Do not invent case endings, standardize dialect grammar or remove meaningful foreign accents/Arabic hamza or maddah.",
    "unclear": "غيرواضح/غير واضح may be an annotation marker or actual spoken words. Listen to distinguish them. Do not fill gaps from context, guess an email/name, or delete uncertain speech to improve WER. Any span remaining unresolved after listening makes the whole clip unresolved in this pilot; no partial-reference scoring.",
    "repetition": "Generation caps, omitted middles and extreme repetition are warnings, not evidence of speech. Do not copy loops, collapse genuine repeats, infer missing words, or treat shorter output as automatically correct. Audio review must establish the complete words.",
    "duplicates": "Identical prepared-audio hashes require consistent verified_text and ideal_text across reviewed rows; conflicting transcripts need adjudication. Original rows remain immutable. Duplicate diagnostics are separate from the sampled pilot.",
    "verification": "Text-only corrector output is always proposed, never verified. Drafts may retain a complete candidate with uncertainties; they are not final references. Every risk flag needs an explicit proposal note and a human resolution note. Ideal spelling refinements require their own explanation. A human listening attestation is not independent verification.",
    "fairness": "Hide model identities, listen before revealing candidates, and never optimize references to favor any system. Score original and reviewed references on the same reviewed subset; retain unresolved coverage and keep diagnostics separate.",
    "publication": "Internal pilot; source rights, independent review and reference freeze are required before public release. Evaluation audio and corrected references must not enter training; training cleanup and fresh-holdout evaluation are separate stages.",
}

UNCLEAR_RE = re.compile(r"غير\s*واضح", re.IGNORECASE)
MARKED_UNCLEAR_RE = re.compile(
    r"[\[<({]\s*(?:unclear|inaudible|unintelligible|غير\s*(?:واضح|مفهوم|مسموع))\s*[\]>)}]",
    re.IGNORECASE,
)
ISSUE_DESCRIPTIONS = {
    "unclear_span": "Possible annotation marker or unclear span: distinguish spoken words from a placeholder; do not guess missing speech.",
    "foreign_script": "Possible foreign speech or script disagreement: identify spoken language/words from audio, then review script and spelling as editorial choices; script alone is not evidence.",
    "repetition_or_truncation": "A candidate is capped, clipped or contains long repetition: establish the complete utterance and which repeats are audible.",
    "duplicate_audio": "Identical prepared audio occurs in multiple rows, possibly with different references: resolve from audio and use consistent reviewed texts.",
}


def suspected_repetition(text):
    tokens = text.split()
    for width in range(1, 5):
        required = max(8, math.ceil(24 / width))
        run = 0
        for i in range(width, len(tokens)):
            run = run + 1 if tokens[i] == tokens[i - width] else 0
            if run >= (required - 1) * width:
                return True
    return False


def review_issues(row):
    texts = [row["text"], *(c["text"] for c in row["candidates"])]
    issues = set()
    if any(UNCLEAR_RE.search(t) or MARKED_UNCLEAR_RE.search(t) for t in texts):
        issues.add("unclear_span")
    if any(re.search(r"[A-Za-z]", t) for t in texts):
        issues.add("foreign_script")
    if any(len(t) > 1800 or suspected_repetition(t) for t in texts) or any(
        c["generation_limit_hit"] for c in row["candidates"]
    ):
        issues.add("repetition_or_truncation")
    if "duplicate_audio" in row["flags"]:
        issues.add("duplicate_audio")
    return {key: ISSUE_DESCRIPTIONS[key] for key in sorted(issues)}


def validate_issue_notes(notes, row):
    if not isinstance(notes, dict) or set(notes) != set(review_issues(row)):
        raise ValueError("Every review issue requires its own note")
    if not all(isinstance(note, str) and note.strip() for note in notes.values()):
        raise ValueError("Review issue notes must be nonempty")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            yield row


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    atomic_text(
        path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows)
    )


# Protect punctuation carrying lexical meaning before treating sentence punctuation as layout.
PROTECTED_TOKEN_RE = re.compile(
    r"(?P<identifier>(?:https?://|www\.)[^\s<>]+"
    r"|[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)"
    r"|(?P<number>(?<!\w)(?:[+\-−±]\s*)?(?:\d+(?:[.,٫٬:/\-]\d+)*|[.٫]\d+)(?:\s*[%‰٪])?)"
    r"|\b\w+(?:[._/@:+#&='’\-]\w+)+\b",
    re.IGNORECASE,
)
ARABIC_VOWEL_MARKS = frozenset(chr(i) for i in range(0x064B, 0x0653)) | {"\u0670"}
LAYOUT_PUNCTUATION = frozenset(".,،;؛:!?؟…“”«»„\"'()[]{}—–-")


def format_signature(text):
    """Preserve word boundaries and meaningful punctuation; ignore only permitted formatting."""
    text = "".join(
        c
        for c in unicodedata.normalize("NFC", text)
        if c not in ARABIC_VOWEL_MARKS and c != "\u0640"
    )
    tokens = []

    def layout(part):
        tokens.extend(
            "".join(" " if c in LAYOUT_PUNCTUATION else c for c in part).casefold().split()
        )

    end = 0
    for match in PROTECTED_TOKEN_RE.finditer(text):
        layout(text[end : match.start()])
        token = match.group().rstrip(".,،;؛!?؟")
        tokens.append(re.sub(r"\s+", "", token) if match.lastgroup else token.casefold())
        end = match.start() + len(token)
    layout(text[end:])
    return " ".join(tokens)


def ideal_spelling_pairs(before, after):
    """Narrow pilot spelling allowance; meaning still needs explicit human review."""
    left, right = format_signature(before), format_signature(after)
    if len(left) != len(right) or any(
        a != b and not (a == "ا" and b in "أإ") for a, b in zip(left, right, strict=True)
    ):
        raise ValueError(
            "Ideal spelling edits may only add hamza to bare alif; other word changes require lexical review"
        )
    return {(a, b) for a, b in zip(left.split(), right.split(), strict=True) if a != b}


def validate_ideal_pair(before, after, spelling_reason=""):
    if not format_signature(before):
        raise ValueError("Ideal text requires spoken words")
    if format_signature(before) == format_signature(after):
        return set()
    pairs = ideal_spelling_pairs(before, after)
    if not isinstance(spelling_reason, str) or not spelling_reason.strip():
        raise ValueError("Ideal spelling changes require an explicit spelling explanation")
    return pairs


def binding(row):
    return digest({k: v for k, v in row.items() if k != "binding"})


def validate_source(row):
    ds = row.get("dataset")
    if ds not in DATASETS or not re.fullmatch(re.escape(str(ds)) + r":\d{6}", row.get("id", "")):
        raise ValueError(f"Invalid benchmark ID: {row.get('id')}")
    if type(row.get("ordinal")) is not int or int(row["id"].split(":")[1]) != row["ordinal"]:
        raise ValueError(f"Ordinal mismatch: {row['id']}")
    for key in ("text", "pred_text", "audio", "audio_sha256", "preparation_id"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"{row['id']}: missing {key}")
    if not re.fullmatch(r"[0-9a-f]{64}", row["audio_sha256"]):
        raise ValueError("Invalid audio SHA256")
    duration = row.get("actual_duration")
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"{row['id']}: invalid actual_duration")
    if row.get("mode") != "arabic":
        raise ValueError("This pilot accepts forced-Arabic predictions only")


def prepare(predictions, output, per_dataset=100, seed=20261019):
    if per_dataset < 1 or not predictions:
        raise ValueError("Positive sample size and prediction inputs required")
    output = Path(output)
    pools, source_info = [], []
    names = set()
    for name, path in predictions:
        if not name or name in names:
            raise ValueError("Candidate names must be nonempty and unique")
        names.add(name)
        pool = {}
        before = sha256(path)
        for row in read_jsonl(path):
            validate_source(row)
            if row["id"] in pool:
                raise ValueError(f"Duplicate ID: {row['id']}")
            pool[row["id"]] = row
        if before != sha256(path):
            raise ValueError(f"Input changed while reading: {path}")
        pools.append(pool)
        run_path = Path(path).parent / "run.json"
        source_info.append(
            {
                "name": name,
                "predictions_sha256": before,
                "run": json.loads(run_path.read_text()) if run_path.exists() else None,
            }
        )
    base = pools[0]
    for pool in pools[1:]:
        if set(pool) != set(base):
            raise ValueError("Candidate ID coverage differs")
        for identifier, row in base.items():
            for key in ("text", "audio_sha256", "preparation_id", "actual_duration", "dataset"):
                if pool[identifier][key] != row[key]:
                    raise ValueError(f"{identifier}: candidate {key} mismatch")
    counts = Counter(row["dataset"] for row in base.values())
    if set(counts) != set(DATASETS):
        raise ValueError("Pilot requires all six datasets")
    hashes = defaultdict(list)
    for row in base.values():
        hashes[row["audio_sha256"]].append(row["id"])
    flags = defaultdict(set)
    for row in base.values():
        identifier = row["id"]
        if re.search(r"غير\s*واضح", row["text"]):
            flags[identifier].add("unclear_reference_candidate")
        if any(pool[identifier].get("generation_limit_hit") is True for pool in pools):
            flags[identifier].add("generation_limit")
        if any(re.search(r"[A-Za-z]", pool[identifier]["pred_text"]) for pool in pools):
            flags[identifier].add("latin_prediction_candidate")
    for identifiers in hashes.values():
        if len(identifiers) > 1:
            conflicting = len({format_signature(base[i]["text"]) for i in identifiers}) > 1
            for identifier in identifiers:
                flags[identifier].add("duplicate_audio")
                if conflicting:
                    flags[identifier].add("conflicting_duplicate_reference")
    pilot = set()
    for ds in DATASETS:
        # Sampling ignores all predictions and scores. Exact duplicates occupy one slot.
        representatives = {}
        for row in sorted(base.values(), key=lambda r: r["id"]):
            if row["dataset"] == ds:
                representatives.setdefault(row["audio_sha256"], row["id"])
        ranked = sorted(representatives, key=lambda h: digest([seed, ds, h]))
        if len(ranked) < per_dataset:
            raise ValueError(f"{ds}: fewer unique audio clips than requested")
        pilot.update(representatives[h] for h in ranked[:per_dataset])
    selection = pilot | set(flags)
    rows = []
    for identifier in sorted(selection):
        raw = base[identifier]
        row = {
            k: raw[k]
            for k in (
                "id",
                "dataset",
                "ordinal",
                "text",
                "audio",
                "audio_sha256",
                "preparation_id",
                "actual_duration",
            )
        }
        row.update(
            {
                "source": raw.get("source"),
                "metadata": raw.get("metadata", {}),
                "relative_audio": raw.get("relative_audio"),
                "split": "test",
                "purpose": "evaluation_only",
                "verified_text": None,
                "ideal_text": None,
                "review_status": "pending",
                "policy_sha256": digest(POLICY),
                "cohorts": (["pilot"] if identifier in pilot else [])
                + (["diagnostic"] if identifier in flags else []),
                "flags": sorted(flags.get(identifier, [])),
                "candidates": [
                    {
                        "name": name,
                        "text": pool[identifier]["pred_text"],
                        "generation_limit_hit": pool[identifier].get("generation_limit_hit", False),
                        "generated_tokens": pool[identifier].get("generated_tokens"),
                    }
                    for (name, _), pool in zip(predictions, pools, strict=True)
                ],
            }
        )
        row["binding"] = binding(row)
        rows.append(row)
    plan = {
        "schema_version": 1,
        "policy": POLICY,
        "policy_sha256": digest(POLICY),
        "seed": seed,
        "per_dataset": per_dataset,
        "sources": source_info,
        "input_counts": dict(counts),
        "pilot_clips": len(pilot),
        "selected_clips": len(rows),
        "diagnostic_clips": len(flags),
        "cohort_overlap": len(pilot & set(flags)),
        "flag_counts": dict(Counter(f for row in rows for f in row["flags"])),
        "manifest_digest": digest(rows),
        "sampling": "100 per dataset by default; uniform hash ranking of unique prepared audio. Diagnostics are a separate, overlapping, outcome-selected cohort. Neither is a full-suite score.",
    }
    if (output / "plan.json").exists():
        old, previous = load_plan(output)
        if old != plan or previous != rows:
            raise ValueError("Existing pilot differs; choose a new output directory")
        return plan
    write_jsonl(output / "manifest.jsonl", rows)
    write_json(output / "plan.json", plan)
    return plan


def load_plan(root):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    rows = list(read_jsonl(root / "manifest.jsonl"))
    if plan["policy_sha256"] != digest(POLICY) or plan["policy"] != POLICY:
        raise ValueError("Policy changed; use the code matching this plan")
    if plan["manifest_digest"] != digest(rows) or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Pilot manifest changed or contains duplicate IDs")
    if any(r["binding"] != binding(r) for r in rows):
        raise ValueError("Row binding mismatch")
    return plan, rows
