#!/usr/bin/env python3
"""Repair reviewed Malayalam spelling corruptions in JSONL `text` only.

Without --output, prints a dry-run change log. With --output, writes a NEW
manifest (never overwrites the input or an existing file). Change records go
to stdout, summary to stderr. Redirect stdout to retain a JSONL audit log.
No ASR/original transcript substitution, language detection, or general spell
checking is performed. Review these corpus-specific rules before wider use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

# Exact word repairs, applied after restoring the broken conjunct below.
WORD_REPAIRS = {
    "ഏററവും": "ഏറ്റവും",
    "പററി": "പറ്റി",
    "പററും": "പറ്റും",
    "പററിയ": "പറ്റിയ",
    "പററില്ല": "പറ്റില്ല",
    "പററുമോ": "പറ്റുമോ",
    "പററുന്ന": "പറ്റുന്ന",
    "പററൂ": "പറ്റൂ",
    "പററാത്ത": "പറ്റാത്ത",
    "മററേ": "മറ്റേ",
    "മററു": "മറ്റു",
    "ഒററ": "ഒറ്റ",
    "പിററേ": "പിറ്റേ",
    "വെളളം": "വെള്ളം",
    "വെളളവും": "വെള്ളവും",
    "വെളളത്തിൽ": "വെള്ളത്തിൽ",
    "ഉളള": "ഉള്ള",
    "ഉളളൂ": "ഉള്ളൂ",
    "ഉളളത്": "ഉള്ളത്",
    "വിവാദങ്ങളാണ": "വിവാദങ്ങളാണ്",
    "ആത്മീയവിഷയങ്ങളാണ": "ആത്മീയവിഷയങ്ങളാണ്",
    "ചെയ്യപെട്ടിരുനത്": "ചെയ്യപ്പെട്ടിരുന്നത്",
    "ഉണ്ടാവുകയുളളൂ": "ഉണ്ടാവുകയുള്ളൂ",
}
# Include Malayalam combining marks in boundaries (Python \w excludes them).
WORDS = re.compile(r"[\u0d00-\u0d7f]+")


def repair(text: str) -> str:
    text = text.replace("ങങ", "ങ്ങ")
    return WORDS.sub(lambda m: WORD_REPAIRS.get(m.group(), m.group()), text)


def repair_record(record: dict) -> str:
    """Additional virama restoration only when both comparison columns agree.

    No whole-sentence copying or arbitrary spelling inference. A candidate must
    be present as a complete word in BOTH org_text and asr_text, and differ
    solely by inserting viramas into the observed broken rr/ll conjuncts.
    """
    text = repair(record["text"])
    org = set(WORDS.findall(record.get("org_text") or ""))
    asr = set(WORDS.findall(record.get("asr_text") or ""))
    def restore(match):
        word = match.group()
        candidate = word.replace("ററ", "റ്റ").replace("ളള", "ള്ള")
        return candidate if candidate in org and candidate in asr else word
    return WORDS.sub(restore, text)


def process(source: Path, output: Path | None = None) -> dict:
    if output is not None and output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    rows = changed = 0
    temporary = None
    try:
        if output is not None:
            temporary = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output.parent,
                prefix=".ml-spelling-", suffix=".tmp", delete=False,
            )
        with source.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    if temporary:
                        temporary.write(line)
                    continue
                record = json.loads(line)
                if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                    raise ValueError(f"{source}:{number}: expected an object with string text")
                rows += 1
                before = record["text"]
                after = repair_record(record)
                if before != after:
                    changed += 1
                    print(json.dumps({"file": str(source), "line": number,
                                      "before": before, "after": after}, ensure_ascii=False))
                    record["text"] = after
                    line = json.dumps(record, ensure_ascii=False) + "\n"
                if temporary:
                    temporary.write(line)
        if temporary:
            temporary.close()
            # Publish only a complete file; fail if destination appeared meanwhile.
            os.link(temporary.name, output)
        return {"rows": rows, "changed_rows": changed, "dry_run": output is None,
                "output": str(output) if output else None}
    finally:
        if temporary:
            temporary.close()
            Path(temporary.name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, help="New corrected JSONL; default: dry run")
    args = parser.parse_args()
    print(json.dumps(process(args.input, args.output)), file=sys.stderr)


if __name__ == "__main__":
    main()
