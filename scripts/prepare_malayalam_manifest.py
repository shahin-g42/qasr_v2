#!/usr/bin/env python3
"""Audit every row and write conservative text repairs to a new output directory.

Outputs a same-basename manifest, changes.jsonl, review.jsonl and report.json.
Review flags are heuristics, NOT proven errors. No records are dropped. Audio
is not inspected. Source and all fields except text are preserved.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from fix_malayalam_manifest_spelling import repair_record

CONSONANTS = 'കഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരലവശഷസഹളഴറ'
BARE_RUN = re.compile('[' + CONSONANTS + ']{5,}')
BAD = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]|<\|[^>]+\|>|</?think>|```')


def low_mark_ratio(text):
    chars = [c for c in (text or '') if '\u0d00' <= c <= '\u0d7f']
    return len(chars) >= 20 and sum(unicodedata.category(c).startswith('M') for c in chars) / len(chars) < .08


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--eval', type=Path)
    args = parser.parse_args()
    source_hash = digest(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    target = args.output_dir / args.input.name
    counts = Counter(); paths = Counter(); texts = Counter(); hours = 0
    eval_paths = set()
    if args.eval:
        eval_paths = {json.loads(l)['audio_filepath'].replace('/vast/', '/lustrefs/taiga/vast40/', 1)
                      for l in args.eval.read_text().splitlines() if l.strip()}
    with args.input.open() as src, target.open('w') as out, \
         (args.output_dir / 'changes.jsonl').open('w') as log, \
         (args.output_dir / 'review.jsonl').open('w') as review:
        for line_no, line in enumerate(src, 1):
            if not line.strip():
                out.write(line); continue
            r = json.loads(line)
            before = r['text']
            if not isinstance(before, str): raise ValueError(f'Invalid text at {line_no}')
            counts['rows'] += 1
            after = repair_record(r)
            issues = []
            if after != before:
                counts['changed_rows'] += 1
                log.write(json.dumps({'line':line_no, 'before':before, 'after':after,
                                      'org_text':r.get('org_text'), 'asr_text':r.get('asr_text')}, ensure_ascii=False)+'\n')
                r['text'] = after
            if not after.strip(): issues.append('empty_text')
            ml = [c for c in before if '\u0d00' <= c <= '\u0d7f']
            marks = sum(unicodedata.category(c).startswith('M') for c in ml)
            # Inspect BEFORE repairs so partial repairs cannot hide damaged rows.
            if len(ml) >= 20 and marks / len(ml) < .08:
                issues.append('very_low_malayalam_mark_ratio')
                if low_mark_ratio(r.get('org_text')) and low_mark_ratio(r.get('asr_text')):
                    counts['low_mark_ratio_in_both_comparison_columns'] += 1
            if BARE_RUN.search(before): issues.append('long_unmarked_consonant_run')
            if not ml and after.strip(): issues.append('no_malayalam_script')
            if BAD.search(after): issues.append('control_replacement_or_markup')
            duration = r.get('duration')
            if not isinstance(duration, (int,float)) or isinstance(duration,bool) or not math.isfinite(duration) or duration <= 0:
                issues.append('invalid_duration')
            else:
                hours += duration / 3600
                if duration > 35: issues.append('duration_over_35s')
            audio = r.get('audio_filepath','')
            if not audio: issues.append('missing_audio_path')
            paths[audio] += 1; texts[after] += 1
            if audio.replace('/vast/','/lustrefs/taiga/vast40/',1) in eval_paths:
                issues.append('eval_audio_path_overlap')
            if issues:
                counts['review_rows'] += 1; counts.update(issues)
                review.write(json.dumps({'line':line_no,'issues':issues, **r},ensure_ascii=False)+'\n')
            out.write(json.dumps(r,ensure_ascii=False)+'\n' if after != before else line)
            if counts['rows'] % 50000 == 0:
                print(f"Scanned {counts['rows']:,}; repaired {counts['changed_rows']:,}; review {counts['review_rows']:,}",flush=True)
    assert digest(args.input) == source_hash, 'Source changed during audit'
    # Independently verify that only text changed, in the original row order.
    with args.input.open() as src, target.open() as dst:
        for a,b in zip(src,dst,strict=True):
            if not a.strip(): assert a==b; continue
            a,b=json.loads(a),json.loads(b)
            a.pop('text'); fixed=b.pop('text')
            assert a==b, 'Non-text field modified'
            assert repair_record({**b,'text':fixed})==fixed, 'Repair is not idempotent'
    report={'source':str(args.input.resolve()), 'output':str(target.resolve()),
            'source_sha256':source_hash, 'output_sha256':digest(target),
            'counts':counts, 'declared_hours':hours, 'unique_target_texts':len(texts),
            'duplicate_audio_path_rows_beyond_first':sum(n-1 for n in paths.values()),
            'verified':'Same rows/order, only text modified, idempotent, source unchanged.',
            'limitations':['No audio verification. ASR and original are supporting evidence, not ground truth.',
                          'Review flags are heuristic; flagged rows are retained, including partially repaired rows.',
                          'Repairs are an explicit word allowlist, ngng conjunct repair, and rr/ll virama restoration supported by BOTH comparison columns.',
                          'A successful report.json indicates completion; do not use output from an interrupted run.']}
    (args.output_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
