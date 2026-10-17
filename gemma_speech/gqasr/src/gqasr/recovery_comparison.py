"""Evaluate the decoder-restoration candidate on identical multilingual samples."""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from .comparison_audit import (
    dump,
    execute,
    merge_shards,
    paired,
    pin_checkpoint,
    save_rows,
    shard_jobs,
)
from .metrics import score_predictions
from .processing import LANGUAGES
from .qwen_baseline import read_rows
from .text import normalize_text


def select_multilingual(index, predictions):
    rows = read_rows(predictions)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate evaluation IDs')
    if {r['language'] for r in rows} != set(LANGUAGES):
        raise ValueError('Expected evaluation predictions for all five languages')
    samples = []
    with sqlite3.connect(f'file:{Path(index).resolve()}?mode=ro', uri=True) as db:
        for row in rows:
            split, language, ordinal = row['id'].split(':')
            if split != 'eval' or language != row['language']:
                raise ValueError('Evaluation ID/language mismatch')
            found = db.execute('SELECT audio,text,duration,source,line FROM samples WHERE split=? AND language=? AND ordinal=?',
                               (split, language, int(ordinal))).fetchone()
            if found is None or normalize_text(found[1]) != normalize_text(row['reference']):
                raise ValueError(f"Reference mismatch: {row['id']}")
            if not Path(found[0]).is_file():
                raise FileNotFoundError(found[0])
            samples.append(dict(id=row['id'], language=language, reference=row['reference'], audio=found[0],
                                duration=found[2], source=found[3], source_line=found[4], audit_group=language))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--comparison')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--workers-per-gpu', type=int, default=4)
    parser.add_argument('--output')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    prior = Path(args.comparison).resolve() if args.comparison else root/'gqasr/verification/comparison-20261016T210052Z'
    gpus = args.gpus.split(',')
    if args.workers_per_gpu < 1 or not all(gpus) or len(set(gpus)) != len(gpus):
        parser.error('GPU IDs must be unique and nonempty')
    cfg = json.loads((prior/'training-config.json').read_text())
    prior_jobs = json.loads((prior/'jobs.json').read_text())
    command = prior_jobs[0]['command']
    predictions = Path(command[command.index('--predictions')+1])
    samples = select_multilingual(cfg['index_path'], predictions)
    output = Path(args.output).resolve() if args.output else root/'gqasr/verification'/time.strftime('recovery-multilingual-%Y%m%dT%H%M%SZ', time.gmtime())
    output.mkdir(parents=True, exist_ok=False)
    dump(output/'training-config.json', cfg)
    dump(output/'provenance.json', dict(source_comparison=str(prior), predictions=str(predictions),
                                      checkpoints=json.loads((prior/'checkpoint-sources.json').read_text()),
                                      gpus=gpus, workers_per_gpu=args.workers_per_gpu, modes=['auto'], precision='bf16', batch_size=1, max_new_tokens=768))
    for name in ('bridge', 'current'):
        pin_checkpoint(prior/'snapshots'/name, output/'snapshots'/name)
    for asset in ('config.json', 'tokenizer/tokenizer.json', 'preprocessor_config.json'):
        if (output/'snapshots/bridge'/asset).read_bytes() != (output/'snapshots/current'/asset).read_bytes():
            raise ValueError(f'Checkpoint configurations/processors differ: {asset}')
    sample_file = output/'samples.jsonl'
    save_rows(sample_file, samples)
    jobs = []
    for name, checkpoint, flags in (
        ('bridge', 'bridge', []), ('current', 'current', []),
        ('recovery', 'bridge', ['--bridge-from', str(output/'snapshots/current')]),
    ):
        command = [sys.executable, '-m', 'gqasr.language_replay', '--checkpoint', str(output/'snapshots'/checkpoint),
                   '--run-config', str(output/'training-config.json'), '--predictions', str(predictions),
                   '--samples', str(sample_file), '--output', str(output/name), '--modes', 'auto', *flags]
        jobs.append(dict(name=name, output=str(output/name), command=command, modes=['auto']))
    dump(output/'jobs.json', jobs)
    tasks = shard_jobs(jobs, samples, output, len(gpus)*args.workers_per_gpu)
    dump(output/'shard-jobs.json', tasks)
    print(f'Recovery comparison: {len(samples)} samples, {len(tasks)} shards, {len(gpus)} GPUs × {args.workers_per_gpu} workers; {output}', flush=True)
    execute(tasks, gpus, output, args.workers_per_gpu)
    merge_shards(jobs, tasks, samples)
    results, records = {}, {}
    for job in jobs:
        rows = read_rows(Path(job['output'])/'predictions-auto.jsonl')
        records[job['name']] = rows
        results[job['name']] = score_predictions(rows)
    comparisons = {}
    lines = ['# Multilingual recovery comparison', '',
             'Automatic language detection; strict metrics retain punctuation and diacritics. Chinese WER uses whitespace: prioritize CER.', '',
             '| Model | Language | N | WER | CER | Correct language | Limit hits |', '|---|---|---:|---:|---:|---:|---:|']
    for name, rows in records.items():
        for lang in LANGUAGES:
            selected = [r for r in rows if r['language'] == lang]
            scores = results[name][f'strict/language:{lang}']
            correct = sum(r['predicted_language'] == LANGUAGES[lang] for r in selected)
            hits = sum(r['generation_limit_hit'] for r in selected)
            lines.append(f"| {name} | {lang} | {len(selected)} | {scores['wer']:.2%} | {scores['cer']:.2%} | {correct}/{len(selected)} | {hits} |")
    for name in ('bridge', 'current'):
        comparisons[name] = {lang: paired([r for r in records[name] if r['language'] == lang],
                                         [r for r in records['recovery'] if r['language'] == lang]) for lang in LANGUAGES}
    dump(output/'comparison.json', dict(scores=results, recovery_minus_baseline_cer=comparisons))
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(f'COMPLETE: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
