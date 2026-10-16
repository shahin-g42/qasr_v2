"""One-shot, read-only Chinese regression isolation on cluster GPUs."""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .qwen_baseline import read_rows, select_samples, select_training_samples, summary


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def save_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def pin_checkpoint(source, destination):
    """Protect immutable checkpoint files from training's retention cleanup."""
    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    shutil.copytree(source, destination, copy_function=link_or_copy,
                    ignore=shutil.ignore_patterns('training_state.pt'))


def weight_audit(left, right):
    import torch
    from safetensors import safe_open
    groups = {}
    with safe_open(str(left / 'model.safetensors'), framework='pt') as a, safe_open(
        str(right / 'model.safetensors'), framework='pt'
    ) as b:
        if set(a.keys()) != set(b.keys()):
            raise ValueError('Bridge/current tensor keys differ')
        for key in a.keys():
            x, y = a.get_tensor(key), b.get_tensor(key)
            group = key.split('.')[0]
            stat = groups.setdefault(group, dict(tensors=0, changed=0, nonfinite=0, delta_sq=0., original_sq=0.))
            stat['tensors'] += 1
            stat['changed'] += int(not torch.equal(x, y))
            stat['nonfinite'] += int(not torch.isfinite(y).all())
            stat['delta_sq'] += (y.double()-x.double()).square().sum().item()
            stat['original_sq'] += x.double().square().sum().item()
    for stat in groups.values():
        stat['relative_l2_change'] = (stat['delta_sq'] / max(stat['original_sq'], 1e-30)) ** .5
    return groups


def paired(left, right):
    """Paired corpus CER delta with utterance bootstrap; identical IDs required."""
    import numpy as np

    from .metrics import edit_distance
    from .text import normalize_text
    if [r['id'] for r in left] != [r['id'] for r in right]:
        raise ValueError('Cannot compare different sample IDs/order')
    if any(a['reference'] != b['reference'] for a, b in zip(left, right, strict=True)):
        raise ValueError('Reference mismatch')
    errors, lengths = [], []
    for a, b in zip(left, right, strict=True):
        ref = normalize_text(a['reference']).replace(' ', '')
        ha = normalize_text(a['hypothesis']).replace(' ', '')
        hb = normalize_text(b['hypothesis']).replace(' ', '')
        errors.append(edit_distance(ref, hb)-edit_distance(ref, ha))
        lengths.append(len(ref))
    errors, lengths = np.array(errors), np.array(lengths)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(errors), size=(2000, len(errors)))
    delta = errors[draws].sum(1) / lengths[draws].sum(1).clip(1)
    return dict(cer_delta_right_minus_left=float(errors.sum()/max(lengths.sum(), 1)),
                paired_bootstrap_95_percent_interval=np.quantile(delta, [.025, .975]).tolist(),
                identical_transcripts=sum(a['hypothesis'] == b['hypothesis'] for a,b in zip(left,right,strict=True)),
                n=len(left))


def audit_data(samples, cfg, checkpoint, output):
    import numpy as np

    from ._qasr.audio import load_mono_audio
    from .augmentation import build_training_transforms
    from .processing import GQASRProcessor
    processor = GQASRProcessor(checkpoint)
    # Only construct code switching: noise augmentation is irrelevant to this audit.
    _, switcher = build_training_transforms(processor, {**cfg, 'augmentation': None})
    records, mixes = [], []
    rng = random.Random(42)
    candidates = [r['id'] for r in samples if r['id'].startswith('train:')]
    mix_ids = set(rng.sample(candidates, min(100, len(candidates))))
    for i, row in enumerate(samples):
        wave = load_mono_audio(row['audio'], 16000)
        ids = processor.target(row['reference'], 'zh')
        records.append(dict(id=row['id'], audio=row['audio'], samples=len(wave),
                            waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                            finite=bool(np.isfinite(wave).all()), rms=float(np.sqrt(np.mean(wave**2))),
                            duration_delta=len(wave)/16000-row['duration'], target_tokens=len(ids),
                            target_roundtrip=processor.decode(ids) == row['reference'].strip()))
        if switcher and row['id'] in mix_ids:
            mixed, audit = switcher.augment(dict(audio_path=row['audio'], waveform=wave,
                                                text=row['reference'], language='zh'), p=1., rng=rng)
            segments = mixed.get('segments', [])
            mixes.append(dict(id=row['id'], audit=audit, assigned_language=mixed['language'],
                              text=mixed['text'], segments=segments,
                              exact_space_join=(mixed['text'] == ' '.join(s['text'] for s in segments)) if segments else None,
                              target=processor.tokenizer.decode(processor.target(mixed['text'], mixed['language']), skip_special_tokens=False)))
        if i % 50 == 0:
            print(json.dumps(dict(event='data_audit', done=i+1, total=len(samples))), flush=True)
    save_rows(output / 'audio-target-audit.jsonl', records)
    save_rows(output / 'code-switch-audit.jsonl', mixes)
    dump(output / 'data-summary.json', dict(
        examples=len(records), nonfinite=sum(not r['finite'] for r in records),
        target_roundtrip_mismatches=sum(not r['target_roundtrip'] for r in records),
        overlong_targets=sum(r['target_tokens'] > cfg['max_target_length'] for r in records),
        duration_mismatches=sum(abs(r['duration_delta']) > .1 for r in records),
        duplicate_waveforms=len(records)-len({r['waveform_sha256'] for r in records}),
        code_switch_language_policy=switcher.cfg['language_policy'] if switcher else None,
        applied_mixes=sum(r['audit']['applied'] for r in mixes),
        chinese_anchor_relabelled=sum(r['assigned_language'] != 'zh' for r in mixes),
        note='Code-switch attempts force p=1 for coverage; this is not the production frequency. Text roundtrip mismatches need inspection, not automatic rejection.',
    ))


def execute(jobs, gpus, output):
    pending = list(jobs)
    active = []
    try:
        while pending or active:
            used = {gpu for _, gpu, _, _ in active}
            for gpu in gpus:
                if gpu in used or not pending:
                    continue
                job = pending.pop(0)
                log = (output / (job['name'] + '.log')).open('w')
                child = subprocess.Popen(job['command'], stdout=log, stderr=subprocess.STDOUT,
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': gpu, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1'})
                active.append((job, gpu, child, log))
                print(f"Started {job['name']} on GPU {gpu}", flush=True)
            completed = []
            for item in active:
                job, gpu, child, log = item
                code = child.poll()
                if code is not None:
                    log.close()
                    if code:
                        raise RuntimeError(f"{job['name']} failed ({code}); see {output / (job['name']+'.log')}")
                    completed.append(item)
                    print(f"Completed {job['name']}", flush=True)
            active = [item for item in active if item not in completed]
            progress = {}
            for job, _, _, _ in active:
                progress[job['name']] = sum(sum(1 for _ in p.open()) for p in Path(job['output']).glob('predictions-*.jsonl'))
            print(json.dumps(dict(event='comparison_progress', queued=len(pending), generated_rows=progress)), flush=True)
            if active:
                time.sleep(5)
    finally:
        for _, _, child, _ in active:
            if child.poll() is None:
                child.terminate()
        for _, _, child, log in active:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            log.close()


def report(jobs, output):
    results = {}
    all_rows = {}
    for job in jobs:
        modes = {}
        for mode in ('auto', 'chinese'):
            rows = read_rows(Path(job['output']) / f'predictions-{mode}.jsonl')
            all_rows[job['name'], mode] = rows
            modes[mode] = {group: summary([r for r in rows if r['audit_group'] == group])
                           for group in sorted({r['audit_group'] for r in rows})}
        results[job['name']] = modes
        teacher_rows = [r for r in all_rows[job['name'], 'auto'] if r.get('teacher_forced')]
        if teacher_rows:
            dump(Path(job['output']) / 'teacher-summary.json', {
                group: {part + '_nll': sum(r['teacher_forced'][part + '_nll'] * r['teacher_forced'][part + '_tokens']
                                          for r in teacher_rows if r['audit_group'] == group) /
                                      sum(r['teacher_forced'][part + '_tokens'] for r in teacher_rows if r['audit_group'] == group)
                        for part in ('header', 'transcript')}
                for group in sorted({r['audit_group'] for r in teacher_rows})
            })
    comparisons = {}
    for name in results:
        comparisons[name + ': forced-minus-auto'] = paired(all_rows[name, 'auto'], all_rows[name, 'chinese'])
    for name in results:
        if name == 'current':
            continue
        left, right = all_rows['current', 'chinese'], all_rows[name, 'chinese']
        comparisons[name + ': minus-current-forced'] = paired(left, right)
        for group in sorted({r['audit_group'] for r in left}):
            comparisons[name + ': minus-current-forced:' + group] = paired(
                [r for r in left if r['audit_group'] == group],
                [r for r in right if r['audit_group'] == group])
    dump(output / 'comparison.json', dict(scores=results, paired_comparisons=comparisons,
        interpretation={
            'language_selection': 'Forced Chinese improving CER localizes a language-selection contribution; inspect header NLL too.',
            'components': 'Compare the full 2x2 bridge/decoder matrix. Swaps can break coadaptation; swap failure alone does not prove decoder forgetting.',
            'audio': 'Shifted audio should worsen reference NLL/CER. Similar outputs alone are not proof that the encoder is ignored.',
            'batch_precision': 'Compare current-b4/current-fp32 against current on identical examples.',
            'data': 'Baselines succeeding on the same audio/reference argues against widespread label corruption; disagreements still need listening.',
            'causality': 'This isolates inference/component failure. Attribution to code switching, sampling or augmentation needs controlled retraining; no automatic causal verdict.',
        }))
    lines = ['# Chinese regression comparison', '', '| Experiment | Group | Auto CER | Forced CER |', '|---|---|---:|---:|']
    for name, modes in results.items():
        for group, value in modes['auto'].items():
            lines.append(f"| {name} | {group} | {value['strict']['cer']:.2%} | {modes['chinese'][group]['strict']['cer']:.2%} |")
    lines += ['', 'See comparison.json for paired confidence intervals, audio-target-audit.jsonl for data checks,',
              'code-switch-audit.jsonl for actual assembled targets, and per-job predictions for teacher-forced NLL.',
              'Component swaps are diagnostic interventions, not deployable checkpoints or proof of a training cause.']
    (output / 'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--run-dir')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--per-group', type=int, default=250)
    parser.add_argument('--eval-step', default='7294')
    parser.add_argument('--output')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if args.per_group < 1 or not all(gpus) or len(set(gpus)) != len(gpus):
        parser.error('Positive per-group and unique nonempty GPU IDs required')
    root = Path(args.root).resolve()
    run = Path(args.run_dir).resolve() if args.run_dir else root / 'gqasr/outputs/full_b40_frozen'
    cfg = json.loads((run / 'config.json').read_text())
    output = Path(args.output).resolve() if args.output else root / 'gqasr/verification' / time.strftime('comparison-%Y%m%dT%H%M%SZ', time.gmtime())
    output.mkdir(parents=True, exist_ok=False)
    print(f'Audit output: {output}', flush=True)
    dump(output / 'training-config.json', cfg)
    import torch
    import transformers
    dump(output / 'runtime.json', dict(torch=str(torch.__version__), transformers=transformers.__version__,
        python=sys.version, git_commit=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        gpus=gpus))
    predictions = run / f'predictions-{args.eval_step}.jsonl'
    samples = [{**r, 'audit_group': 'eval', 'stratum': 'eval'} for r in select_samples(cfg['index_path'], predictions)]
    train, population = select_training_samples(cfg['index_path'], args.per_group, 42)
    samples += [{**r, 'audit_group': 'train_' + r['stratum']} for r in train]
    sample_file = output / 'samples.jsonl'
    save_rows(sample_file, samples)
    dump(output / 'selection.json', dict(seed=42, population=population, per_group=args.per_group,
        sample_sha256=hashlib.sha256(sample_file.read_bytes()).hexdigest()))
    sources = {'bridge': root / 'gqasr/outputs/bridge/final',
               'current': run / json.loads((run / 'latest.json').read_text())['checkpoint']}
    early = run / 'checkpoint-00001042'
    if early.exists():
        sources['early1042'] = early
    dump(output / 'checkpoint-sources.json', {k: str(v) for k,v in sources.items()})
    for name, source in sources.items():
        print(f'Pinning {name}: {source}', flush=True)
        pin_checkpoint(source, output / 'snapshots' / name)
    snapshots = output / 'snapshots'
    drift = weight_audit(snapshots / 'bridge', snapshots / 'current')
    dump(output / 'weight-drift.json', drift)
    if any(v['nonfinite'] for v in drift.values()):
        raise ValueError('Nonfinite checkpoint tensors detected; inspect weight-drift.json')
    if drift['gqasr_acoustic']['changed']:
        raise ValueError('Frozen encoder changed: see weight-drift.json. A bridge/decoder-only matrix would not isolate components.')
    for source in sources:
        for asset in ('tokenizer/tokenizer.json', 'preprocessor_config.json'):
            if (snapshots / source / asset).read_bytes() != (snapshots / 'bridge' / asset).read_bytes():
                raise ValueError(f'Processor changed in {source}: {asset}; cannot isolate weights from preprocessing')
    audit_data(samples, cfg, snapshots / 'current', output)
    variants = [(name, name, []) for name in sources]
    variants += [('current-old-bridge', 'current', ['--bridge-from', str(snapshots / 'bridge')]),
                 ('bridge-current-bridge', 'bridge', ['--bridge-from', str(snapshots / 'current')]),
                 ('current-shifted-audio', 'current', ['--audio-control', 'shift']),
                 ('current-b4', 'current', ['--batch-size', '4']),
                 ('current-fp32', 'current', ['--precision', 'fp32'])]
    jobs = []
    for name, checkpoint, flags in variants:
        destination = output / name
        command = [sys.executable, '-m', 'gqasr.language_replay', '--checkpoint', str(snapshots / checkpoint),
                   '--run-config', str(output / 'training-config.json'), '--predictions', str(predictions),
                   '--samples', str(sample_file), '--output', str(destination), '--teacher-forced', *flags]
        jobs.append(dict(name=name, output=str(destination), command=command))
    for size in ('0.6', '1.7'):
        name = f'qwen{size}b'
        model = root / f'models/qwen3-asr-{size}b-hf'
        if not (model / 'config.json').exists():
            raise FileNotFoundError(f'{model}; run the comparison shell script to download baselines first')
        command = [sys.executable, '-m', 'gqasr.qwen_baseline', '--train', '--samples', str(sample_file),
                   '--run-config', str(output / 'training-config.json'), '--model', str(model),
                   '--output', str(output / name)]
        jobs.append(dict(name=name, output=str(output / name), command=command))
    dump(output / 'jobs.json', jobs)
    execute(jobs, gpus, output)
    report(jobs, output)
    print(f'COMPLETE: {output / "REPORT.md"}', flush=True)


if __name__ == '__main__':
    main()
