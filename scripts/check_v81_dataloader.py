"""Bounded cluster diagnostic of the real dataset, sampler, collator and augmentations.

No model weights, optimizer, or training output are written. Run once per node,
optionally using torchrun for independent rank/worker RNG and I/O stress.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from qasr.augmentation import build_augmenter
from qasr.code_switch import OnlineCodeSwitch
from qasr.collator import QASRDataCollator
from qasr.config import TrainConfig
from qasr.data import CombinedSpeechDataset, JsonlSpeechDataset, ResilientAudioDataset
from qasr.processing import QASRProcessor
from qasr.sampling import WeightedStratifiedLanguageSampler


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def check_wave(wave, sr, max_seconds):
    if wave.ndim != 1 or not np.isfinite(wave).all() or not 0 < len(wave) <= sr * max_seconds:
        raise AssertionError('Invalid or overlong augmented waveform')


class TrackedTransform:
    """Observe the actual transform output without replacing its RNG or gate."""
    def __init__(self, name, transform, counts):
        self.name, self.transform, self.counts = name, transform, counts

    def __call__(self, original, *args):
        result = self.transform(original, *args)
        self.counts[self.name + '_calls'] += 1
        equal = torch.equal(original, result) if isinstance(original, torch.Tensor) else np.array_equal(original, result)
        self.counts[self.name + '_changed'] += int(not equal)
        if not (torch.isfinite(result).all() if isinstance(result, torch.Tensor) else np.isfinite(result).all()):
            raise AssertionError(f'{self.name} produced nonfinite values')
        if self.name == 'noise':
            source = getattr(self.transform, 'last_noise_source', None)
            if source == 'gaussian':
                raise AssertionError('MUSAN noise fell back to Gaussian; check noise files and mounts')
        if self.name == 'spec':
            mask = args[0].bool()
            if not torch.equal(original[~mask], result[~mask]):
                raise AssertionError('SpecAugment modified padded frames')
        return result


class AuditedCollator:
    def __init__(self, collator):
        self.collator = collator = copy.deepcopy(collator)
        self.counts = Counter()
        self.rows = []
        self.collator.audit_callback = self.capture
        aug = collator.augmenter
        if aug:
            for name, attr in [('speed', 'speed_perturb'), ('noise', 'noise_injection'),
                               ('codec', 'codec_augment'), ('spec', 'spec_augment')]:
                transform = getattr(aug, attr)
                if transform:
                    setattr(aug, attr, TrackedTransform(name, transform, self.counts))

    def capture(self, features, audits):
        self.rows = audits
        for row in features:
            self.counts['label:' + str(row.get('language'))] += 1
            if row.get('code_switch'):
                assert row['text'] == ' '.join(s['text'] for s in row['segments'])
                check_wave(row['waveform'], self.collator.sampling_rate, 35)

    def __call__(self, features):
        self.counts.clear()
        batch = self.collator(features)
        for value in batch.values():
            if not torch.isfinite(value).all():
                raise AssertionError('Nonfinite batch tensor')
        if not (batch['labels'] != -100).any(dim=1).all():
            raise AssertionError('Example has no supervised target tokens')
        self.counts['examples'] = len(features)
        for feature in features:
            self.counts['anchor:' + str(feature.get('language'))] += 1
            self.counts.update(feature.get('_loader_skips', {}))
        for audit in self.rows:
            self.counts['triggered'] += int(audit['triggered'])
            self.counts['applied'] += int(audit['applied'])
            self.counts['reason:' + audit['reason']] += 1
            if audit.get('pair'):
                self.counts['pair:' + audit['pair']] += 1
        return dict(self.counts)


class AuditedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        before = self.dataset.skip_stats
        row = self.dataset[index]
        after = self.dataset.skip_stats
        return {**row, '_loader_skips': {key: after[key] - before[key] for key in before}}


def forced_checks(cfg, sr):
    """Test each configured acoustic transform in isolation at p=1 and p=0."""
    t = np.arange(sr * 2, dtype=np.float32) / sr
    wave = (0.15 * np.sin(2 * np.pi * 233 * t) + 0.07 * np.sin(2 * np.pi * 3200 * t)).astype(np.float32)
    report = {}
    attributes = {'speed_perturb': 'speed_perturb', 'noise_injection': 'noise_injection',
                  'codec_augment': 'codec_augment', 'spec_augment': 'spec_augment'}
    for name, attr in attributes.items():
        if not (cfg.augmentation or {}).get(name, {}).get('enabled'):
            report[name] = 'disabled'
            continue
        section = copy.deepcopy(cfg.augmentation[name])
        section['p'] = 1.0
        aug = build_augmenter({name: section}, sr)
        transform = getattr(aug, attr)
        if name == 'noise_injection' and not transform._noise_files:
            raise AssertionError('No MUSAN audio discovered; refusing Gaussian fallback')
        changed = 0
        for _ in range(8):
            if name == 'spec_augment':
                source = torch.ones(2, 200, 128)
                mask = torch.ones(2, 200, dtype=torch.bool)
                mask[1, 130:] = False
                result = transform(source, mask)
                assert torch.equal(source[~mask], result[~mask])
                assert torch.isfinite(result).all()
                changed += int(not torch.equal(source, result))
            else:
                result = transform(wave, sr)
                check_wave(result, sr, 35)
                changed += int(not np.array_equal(result, wave))
                if name == 'noise_injection':
                    assert transform.last_noise_source != 'gaussian', 'MUSAN file failed to load'
            transform.p = 0.0
            if name == 'spec_augment':
                assert torch.equal(transform(source, mask), source)
            else:
                assert np.array_equal(transform(wave, sr), wave)
            transform.p = 1.0
        if not changed:
            raise AssertionError(f'{name} was a no-op in every forced trial')
        report[name] = {'forced_changed': changed, 'trials': 8, 'disabled_identity': True}
    return report


def check_rate(counts, p):
    n = counts['examples']
    deviation = abs(counts['triggered'] - n * p)
    tolerance = max(2, 5 * math.sqrt(n * p * (1 - p)))
    if deviation > tolerance:
        raise AssertionError(f'Code-switch trigger count inconsistent with p={p}: {dict(counts)}')
    if p and not counts['applied']:
        raise AssertionError('No successful code switch; inspect eligibility/index')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v8.1/02_full_10node.yaml')
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--records-per-manifest', type=int, default=2000,
                        help='bounded prefix per manifest; 0 scans all manifests as training does')
    parser.add_argument('--output', default='dataloader_checks')
    parser.add_argument('--allow-pilot', action='store_true')
    args = parser.parse_args()
    if args.batches < 1 or args.workers < 0 or args.records_per_manifest < 0:
        parser.error('Invalid batches/workers/records-per-manifest')
    cfg = TrainConfig.from_yaml(args.config)
    cfg.validate()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    seed = cfg.seed + 100003 * rank
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    processor = QASRProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    sr = int(processor.feature_extractor.sampling_rate)
    mixer = OnlineCodeSwitch(cfg.code_switch_index, processor.tokenizer, allow_pilot=args.allow_pilot)
    if sr != mixer.cfg['sample_rate'] or mixer.cfg['max_target_tokens'] > cfg.max_target_length:
        raise ValueError('Index and training audio/token configuration differ')
    indexed = {source['path']: source for source in mixer.index['sources']}
    if {str(Path(path).resolve()) for path in cfg.train_manifest_paths} != {
            path for path in indexed if Path(path).name.startswith('train_')}:
        raise ValueError('Training manifests differ from indexed manifests')
    for path, source in indexed.items():
        stat = Path(path).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source['size'], source['mtime_ns']):
            raise ValueError(f'Indexed manifest changed: {path}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {'rank': rank, 'world_size': world, 'seed': seed, 'workers': args.workers,
              'prefix_records_per_manifest': args.records_per_manifest,
              'p': cfg.code_switch_probability, 'forced_checks': forced_checks(cfg, sr)}
    with tempfile.TemporaryDirectory(prefix='qasr-loader-') as temp:
        parts = []
        for i, (path, lang) in enumerate(cfg.train_manifest_specs):
            source = Path(path)
            if args.records_per_manifest:
                prefix = Path(temp) / f'{i}.jsonl'
                with source.open() as reader, prefix.open('w') as writer:
                    writer.writelines(itertools.islice(reader, args.records_per_manifest))
                path = prefix
            parts.append(JsonlSpeechDataset(path, language=lang,
                audio_root=cfg.audio_root or source.parent,
                min_duration_seconds=cfg.min_duration_seconds,
                max_duration_seconds=cfg.max_duration_seconds))
        dataset = ResilientAudioDataset(CombinedSpeechDataset(parts), sampling_rate=sr,
            min_audio_seconds=cfg.min_duration_seconds, max_audio_seconds=cfg.max_duration_seconds,
            tokenizer=processor.tokenizer, max_target_length=cfg.max_target_length)
        sampler = WeightedStratifiedLanguageSampler(parts, batch_size=cfg.per_device_train_batch_size,
            language_batch_counts=cfg.language_batch_counts, seed=cfg.data_seed or cfg.seed,
            num_replicas=world, rank=rank)
        collator = QASRDataCollator(processor, cfg.language, cfg.min_duration_seconds,
            cfg.max_duration_seconds, cfg.max_target_length,
            augmenter=build_augmenter(cfg.augmentation, sr), enforce_augmented_duration=True,
            code_switcher=mixer, code_switch_probability=cfg.code_switch_probability)
        loader = DataLoader(AuditedDataset(dataset), sampler=sampler, batch_size=cfg.per_device_train_batch_size,
            num_workers=args.workers, collate_fn=AuditedCollator(collator),
            worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed),
            **({'multiprocessing_context': 'spawn'} if args.workers else {}))
        counts = Counter()
        started = time.monotonic()
        batches = 0
        for item in itertools.islice(loader, args.batches):
            counts.update(item)
            batches += 1
            if batches % 10 == 0:
                print(f'rank={rank} batches={batches} examples={counts["examples"]} '
                      f'triggered={counts["triggered"]} applied={counts["applied"]}', flush=True)
        report.update(counts=dict(counts), batches=batches, seconds=time.monotonic() - started,
                      trigger_rate=counts['triggered'] / max(1, counts['examples']),
                      applied_rate=counts['applied'] / max(1, counts['examples']))
        (output / f'rank-{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        if batches != args.batches:
            raise AssertionError('Dataset exhausted before requested batches; increase --records-per-manifest')
        check_rate(counts, cfg.code_switch_probability)
        print(json.dumps(report, indent=2), flush=True)
        print(f'PASS rank {rank}: loader/augmentation checks; model/GPU forward not tested.', flush=True)


if __name__ == '__main__':
    main()
