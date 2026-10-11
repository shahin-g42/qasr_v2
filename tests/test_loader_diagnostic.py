"""Run the cluster diagnostic's real transform probes and spawn-worker collator."""
import random
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from check_v81_dataloader import AuditedCollator, check_rate, forced_checks, seed_worker
from test_qasr_processor import toy_processor

from qasr.augmentation import build_augmenter
from qasr.collator import QASRDataCollator


def recipe(tmp_path):
    sf.write(tmp_path / 'noise.wav', np.random.default_rng(1).normal(0, .1, 32000), 16000)
    return {'speed_perturb': {'enabled': True, 'p': .37, 'rate_range': [.85, 1.15]},
            'noise_injection': {'enabled': True, 'p': .37, 'noise_dir': str(tmp_path)},
            'codec_augment': {'enabled': True, 'p': .37},
            'spec_augment': {'enabled': True, 'p': .37}}


def test_forced_checks_and_noise_failure(tmp_path):
    random.seed(42)
    cfg = SimpleNamespace(augmentation=recipe(tmp_path))
    report = forced_checks(cfg, 16000)
    assert all(v['forced_changed'] > 0 for v in report.values())
    (tmp_path / 'noise.wav').unlink()
    with pytest.raises(AssertionError, match='MUSAN'):
        forced_checks(cfg, 16000)


def test_rate_check_detects_always_on_and_off():
    check_rate(Counter(examples=2000, triggered=810, applied=700), .4)
    for triggered in [0, 2000]:
        with pytest.raises(AssertionError, match='inconsistent'):
            check_rate(Counter(examples=2000, triggered=triggered, applied=triggered), .4)


def test_real_collator_with_spawn_workers(tmp_path):
    processor = toy_processor()
    collator = QASRDataCollator(processor, 'en', .1, 35, 512,
        augmenter=build_augmenter(recipe(tmp_path), 16000), enforce_augmented_duration=True)
    wave = np.sin(np.arange(16000, dtype=np.float32) / 12) * .1
    rows = [{'audio_path': '/preloaded.wav', 'waveform': wave, 'text': 'hello world', 'language': 'en'}] * 12
    def run():
        loader = DataLoader(rows, batch_size=2, num_workers=2,
            multiprocessing_context='spawn', worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(51), collate_fn=AuditedCollator(collator))
        return list(loader)
    first = run()
    assert first == run()
    assert sum(row['examples'] for row in first) == 12
    assert sum(row['speed_calls'] for row in first) == 12
    assert sum(row['spec_calls'] for row in first) == 6
