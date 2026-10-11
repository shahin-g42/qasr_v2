"""v8.0 coverage, batch composition, evaluation budgets and launch contracts."""
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from qasr.config import TrainConfig
from qasr.eval_callback import WEREvalCallback
from qasr.sampling import WeightedStratifiedLanguageSampler

ROOT = Path(__file__).resolve().parents[1]


class Part:
    def __init__(self, n, language):
        self.n, self.language = n, language

    def __len__(self):
        return self.n


def test_distributed_coverage_and_batch_quotas():
    parts = [Part(60, 'ar'), Part(17, 'en'), Part(40, 'ar'), Part(3, 'ml')]
    languages = ['ar'] * 60 + ['en'] * 17 + ['ar'] * 40 + ['ml'] * 3
    quotas = {'ar': 5, 'en': 2, 'ml': 1}
    counts = Counter()
    streams = []
    for rank in range(4):
        sampler = WeightedStratifiedLanguageSampler(parts, batch_size=8, seed=42,
                    language_batch_counts=quotas, num_replicas=4, rank=rank)
        stream = list(sampler)
        assert len(stream) == len(sampler) == 40
        assert stream == list(sampler)
        for i in range(0, len(stream), 8):
            assert Counter(languages[j] for j in stream[i:i+8]) == quotas
        counts.update(stream)
        streams.append(stream)
        sampler.set_epoch(1)
        assert list(sampler) != stream
    assert set(counts) == set(range(120))
    for i, lang in enumerate(languages):
        total = sum(p.n for p in parts if p.language == lang)
        repeats = 5 * 4 * quotas[lang] / total
        assert counts[i] in (math.floor(repeats), math.ceil(repeats))
    # Exactly one traversal of the dominant language, disjoint across ranks.
    assert all(counts[i] == 1 for i, lang in enumerate(languages) if lang == 'ar')


def test_production_budget_and_config():
    config = TrainConfig.from_yaml(ROOT / 'configs/v8.0/02_full_10node.yaml')
    config.validate()
    snapshot = json.loads((ROOT / 'configs/v8.0/manifest_counts.json').read_text())['counts']
    totals = {lang: sum(n for name, n in files.items() if name.startswith('train_'))
              for lang, files in snapshot.items()}
    assert sum(totals.values()) == 99196711
    sampler = WeightedStratifiedLanguageSampler([Part(n, lang) for lang, n in totals.items()],
        batch_size=20, seed=42, language_batch_counts=config.language_batch_counts, num_replicas=80)
    assert sampler.num_batches == config.max_steps == 75469
    assert len(config.train_manifest_paths) == 22
    assert len(config.eval_manifest_paths) == 6
    assert config.resume_from_checkpoint is None
    assert config.expected_world_size == 80


@pytest.mark.parametrize('rank,extra,ok', [('0', {}, True), ('8', {}, True), ('9', {}, True),
    ('10', {}, False), ('-1', {}, False), ('0', {'NNODES': '8'}, False)])
def test_launcher_rank_contract(rank, extra, ok):
    result = subprocess.run(['bash', str(ROOT / 'scripts/run_v80_phase2_node.sh'), rank, 'master'],
        env={**os.environ, 'DRY_RUN': '1', **extra}, text=True, capture_output=True)
    assert (result.returncode == 0) == ok
    if ok:
        assert '--nnodes=10' in result.stdout
        assert f'--node_rank={rank}' in result.stdout


@pytest.mark.parametrize('global_budget,expected', [(True, 2), (False, 4)])
def test_eval_budget_across_manifests(tmp_path, global_budget, expected):
    paths = []
    for part in range(2):
        path = tmp_path / f'eval_{part}.jsonl'
        path.write_text(''.join(json.dumps({'audio_filepath': f'/fake/{part}_{i}.wav',
            'text': f'example {part} {i}', 'duration': 2.0}) + '\n' for i in range(4)))
        paths.append((str(path), 'en'))
    callback = WEREvalCallback(processor=SimpleNamespace(), eval_manifest_specs=paths,
        samples_per_language=2, seed=42, global_language_budget=global_budget,
        dataset_kwargs={"min_duration_seconds": 0.1, "max_duration_seconds": 35.0})
    seen = []
    callback._transcribe = lambda trainer, feature: seen.append(feature['text']) or feature['text']
    callback._evaluate_language(SimpleNamespace(), 'en', paths)
    assert len(seen) == expected


@pytest.mark.parametrize('quotas', [{'ar': 5}, {'ar': 5, 'en': 0}, {'ar': 5, 'en': True}, {'ar': 5, 'en': 4}])
def test_invalid_quotas_rejected(quotas):
    with pytest.raises(ValueError):
        WeightedStratifiedLanguageSampler([Part(20, 'ar'), Part(10, 'en')],
            batch_size=8, seed=42, language_batch_counts=quotas)


def test_preflight_catches_filter_changed_budget(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location('preflight_v80', ROOT / 'scripts/preflight_v80.py')
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)
    path = tmp_path / 'train_ar.jsonl'
    path.write_text(''.join(json.dumps({'audio_filepath': f'/fake/{i}.wav',
        'text': 'example', 'duration': duration}) + '\n' for i, duration in enumerate([2, 2, 40])))
    config = SimpleNamespace(train_manifest_specs=[(str(path), 'ar')], eval_manifest_specs=[],
        audio_root=None, min_duration_seconds=0.1, max_duration_seconds=35,
        expected_world_size=1, language_batch_counts={'ar': 1}, gradient_accumulation_steps=1, max_steps=2)
    assert preflight.scan_counts(config, {'ar': {'train_ar': 2}}) == {'ar': 2}
    with pytest.raises(ValueError, match='expected 3, eligible 2'):
        preflight.scan_counts(config, {'ar': {'train_ar': 3}})
    config.max_steps = 3
    with pytest.raises(ValueError, match='computed steps=2, configured=3'):
        preflight.scan_counts(config, {'ar': {'train_ar': 2}})
