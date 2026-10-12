"""96-GPU configuration and launcher contracts without cluster access."""
import json
import os
import subprocess
from pathlib import Path

import pytest

from qasr.config import TrainConfig
from qasr.sampling import WeightedStratifiedLanguageSampler


class Part:
    def __init__(self, n, language):
        self.n, self.language = n, language

    def __len__(self):
        return self.n


def test_budget_and_unchanged_data_recipe():
    old = TrainConfig.from_yaml('configs/v8.1/02_full_10node.yaml')
    cfg = TrainConfig.from_yaml('configs/v8.1/02_full_12node.yaml')
    cfg.validate()
    snapshot = json.loads(Path('configs/v8.0/manifest_counts.json').read_text())['counts']
    parts = [Part(sum(n for name, n in files.items() if name.startswith('train_')), lang)
             for lang, files in snapshot.items()]
    sampler = WeightedStratifiedLanguageSampler(parts, batch_size=cfg.per_device_train_batch_size,
        seed=42, language_batch_counts=cfg.language_batch_counts, num_replicas=cfg.expected_world_size)
    assert sampler.num_batches == cfg.max_steps == 62891
    assert cfg.expected_world_size * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 1920
    for attr in ('train_manifest', 'eval_manifest', 'code_switch_index', 'code_switch_probability',
                 'augmentation', 'language_batch_counts', 'model_name_or_path', 'learning_rate'):
        assert getattr(cfg, attr) == getattr(old, attr)
    assert cfg.output_dir != old.output_dir
    assert cfg.warmup_steps == 833
    assert cfg.lr_scheduler_kwargs['num_decay_steps'] == 6250
    assert cfg.save_steps == cfg.eval_steps == 2083


@pytest.mark.parametrize('rank', range(12))
def test_all_twelve_ranks(rank):
    result = subprocess.run(['bash', 'scripts/run_v81_phase2_12node.sh', str(rank)],
        env={**os.environ, 'DRY_RUN': '1', 'NNODES': '12', 'NPROC_PER_NODE': '8',
             'MASTER_ADDR': 'inception-H100-hpc-051'}, capture_output=True, text=True, check=True)
    assert '--nnodes=12' in result.stdout
    assert f'--node_rank={rank}' in result.stdout
    assert '--master_addr=inception-H100-hpc-051' in result.stdout
    assert '--master_port=29582' in result.stdout
    assert 'configs/v8.1/02_full_12node.yaml' in result.stdout


@pytest.mark.parametrize('args,env,ok', [(['12'], {}, False), (['-1'], {}, False),
    (['0'], {'NNODES': '10'}, False), (['0', 'localhost', '--smoke-test'], {'NNODES': '1'}, True)])
def test_launcher_limits_and_smoke(args, env, ok):
    result = subprocess.run(['bash', 'scripts/run_v81_phase2_12node.sh', *args],
        env={**os.environ, 'DRY_RUN': '1', **env}, capture_output=True, text=True)
    assert (result.returncode == 0) == ok
    if ok:
        assert '--master_addr=localhost' in result.stdout
        assert '--smoke-test' in result.stdout
