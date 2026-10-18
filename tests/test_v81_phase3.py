"""Stage-transition safety, real sampler budget and repeatable evaluation commands."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from qasr.config import TrainConfig
from qasr.sampling import WeightedStratifiedLanguageSampler


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path('scripts') / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curated_budget_with_real_sampler():
    cfg = TrainConfig.from_yaml('configs/v8.1/03_hq_12node.yaml')
    previous = TrainConfig.from_yaml('configs/v8.1/02_full_12node.yaml')
    cfg.validate()
    counts = json.loads(Path('configs/v8.0/manifest_counts.json').read_text())['counts']

    class Part:
        def __init__(self, path, lang):
            self.language = lang
            self.n = counts[lang][Path(path).stem]

        def __len__(self):
            return self.n

    parts = [Part(path, lang) for path, lang in cfg.train_manifest_specs]
    sampler = WeightedStratifiedLanguageSampler(
        parts, batch_size=20, seed=42, language_batch_counts=cfg.language_batch_counts,
        num_replicas=96,
    )
    assert sampler.num_batches == cfg.max_steps == 32679
    assert cfg.model_name_or_path == previous.output_dir
    assert cfg.output_dir != previous.output_dir
    assert cfg.resume_from_checkpoint is None
    assert cfg.learning_rate == pytest.approx(previous.learning_rate / 5)
    assert cfg.code_switch_probability == previous.code_switch_probability == 0.4
    assert '/cs_v1_ml_v2/assembled/index.json' in cfg.code_switch_index
    assert '/cs_v1/assembled/index.json' in previous.code_switch_index
    assert cfg.eval_manifest == previous.eval_manifest
    for lang in ('ar', 'en'):
        assert not any(p.endswith(f'train_{lang}_q3asr.jsonl') for p in cfg.train_manifest[lang])
    for lang in ('hi', 'ml', 'zh'):
        assert cfg.train_manifest[lang] == previous.train_manifest[lang]


@pytest.mark.parametrize('step,smoke,loss,ok', [
    (62891, False, 1.5, True), (59470, False, 1.5, False),
    (62891, True, 1.5, False), (62891, False, float('nan'), False),
])
def test_completion_gate(tmp_path, step, smoke, loss, ok):
    gate = load_script('preflight_v81_phase3').check_stage2_export
    model = tmp_path / 'full'
    model.mkdir()
    with pytest.raises(ValueError, match='not ready'):
        gate(str(model), str(tmp_path / 'hq'))
    for name, value in {
        'training_config.json': {'max_steps': 62891, 'smoke_test': smoke},
        'trainer_state.json': {'global_step': step},
        'train_results.json': {'train_loss': loss},
    }.items():
        (model / name).write_text(json.dumps(value))
    if ok:
        gate(str(model), str(tmp_path / 'hq'))
    else:
        with pytest.raises(ValueError):
            gate(str(model), str(tmp_path / 'hq'))
    with pytest.raises(ValueError, match='overwrite'):
        gate(str(model), str(model))


@pytest.mark.parametrize('rank', range(12))
def test_stage3_launch_ranks(rank):
    env = {k: v for k, v in os.environ.items()
           if k not in ('CONFIG', 'MASTER_PORT', 'MASTER_ADDR', 'NNODES', 'NPROC_PER_NODE')}
    result = subprocess.run(['bash', 'scripts/run_v81_phase3_12node.sh', str(rank)],
                            env={**env, 'DRY_RUN': '1'}, capture_output=True, text=True, check=True)
    assert '--nnodes=12' in result.stdout
    assert f'--node_rank={rank}' in result.stdout
    assert '--master_addr=inception-H100-hpc-051' in result.stdout
    assert '--master_port=29583' in result.stdout
    assert '03_hq_12node.yaml' in result.stdout


def test_before_after_eval_uses_same_inputs(tmp_path):
    commands = load_script('eval_v81_phase3').evaluation_commands
    cfg = TrainConfig.from_yaml('configs/v8.1/03_hq_12node.yaml')
    before = list(commands(cfg, 'before', tmp_path, 1000))
    after = list(commands(cfg, 'after', tmp_path, 1000))
    assert len(before) == len(after) == 6
    for (out_a, a), (out_b, b) in zip(before, after, strict=True):
        assert out_a == out_b
        a[a.index('--model') + 1] = b[b.index('--model') + 1]
        assert a == b
        assert a[a.index('--max-new-tokens') + 1] == '768'
    assert len(list(commands(cfg, 'before', tmp_path, 1000, 'ml'))) == 1
