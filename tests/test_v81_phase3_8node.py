"""Fixed-step, no-code-switch Stage 3 launch and coverage contracts."""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from qasr.config import TrainConfig
from qasr.sampling import WeightedStratifiedLanguageSampler

CONFIG = 'configs/v8.1/03_hq_8node_b50_no_cs.yaml'
LAUNCHER = 'scripts/run_v81_phase3_8node_b50.sh'


def test_recipe_and_coverage():
    cfg = TrainConfig.from_yaml(CONFIG)
    old = TrainConfig.from_yaml('configs/v8.1/03_hq_12node.yaml')
    cfg.validate()
    assert cfg.code_switch_index is None and cfg.code_switch_probability == 0
    assert cfg.max_steps == 32679
    assert cfg.expected_world_size * cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps == 3200
    assert cfg.dataloader_num_workers == 10
    assert cfg.augmentation == old.augmentation
    assert cfg.train_manifest == old.train_manifest
    assert cfg.model_name_or_path == old.model_name_or_path
    assert cfg.output_dir != old.output_dir
    snapshot = json.loads(Path('configs/v8.0/manifest_counts.json').read_text())['counts']

    class Part:
        def __init__(self, lang, path):
            self.language = lang
            self.count = snapshot[lang][Path(path).stem]

        def __len__(self):
            return self.count

    sampler = WeightedStratifiedLanguageSampler(
        [Part(lang, path) for path, lang in cfg.train_manifest_specs],
        batch_size=50, seed=42, num_replicas=64, language_batch_counts=cfg.language_batch_counts,
    )
    assert sampler.num_batches == 19608
    assert 1 < cfg.max_steps / sampler.num_batches < 2


@pytest.mark.parametrize('rank', range(8))
def test_eight_node_commands(rank):
    env = {k: v for k, v in os.environ.items()
           if k not in ('CONFIG', 'MASTER_ADDR', 'MASTER_PORT', 'NNODES', 'NPROC_PER_NODE')}
    result = subprocess.run(['bash', LAUNCHER, str(rank)],
                            env={**env, 'DRY_RUN': '1'}, capture_output=True, text=True, check=True)
    for value in ('--nnodes=8', '--nproc_per_node=8', f'--node_rank={rank}',
                  '--master_addr=inception-H100-hpc-053', '--master_port=29584', CONFIG):
        assert value in result.stdout


def test_no_index_is_opened(monkeypatch, tmp_path):
    import qasr.code_switch
    from qasr.train import _build_code_switcher

    def forbidden(*args, **kwargs):
        pytest.fail('Disabled augmentation must not load a partner index')

    monkeypatch.setattr(qasr.code_switch, 'OnlineCodeSwitch', forbidden)
    cfg = TrainConfig.from_yaml(CONFIG)
    assert _build_code_switcher(cfg, None, None, []) is None
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    spec = importlib.util.spec_from_file_location('preflight_no_cs', 'scripts/preflight_v81_phase3_no_cs.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg.model_name_or_path = str(tmp_path)
    (tmp_path / 'config.json').write_text('{}')
    monkeypatch.setattr(module.TrainConfig, 'from_yaml', lambda _: cfg)
    monkeypatch.setattr(module, 'check_inputs', lambda *args, **kwargs: None)
    monkeypatch.setattr(module, 'check_stage2_export', lambda *args: None)
    monkeypatch.setattr(module.QASRProcessor, 'from_pretrained', lambda *args, **kw: SimpleNamespace(
        feature_extractor=SimpleNamespace(max_audio_clip_s=35), projector_pool_stride=1))
    monkeypatch.setattr('sys.argv', ['preflight'])
    module.main()
    cfg.code_switch_index = '/missing/index.json'
    with pytest.raises(ValueError, match='requires code_switch_index'):
        module.check_no_code_switch(cfg)


def test_longer_step_budget_scan(monkeypatch):
    import qasr.data

    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    from preflight_v81_phase3 import scan_coverage

    cfg = TrainConfig.from_yaml(CONFIG)
    monkeypatch.setattr(qasr.data, 'JsonlSpeechDataset', lambda *a, **kw: [0] * 10)
    scan_coverage(cfg, exact=False)
    with pytest.raises(ValueError, match='Coverage differs'):
        scan_coverage(cfg)
    cfg.max_steps = 0
    with pytest.raises(ValueError, match='Coverage differs'):
        scan_coverage(cfg, exact=False)
