import copy
import json
import os
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.multiprocessing as mp
import yaml
from gqasr.config import load_config
from gqasr.resume import validate_resume, restore_rng
from gqasr.train import train, schedule
from test_restart import saved_recipe, distributed_train_worker
from test_training import fixture

ROOT = Path(__file__).resolve().parents[3]


def four_config():
    return load_config(ROOT / 'gemma_speech/gqasr/configs/03_encoder12_b120_resume_4node.yaml')


def test_four_node_budget_and_guarded_resume():
    saved, _ = saved_recipe()
    cfg = four_config()
    assert cfg['batch_size'] == 120 and cfg['accumulation_steps'] == 1
    assert cfg['expected_world_size'] == 32
    assert cfg['workers'] == 8 and cfg['prefetch_factor'] == 2
    assert cfg['language_batch_counts'] == dict(ar=60, en=36, hi=12, ml=6, zh=6)
    with pytest.raises(ValueError):
        validate_resume(saved, cfg, {'x': 1}, allow_batch_change=True)
    result = validate_resume(saved, cfg, {'x': 1}, allow_batch_change=True, allow_world_size_change=True)
    assert result['global_batch'] == {'from': 7680, 'to': 3840}
    assert result['rng_policy'] == 'reseed_for_new_world'
    assert result['next_update'] == 7501 and result['remaining_updates'] == 23946
    cfg['learning_rate'] *= 2
    with pytest.raises(ValueError):
        validate_resume(saved, cfg, {'x': 1}, allow_batch_change=True, allow_world_size_change=True)


def test_default_launcher_uses_four_nodes_and_rejects_rank_four():
    env = {k: v for k, v in os.environ.items() if k not in {
        'GQASR_CONFIG', 'GQASR_LAUNCH_NNODES', 'PROJECT_DIR', 'NNODES', 'NPROC_PER_NODE',
        'MASTER_ADDR', 'MASTER_PORT', 'GQASR_RESUME_CHECKPOINT'}}
    env['DRY_RUN'] = '1'
    script = str(ROOT / 'scripts/resume_gqasr_encoder.sh')
    result = subprocess.run(['bash', script, '0'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for arg in ('--nnodes=4', '--nproc_per_node=8', '--node_rank=0',
                '--master_addr=inception-H100-hpc-020', '03_encoder12_b120_resume_4node.yaml',
                '--allow-world-size-change', 'checkpoint-00007500'):
        assert arg in result.stdout
    result = subprocess.run(['bash', script, '4'], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and '0 through 3' in result.stderr


def test_new_world_rng_is_reproducible_and_rank_specific():
    saved = {'config': {'expected_world_size': 96}, 'step': 7500}
    cfg = {'seed': 42}
    with torch.random.fork_rng(devices=[]):
        rt = SimpleNamespace(world=32, rank=0, device=torch.device('cpu'))
        restore_rng(saved, cfg, rt)
        first = torch.rand(16)
        restore_rng(saved, cfg, rt)
        assert torch.equal(first, torch.rand(16))
        rt.rank = 1
        restore_rng(saved, cfg, rt)
        assert not torch.equal(first, torch.rand(16))


def test_two_rank_checkpoint_resumes_on_one_with_optimizer_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('WANDB_MODE', 'disabled')
    path, cfg = fixture(tmp_path)
    cfg.update(batch_size=1, accumulation_steps=1, language_batch_counts={'ar': 1},
               expected_world_size=2, acoustic_layers=2, decoder_layers=2,
               train_embeddings=True, logging_steps=1)
    path.write_text(yaml.safe_dump(cfg))
    context = mp.spawn(distributed_train_worker,
        args=(str(path), str(tmp_path / 'rendezvous'), False), nprocs=2, join=False)
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail('two-rank fixture timed out')
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    checkpoint = tmp_path / 'out/checkpoint-00000001'
    saved = torch.load(checkpoint / 'training_state.pt', weights_only=True)
    assert len(saved['rng']) == 2
    cfg.update(expected_world_size=1, batch_size=2, language_batch_counts={'ar': 2},
               output_dir=str(tmp_path / 'resumed'))
    path.write_text(yaml.safe_dump(cfg))
    observed = []
    original = torch.optim.AdamW.step
    def capture(opt, *args, **kwargs):
        observed.append(copy.deepcopy(opt.state_dict()))
        return original(opt, *args, **kwargs)
    monkeypatch.setattr(torch.optim.AdamW, 'step', capture)
    train(path, checkpoint, allow_batch_change=True, allow_world_size_change=True)
    assert len(observed) == 1
    for key, state in saved['optimizer']['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(observed[0]['state'][key][name], value, rtol=0, atol=0)
    for group in observed[0]['param_groups']:
        assert group['lr'] == group['base_lr'] * schedule(1, cfg)
    record = json.loads((tmp_path / 'resumed/resume.json').read_text())
    assert record['world_size_changed'] and not record['effective_batch_changed']
    assert record['global_batch'] == {'from': 2, 'to': 2}
    final = torch.load(tmp_path / 'resumed/final/training_state.pt', weights_only=True)
    assert final['step'] == 2 and len(final['rng']) == 1
    assert all(s['step'].item() == 2 for s in final['optimizer']['state'].values())
