import json
from pathlib import Path
import pytest
from gqasr import env_check


def test_new_runtime_requires_matching_passed_profile(tmp_path, monkeypatch):
    runtime = {'torch': '2.13.0+cu129', 'transformers': '5.20.2', 'cuda_runtime': '12.9'}
    monkeypatch.setattr(env_check.sys, 'prefix', str(tmp_path))
    monkeypatch.setattr(env_check, 'runtime_versions', lambda: runtime)
    with pytest.raises(RuntimeError, match='Unvalidated'):
        env_check.check_training_profile()
    path = tmp_path / 'gqasr-training-profile.json'
    path.write_text(json.dumps({'passed': True, 'runtime': runtime}))
    assert env_check.check_training_profile() == runtime
    path.write_text(json.dumps({'passed': True, 'runtime': {**runtime, 'torch': 'old'}}))
    with pytest.raises(RuntimeError, match='changed'):
        env_check.check_training_profile()


def test_original_runtime_remains_accepted_without_profile(tmp_path, monkeypatch):
    monkeypatch.setattr(env_check.sys, 'prefix', str(tmp_path))
    monkeypatch.setattr(env_check, 'runtime_versions', lambda: {
        'torch': '2.11.0+cu128', 'transformers': '5.19.0', 'cuda_runtime': '12.8'})
    assert env_check.check_training_profile()['torch'] == '2.11.0+cu128'


def test_setup_refuses_existing_environment(tmp_path):
    import os
    import subprocess
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(['bash', str(root / 'gemma_speech/cluster/setup_training_env2.sh')],
        env={**os.environ, 'GQASR_TRAIN_ENV': str(tmp_path)}, capture_output=True, text=True)
    assert result.returncode == 2 and 'Refusing to overwrite' in result.stderr
    assert not list(tmp_path.iterdir())
