import json
from argparse import Namespace

import numpy as np
import pytest
import soundfile as sf
from gqasr.comparison_audit import paired, pin_checkpoint, report, save_rows, weight_audit
from gqasr.language_replay import run
from test_training import artifact


def test_paired_rejects_mismatch_and_counts_insertions():
    a = [dict(id='1', reference='中文', hypothesis='中文')]
    b = [dict(id='1', reference='中文', hypothesis='中文中文')]
    assert paired(a, b)['cer_delta_right_minus_left'] == 1
    with pytest.raises(ValueError, match='IDs'):
        paired(a, [{**b[0], 'id': '2'}])
    with pytest.raises(ValueError, match='Reference'):
        paired(a, [{**b[0], 'reference': '错'}])


@pytest.mark.parametrize('batch_size,control,swap', [(1, 'real', False), (2, 'real', True), (2, 'shift', False)])
def test_replay_real_tiny_model_with_teacher_forcing(tmp_path, batch_size, control, swap):
    artifact(tmp_path / 'base')
    pin_checkpoint(tmp_path / 'base', tmp_path / 'copy')
    drift = weight_audit(tmp_path / 'base', tmp_path / 'copy')
    assert all(v['changed'] == 0 for v in drift.values())
    rows = []
    for i, length in enumerate([4800, 6400]):
        audio = tmp_path / f'{i}.wav'
        sf.write(audio, np.sin(np.arange(length)/10).astype('float32')*.1, 16000)
        rows.append(dict(id=f'eval:zh:{i}', reference='中文', language='zh', audio=str(audio), audit_group='eval'))
    save_rows(tmp_path / 'samples.jsonl', rows)
    (tmp_path / 'run.json').write_text('{}')
    run(Namespace(run_config=str(tmp_path / 'run.json'), samples=str(tmp_path / 'samples.jsonl'),
                  checkpoint=str(tmp_path / 'base'), output=str(tmp_path / 'current'), device='cpu',
                  predictions=None, bridge_from=str(tmp_path / 'copy') if swap else None,
                  decoder_from=str(tmp_path / 'copy') if swap else None, batch_size=batch_size,
                  precision='fp32', audio_control=control, teacher_forced=True, max_new_tokens=3))
    actual = [json.loads(s) for s in (tmp_path / 'current/predictions-auto.jsonl').read_text().splitlines()]
    assert len(actual) == 2
    assert all(np.isfinite(r['teacher_forced']['transcript_nll']) for r in actual)
    assert (actual[0]['audio_source_id'] == rows[1]['id']) == (control == 'shift')
    report([dict(name='current', output=str(tmp_path / 'current'))], tmp_path)
    assert (tmp_path / 'REPORT.md').exists()
    assert (tmp_path / 'current/teacher-summary.json').exists()


def test_pinned_checkpoint_survives_source_removal(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    (source / 'training_state.pt').write_bytes(b'optimizer')
    target = tmp_path / 'snapshots/test'
    pin_checkpoint(source, target)
    (source / 'model.safetensors').unlink()
    assert (target / 'model.safetensors').read_bytes() == b'weights'
    assert not (target / 'training_state.pt').exists()


def test_worker_failure_is_not_reported_as_success(tmp_path):
    import sys

    from gqasr.comparison_audit import execute
    jobs = [dict(name='broken', output=str(tmp_path / 'broken'),
                 command=[sys.executable, '-c', 'raise RuntimeError("test failure")'])]
    with pytest.raises(RuntimeError, match='broken failed'):
        execute(jobs, ['0'], tmp_path)
    assert 'test failure' in (tmp_path / 'broken.log').read_text()
