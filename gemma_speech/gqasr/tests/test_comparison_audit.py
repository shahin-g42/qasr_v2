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


def test_shards_preserve_batch_boundaries_and_merge_order(tmp_path):
    from gqasr.comparison_audit import merge_shards, shard_jobs
    rows = [dict(id=str(i), reference='中文', audit_group='eval', duration=i+1,
                 audit_shift_id=str((i+1)%19), audit_shift_audio=f'{(i+1)%19}.wav') for i in range(19)]
    jobs = [dict(name='current-shifted-audio', output=str(tmp_path/'merged'),
                 command=['worker', '--samples', 'original', '--output', 'original'])]
    tasks = shard_jobs(jobs, rows, tmp_path, 8)
    assert len(tasks) == 5
    covered = []
    for task in tasks:
        ids = task['sample_ids']
        assert int(ids[0]) % 4 == 0
        assert len(ids) % 4 == 0 or ids[-1] == '18'
        covered += ids
        directory = __import__('pathlib').Path(task['output'])
        directory.mkdir()
        predictions = [{**rows[int(i)], 'hypothesis': '中文',
                        'audio_source_id': rows[int(i)]['audit_shift_id']} for i in ids]
        for mode in ('auto', 'chinese'):
            save_rows(directory/f'predictions-{mode}.jsonl', predictions)
    assert set(covered) == {r['id'] for r in rows}
    merge_shards(jobs, tasks, rows)
    result = [json.loads(line) for line in (tmp_path/'merged/predictions-auto.jsonl').read_text().splitlines()]
    assert [r['id'] for r in result] == [r['id'] for r in rows]
    assert result[-1]['audio_source_id'] == '0'


def test_merge_rejects_incomplete_shard(tmp_path):
    from pathlib import Path

    from gqasr.comparison_audit import merge_shards, shard_jobs
    rows = [dict(id='1', reference='中文', audit_group='eval')]
    jobs = [dict(name='current', output=str(tmp_path/'merged'),
                 command=['worker', '--samples', 'original', '--output', 'original'])]
    tasks = shard_jobs(jobs, rows, tmp_path, 8)
    Path(tasks[0]['output']).mkdir()
    save_rows(Path(tasks[0]['output'])/'predictions-auto.jsonl', [])
    with pytest.raises(ValueError, match='Missing/reordered'):
        merge_shards(jobs, tasks, rows)
