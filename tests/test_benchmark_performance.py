"""Timing denominators, exact output binding, and absent instrumentation."""
import json

import pytest
from gqasr.arabic_benchmark.common import atomic_json, fingerprint
from data_processing.arabic_reference.benchmark_performance import read_performance
from data_processing.arabic_reference.core import sha256, write_jsonl


def fixture_run(root, *, measured=True, modes=('arabic', 'auto')):
    root.mkdir(exist_ok=True)
    identity = {'modes': list(modes), 'precision': 'bf16'}
    atomic_json(root / 'run.json', identity)
    rows = [{'id': str(i), 'audio_sha256': str(i)*64, 'mode': 'arabic', 'pred_text': 'مرحبا'} for i in range(2)]
    write_jsonl(root / 'predictions-arabic.jsonl', rows)
    outputs = [{**r, 'mode': mode} for mode in modes for r in rows]
    stats = {'inference_seconds': 6.0}
    if measured:
        stats['inference_batches'] = [{'ids': ['0','1'], 'mode': mode, 'seconds': 6 / len(modes)} for mode in modes]
    atomic_json(root/'tasks/0.json', {'run_id': fingerprint(identity), 'predictions': outputs, 'stats': stats})
    attempts = [{'wall_seconds': 5., 'gpus':['0'], 'workers_per_gpu':2, 'actual_workers':2, 'batch_size':2, 'workers':[{'device':'GPU'}]},
                {'wall_seconds': 3., 'gpus':['0'], 'workers_per_gpu':1, 'actual_workers':1, 'batch_size':1, 'workers':[{'device':'GPU'}]}]
    atomic_json(root/'scores.json', {'protocol': identity, 'performance': {'sum_worker_inference_seconds':6., 'total_attempt_wall_seconds':8., 'attempts':attempts, 'oom_batch_splits':1}})
    return sha256(root/'predictions-arabic.jsonl')


def test_timing_counts_outputs_across_modes_and_all_attempts(tmp_path):
    sha = fixture_run(tmp_path)
    result = read_performance(tmp_path, sha)
    assert result['outputs'] == 4 and result['clips'] == 2
    assert result['mean_batch_seconds'] == 3
    assert result['amortized_compute_seconds_per_output'] == 1.5
    assert result['outputs_per_wall_second'] == .5
    assert len(result['configurations']) == 2
    assert result['oom_batch_splits'] == 1


def test_legacy_time_is_not_claimed_as_request_or_batch_latency(tmp_path):
    sha = fixture_run(tmp_path, measured=False)
    result = read_performance(tmp_path, sha)
    assert result['mean_batch_seconds'] is None
    assert result['amortized_compute_seconds_per_output'] == 1.5
    assert result['batch_timing_status'] == 'not_recorded_for_every_task'


@pytest.mark.parametrize('corrupt', ['prediction', 'duplicate', 'missing', 'batch_membership', 'batch_seconds', 'report', 'negative', 'nan', 'identity'])
def test_rejects_unbound_or_inconsistent_timing(tmp_path, corrupt):
    sha = fixture_run(tmp_path)
    task = json.loads((tmp_path/'tasks/0.json').read_text())
    if corrupt == 'prediction': task['predictions'][0]['pred_text'] = 'changed'
    if corrupt == 'duplicate': task['predictions'].append(task['predictions'][0])
    if corrupt == 'missing': task['predictions'].pop()
    if corrupt == 'batch_membership': task['stats']['inference_batches'][0]['ids'] = ['0','0']
    if corrupt == 'batch_seconds': task['stats']['inference_batches'][0]['seconds'] = 100
    if corrupt == 'negative': task['stats']['inference_seconds'] = -1
    if corrupt == 'nan': task['stats']['inference_seconds'] = float('nan')
    if corrupt == 'identity': task['run_id'] = 'wrong'
    (tmp_path/'tasks/0.json').write_text(json.dumps(task))
    if corrupt == 'report':
        report = json.loads((tmp_path/'scores.json').read_text())
        report['performance']['sum_worker_inference_seconds'] = 20
        atomic_json(tmp_path/'scores.json', report)
    with pytest.raises(ValueError): read_performance(tmp_path, sha)


def test_missing_timing_is_explicit_and_prediction_hash_is_required(tmp_path):
    assert read_performance(tmp_path, 'irrelevant')['status'] == 'unavailable'
    fixture_run(tmp_path)
    with pytest.raises(ValueError, match='predictions differ'): read_performance(tmp_path, 'wrong')


def test_worker_records_each_synchronized_batch_and_mode(tmp_path, monkeypatch):
    import queue
    import threading
    import numpy as np
    from types import SimpleNamespace
    from gqasr.arabic_benchmark import worker
    class Recognizer:
        def __init__(self, *args):
            self.torch = SimpleNamespace(__version__='fixture')
        def features(self, rows): return {'features': np.ones((len(rows), 1))}
        def safe_infer(self, rows, features, mode, budget):
            return [{**r, 'mode': mode, 'pred_text': 'مرحبا'} for r in rows], 1
    monkeypatch.setattr(worker, 'Recognizer', Recognizer)
    # worker_main is normally isolated; restore its process-only settings after this test.
    for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','PYTHONUNBUFFERED'):
        monkeypatch.setenv(key, '')
    jobs, events = queue.Queue(), queue.Queue()
    rows = [{'id':str(i),'actual_duration':1} for i in range(3)]
    jobs.put({'rows':rows,'modes':['arabic','auto'],'modes_by_id':{r['id']:['arabic','auto'] for r in rows},
              'run_id':'test','task_id':'t','output':str(tmp_path/'task.json')})
    jobs.put(None)
    start = threading.Event(); start.set()
    worker.worker_main(0,'cpu','fixture',{'precision':'bf16','memory_fraction':1,'batch_size':2,'max_new_tokens':8},jobs,events,start,threading.Semaphore(1),tmp_path/'worker.log')
    record = json.loads((tmp_path/'task.json').read_text())
    batches = record['stats']['inference_batches']
    assert len(batches) == 4
    assert sum(len(b['ids']) for b in batches) == 6
    assert sum(b['seconds'] for b in batches) == record['stats']['inference_seconds']
    assert record['stats']['oom_batch_splits'] == 4
    assert {b['mode'] for b in batches} == {'arabic','auto'}
