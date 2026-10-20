import json
import torch.distributed as dist
from gqasr.rank_status import RankStatus, summarize


def test_summary_separates_missing_joined_and_waiting():
    records = [
        dict(rank=0, host='master', phase='loss_finiteness_collective', reached=['joined', 'first_forward']),
        dict(rank=1, host='other', phase='worker_startup', reached=['joined']),
    ]
    result = summarize(records, 3)
    assert result['joined'] == 2 and result['not_reported'] == [2]
    assert result['first_batch_ready'] == 1 and result['first_update_complete'] == 0
    assert result['phases'] == {'loss_finiteness_collective': 1, 'worker_startup': 1}
    assert result['pending'][1]['host'] == 'other'


def test_observer_finishes_when_all_updates_reported(monkeypatch, capsys):
    monkeypatch.setenv('RANK', '0')
    monkeypatch.setenv('WORLD_SIZE', '2')
    monkeypatch.setenv('GQASR_RANK_STATUS', '1')
    status = RankStatus()
    status.mark('joined')
    status.mark('first_forward')
    status.mark('first_update_complete')
    record = status.record
    status.mark('backward')
    assert status.record is record
    backing = dist.HashStore()
    store = dist.PrefixStore("gqasr-observed-startup/", backing)
    store.set('1', json.dumps({**record, 'rank': 1}))
    monkeypatch.setattr(dist.distributed_c10d, '_get_default_store', lambda: backing)
    status.attach()
    status.thread.join(timeout=3)
    assert not status.thread.is_alive()
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary['joined'] == summary['first_update_complete'] == 2
    assert summary['not_reported'] == [] and summary['pending'] == []


def test_observation_failure_does_not_raise(monkeypatch, capsys):
    monkeypatch.setenv('GQASR_RANK_STATUS', '1')
    class BrokenStore:
        def set(self, *args):
            raise RuntimeError('offline')
    status = RankStatus()
    status._observe(BrokenStore())
    assert 'status stopped: RuntimeError' in capsys.readouterr().out


def test_disabled_status_is_silent(monkeypatch, capsys):
    monkeypatch.setenv('GQASR_RANK_STATUS', '0')
    status = RankStatus()
    status.mark('joining')
    status.attach()
    status.close()
    assert status.thread is None and not capsys.readouterr().out
