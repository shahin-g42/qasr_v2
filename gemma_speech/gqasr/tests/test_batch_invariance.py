import json
from argparse import Namespace

import torch
from gqasr.batch_invariance import configuration, difference, run
from test_training import fixture


def test_augmentation_isolation_does_not_mutate_saved_config():
    cfg = dict(online_code_switching=True, augmentation={'spec_augment': {'p': .37}, 'noise_injection': {'p': .37}})
    assert configuration(cfg,'batch_clean')['augmentation'] is None
    assert not configuration(cfg,'batch_clean')['online_code_switching']
    assert 'spec_augment' not in configuration(cfg,'waveform')['augmentation']
    assert set(configuration(cfg,'specaugment')['augmentation']) == {'spec_augment'}
    assert configuration(cfg,'code_switch')['online_code_switching']
    assert len(cfg['augmentation']) == 2
    assert difference(torch.ones(3),torch.ones(4))['shape_equal'] is False
    assert difference(torch.ones(3),torch.ones(3))['exact']


def test_full_trace_on_real_tiny_model(tmp_path, monkeypatch):
    import gqasr.batch_invariance as audit
    from gqasr.comparison_audit import pin_checkpoint
    from gqasr.data import ManifestIndex
    _, cfg = fixture(tmp_path)
    cfg.update(online_code_switching=False, augmentation=None, batch_size=2, language_batch_counts={'ar': 2})
    prior = tmp_path/'prior'
    prior.mkdir()
    pin_checkpoint(tmp_path/'base',prior/'snapshots/current')
    (prior/'training-config.json').write_text(json.dumps(cfg))
    (prior/'jobs.json').write_text(json.dumps([{'command':['--predictions','unused']}]))
    row = ManifestIndex(cfg['index_path'],'eval')[('ar',1)]
    sample = dict(id=row['id'],audio=row['audio'],reference=row['text'],duration=row['duration'])
    monkeypatch.setattr(audit,'select_samples',lambda *args:[sample])
    # Exercise the complete tracing/model path; augmentation isolation is tested separately.
    monkeypatch.setattr(audit,'configuration',lambda cfg,mode:cfg)
    class Dataset:
        counts = {'ar':1}
        def __init__(self,*args):pass
        def __getitem__(self,key):
            return {**row,'language':'zh','augmentation_seed':key[2]}
    monkeypatch.setattr(audit,'ManifestIndex',Dataset)
    run(Namespace(device='cpu',comparison=str(prior),samples=1,output=str(tmp_path/'result'),start=0,max_new_tokens=2))
    lines = [json.loads(line) for line in (tmp_path/'result/trace.jsonl').read_text().splitlines()]
    assert len(lines)==6
    assert lines[1]['batch_size']==2
    assert all(torch.isfinite(torch.tensor(r['teacher_forced']['transcript_nll'])) for r in lines)
    assert lines[1]['differences_from_clean_solo']['encoder']['shape_equal']
