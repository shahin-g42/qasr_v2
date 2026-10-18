"""Curated anchors may reuse a larger immutable partner pool."""
from types import SimpleNamespace

import pytest

from qasr.config import TrainConfig
from qasr.train import _build_code_switcher


@pytest.mark.parametrize('case', ['subset', 'missing_anchor', 'changed_partner_source'])
def test_anchor_subset_retains_source_guards(tmp_path, monkeypatch, case):
    import qasr.code_switch

    sources = []
    for name in ('train_curated.jsonl', 'train_replay.jsonl'):
        path = tmp_path / name
        path.write_text('{}\n')
        stat = path.stat()
        sources.append({'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    mixer = SimpleNamespace(
        index={'sources': sources},
        cfg={'sample_rate': 16000, 'max_seconds': 35, 'max_target_tokens': 512, 'language_policy': 'first'},
    )
    monkeypatch.setattr(qasr.code_switch, 'OnlineCodeSwitch', lambda *a, **kw: mixer)
    index = tmp_path / 'index.json'
    index.write_text('{}')
    cfg = TrainConfig.from_yaml('configs/v8.1/03_hq_12node.yaml')
    cfg.code_switch_index = str(index)
    cfg.train_manifest = {'ar': [sources[0]['path']]}
    args = SimpleNamespace(should_save=False, output_dir=str(tmp_path / 'output'))
    processor = SimpleNamespace(tokenizer=None, feature_extractor=SimpleNamespace(sampling_rate=16000))
    if case == 'missing_anchor':
        cfg.train_manifest = {'ar': [str(tmp_path / 'train_unknown.jsonl')]}
        with pytest.raises(ValueError, match='absent from code-switch index'):
            _build_code_switcher(cfg, processor, args, [])
    elif case == 'changed_partner_source':
        (tmp_path / 'train_replay.jsonl').write_text('changed\n')
        with pytest.raises(ValueError, match='Indexed source manifest changed'):
            _build_code_switcher(cfg, processor, args, [])
    else:
        assert _build_code_switcher(cfg, processor, args, []) is mixer
