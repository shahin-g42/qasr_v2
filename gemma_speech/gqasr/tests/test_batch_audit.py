from collections import Counter

import pytest
from gqasr.batch_audit import audit_rank
from gqasr.data import StratifiedUpdateSampler
from test_training import fixture


def test_96_rank_quotas_resume_and_seeds():
    quotas = dict(ar=20, en=12, hi=4, ml=2, zh=2)
    counts = {k: 100000 for k in quotas}
    seeds = set()
    for rank in range(96):
        full = list(StratifiedUpdateSampler(counts, quotas, 40, 4, rank, 96, 42, augmentation_seeds=True))
        resumed = list(StratifiedUpdateSampler(counts, quotas, 40, 4, rank, 96, 42, start=2, augmentation_seeds=True))
        assert resumed == full[2:]
        for batch in full:
            assert Counter(k[0] for k in batch) == quotas
            for _, ordinal, seed in batch:
                assert 1 <= ordinal <= 100000
                assert seed not in seeds
                seeds.add(seed)


@pytest.mark.parametrize('workers', [0, 2])
def test_actual_index_audio_collator_and_worker_order(tmp_path, workers):
    _, cfg = fixture(tmp_path)
    cfg.update(online_code_switching=False, augmentation=None)
    out = tmp_path/'audit'
    out.mkdir()
    result = audit_rank((cfg, 0, 0, 2, workers, str(out)))
    assert result['anchors'] == {'ar': 4}
    assert result['assigned'] == {'ar': 4}


def test_fusion_keeps_each_rows_targets_and_audio_together():
    import torch
    from test_training import tiny_model
    model = tiny_model()
    audio = torch.randn(3, 7, 32)
    valid = torch.arange(7)[None] < torch.tensor([7, 3, 5])[:, None]
    targets = torch.tensor([[3, 4, 5, 1], [6, 7, 1, 0], [8, 9, 10, 1]])
    mask = targets != 0
    embeds, attention, labels = model.fuse(audio, valid, targets, mask)
    for i in range(3):
        prefix = len(model.gqasr_prompt_prefix)
        length = int(valid[i].sum())
        assert torch.equal(embeds[i, prefix:prefix+length], audio[i, :length])
        assert torch.equal(labels[i][labels[i] != -100], targets[i][mask[i]])
        assert torch.equal(labels[:, 1:][i][labels[:, 1:][i] != -100], targets[i][mask[i]])
        assert (labels[i][~attention[i]] == -100).all()
