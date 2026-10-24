import json
import pickle
from collections import Counter
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from qasr.data import JsonlSpeechDataset, ManifestError
from qasr.sampling import WeightedStratifiedLanguageSampler

from audm.efficiency import resolve_efficiency
from audm.health import inspect_gpus
from audm.indexed_data import IndexedSpeechDataset
from audm.sampling import UpdateDurationSampler


def test_cache_matches_uncached_and_invalidates(tmp_path):
    manifest = tmp_path / "data.jsonl"
    rows = [dict(audio_filepath="a.wav", text="hello", duration=d) for d in (1, 5, 30, 31, None)]
    rows += [dict(audio_filepath="b.wav", text="!!!", duration=2)]
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    opts = dict(min_duration_seconds=1, max_duration_seconds=30, language="en")
    plain = JsonlSpeechDataset(manifest, **opts)
    cached = IndexedSpeechDataset(manifest, cache_directory=tmp_path / "cache", **opts)
    assert asdict(plain.stats) == asdict(cached.stats)
    assert list(plain) == list(cached)
    assert isinstance(cached._offsets, np.memmap)
    cloned = pickle.loads(pickle.dumps(cached))
    assert isinstance(cloned._offsets, np.memmap)
    assert list(cloned) == list(plain)
    with manifest.open("a") as stream:
        stream.write("\n" + json.dumps(rows[0]))
    new = IndexedSpeechDataset(manifest, cache_directory=tmp_path / "cache", **opts)
    assert len(new) == len(cached) + 1
    assert new.index_directory != cached.index_directory
    manifest.write_text('{"text": "oops')
    with pytest.raises(ManifestError):
        IndexedSpeechDataset(manifest, cache_directory=tmp_path / "cache", **opts)


@pytest.mark.parametrize("micro,accumulation", [(2, 8), (4, 4), (8, 2)])
@pytest.mark.parametrize("world", [1, 3])
def test_bucketing_preserves_each_update_quotas_and_coverage(micro, accumulation, world):
    class Part:
        def __init__(self, lang, n):
            self.language = lang
            self.durations = np.random.default_rng(n).uniform(1, 30, n)

        def __len__(self):
            return len(self.durations)

    parts = [Part("en", 39), Part("ml", 17)]
    quotas = {"en": 3, "ml": 1}
    for rank in range(world):
        sampler = WeightedStratifiedLanguageSampler(
            parts,
            batch_size=4,
            seed=42,
            language_batch_counts=quotas,
            num_replicas=world,
            rank=rank,
        )
        wrapped = UpdateDurationSampler(sampler, parts, micro, accumulation)
        before, after = list(sampler), list(wrapped)
        assert len(before) == len(after)
        assert after == list(wrapped)
        for start in range(0, len(before), 16):
            assert sorted(before[start : start + 16]) == sorted(after[start : start + 16])
        block = max(4, micro)
        for start in range(0, len(after) - block + 1, block):
            assert Counter("en" if i < 39 else "ml" for i in after[start : start + block]) == {
                "en": 3 * block // 4,
                "ml": block // 4,
            }
        wrapped.set_epoch(1)
        assert list(wrapped) != after


def test_efficiency_validation_and_thermal_flags(tmp_path):
    for value in (
        {"loss_chunk_size": 0},
        {"prefetch_factor": -1},
        {"encoder_checkpoint_every": True},
        {"distributed_wer": 1},
    ):
        with pytest.raises(ValueError):
            resolve_efficiency(value, tmp_path)
    xml = "<nvidia_smi_log><gpu><uuid>one</uuid><clocks_event_reasons><clocks_event_reason_sw_thermal_slowdown>Active</clocks_event_reason_sw_thermal_slowdown></clocks_event_reasons><processes><process_info /></processes></gpu></nvidia_smi_log>"
    assert inspect_gpus(xml)[0]["thermal_slowdown"]
    assert inspect_gpus(xml)[0]["processes"] == 1


def test_distributed_wer_partition_matches_single_rank(monkeypatch):
    from audm.eval_callback import AUDWEREvalCallback

    callback = AUDWEREvalCallback(
        processor=SimpleNamespace(_waves=lambda waves, sr: waves),
        eval_manifest_specs=[],
        samples_per_language=99,
        batch_size=40,
    )
    rows = [dict(audio_path=str(i), text="a long example", language="en") for i in range(97)]
    callback._dataset_for = lambda *args: rows
    monkeypatch.setattr("audm.eval_callback.load_mono_audio", lambda *args: np.zeros(1600))
    callback._transcribe_batch = lambda trainer, features, waves: ["a example" for f in features]
    expected = callback._local_totals(None, "en", [("x", "en")])
    parts = []
    callback.decode_world = 8
    for rank in range(8):
        callback.decode_rank = rank
        parts.append(callback._local_totals(None, "en", [("x", "en")]))
    assert np.array(parts).sum(0).tolist() == expected


def test_category_metrics_count_tokens_without_reweighting():
    import math

    import torch

    from audm.losses import audio_category_metrics

    head = torch.nn.Linear(5, 16, bias=False)
    torch.nn.init.zeros_(head.weight)
    hidden = torch.ones(4, 6, 5)
    labels = torch.tensor(
        [
            [-100, -100, 1, 9, 8, 7],
            [-100, 2, 7, -100, -100, -100],
            [-100, -100, -100, 3, 7, -100],
            [-100, 4, 7, -100, -100, -100],
        ]
    )
    before = labels.clone()
    values = audio_category_metrics(
        hidden, head, labels, dict(speech=1, music=2, noise=3, silence=4)
    )
    assert values[:, 0].tolist() == [1, 1, 1, 1]
    assert values[:, 3].tolist() == [4, 2, 2, 2]
    torch.testing.assert_close(values[:, 1], torch.full((4,), math.log(16)))
    assert torch.equal(before, labels)
    assert not values.requires_grad
    assert not audio_category_metrics(
        hidden[:, :1], head, labels[:, :1], dict(speech=1, music=2, noise=3, silence=4)
    ).any()
