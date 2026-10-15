"""Offline augmentation integration: real WAV/SQLite pools, no pretrained weights."""

import hashlib
import json
import random
import sqlite3

import numpy as np
import pytest
import soundfile as sf
import torch
from gqasr._qasr.augmentation import SpecAugment
from gqasr.augmentation import augmentation_rng, build_training_transforms
from gqasr.data import Collator, StratifiedUpdateSampler


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}


class Processor:
    tokenizer = Tokenizer()

    def target(self, text, lang):
        return [0, 1] + self.tokenizer(text)["input_ids"] + [2]

    def batch(self, waves, texts, languages, maximum):
        lengths = [len(w) // 160 for w in waves]
        mask = torch.arange(max(lengths))[None] < torch.tensor(lengths)[:, None]
        features = torch.ones(len(waves), max(lengths), 128) * mask[:, :, None]
        return {"input_features": features, "input_features_mask": mask}


def fixture(tmp_path):
    root = tmp_path / "cs"
    (root / "_state").mkdir(parents=True)
    pools = {}
    for lang in ["ar", "en"]:
        with sqlite3.connect(root / f"{lang}.sqlite") as db:
            db.execute("CREATE TABLE pool(id INTEGER PRIMARY KEY, record TEXT)")
            for i in range(2):
                audio = root / f"{lang}{i}.wav"
                sf.write(
                    audio,
                    np.sin(np.arange(8000) / 20).astype("float32") * 0.1,
                    16000,
                    subtype="FLOAT",
                )
                wave, _ = sf.read(audio, dtype="float32")
                r = dict(
                    audio_filepath=str(audio),
                    text=f"{lang}word",
                    language=lang,
                    frames=len(wave),
                    waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),
                )
                db.execute("INSERT INTO pool VALUES (?,?)", (i, json.dumps(r)))
        pools[lang] = {"size": (root / f"{lang}.sqlite").stat().st_size, "count": 2}
    config = dict(
        sample_rate=16000,
        seed=7,
        pair_weights={"ar-en": 1},
        segment_counts=[2],
        max_attempts=5,
        gap_seconds=0.1,
        min_source_seconds=0.1,
        max_seconds=3,
        max_target_tokens=512,
        language_policy="first",
    )
    plan = dict(version=2, root=str(root), config=config, preview_samples=2)
    plan_path = root / "_state/plan.json"
    plan_path.write_text(json.dumps(plan))
    index = {
        **plan,
        "pilot": False,
        "pair_targets": {"ar-en": 2},
        "pools": pools,
        "plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        "tokenizer_fingerprint": "different-source-tokenizer",
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index))
    cfg = dict(
        online_code_switching=True,
        code_switch_index=str(index_path),
        code_switch_probability=1.0,
        augmentation=False,
        language_batch_counts={"ar": 1, "en": 1},
        max_target_length=12,
        min_duration_seconds=0.1,
        max_duration_seconds=3.0,
    )
    row = dict(
        audio=str(root / "ar0.wav"), text="anchor", language="ar", augmentation_seed="7:0:0:0"
    )
    return cfg, row


def test_existing_pool_retokenization_mixing_and_clean_eval(tmp_path):
    cfg, row = fixture(tmp_path)
    train = Collator(Processor(), cfg, training=True)
    batch, rows = train([row])
    assert rows[0]["text"] == "anchor enword"
    assert rows[0]["augmentation_audit"]["applied"]
    assert batch["input_features_mask"].sum() == 110
    assert row["text"] == "anchor"  # caller's data untouched
    batch, rows = Collator(Processor(), cfg)([row])
    assert rows[0]["text"] == "anchor" and batch["input_features_mask"].sum() == 50
    assert "augmentation_audit" not in rows[0]


def test_target_budget_falls_back_without_cutting_audio(tmp_path):
    cfg, row = fixture(tmp_path)
    cfg["max_target_length"] = 4
    batch, rows = Collator(Processor(), cfg, training=True)([row])
    assert rows[0]["text"] == "anchor"
    assert not rows[0]["augmentation_audit"]["applied"]
    assert batch["input_features_mask"].sum() == 50


def test_pool_tampering_fails(tmp_path):
    cfg, row = fixture(tmp_path)
    collator = Collator(Processor(), cfg, training=True)
    for path in (tmp_path / "cs").glob("en*.wav"):
        sf.write(path, np.zeros(8000, dtype="float32"), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="source audio changed"):
        collator([row])


def test_specaugment_preserves_padding_and_masks_valid_frames():
    x = torch.ones(2, 100, 128)
    x[1, 40:] = 7  # sentinel padding
    mask = torch.arange(100)[None] < torch.tensor([100, 40])[:, None]
    aug = SpecAugment(p=1)
    with augmentation_rng([{"augmentation_seed": 3}]):
        y = aug(x, mask)
    assert torch.equal(x[1, 40:], y[1, 40:])
    assert not torch.equal(x, y)
    assert torch.all(x == torch.where(mask[:, :, None], 1.0, 7.0))


def test_replay_and_rng_isolation(tmp_path):
    cfg, row = fixture(tmp_path)
    cfg["augmentation"] = {
        "spec_augment": {"enabled": True, "p": 1},
        "speed_perturb": {"enabled": True, "p": 1, "rate_range": [0.85, 1.15]},
        "noise_injection": {"enabled": True, "p": 1},
        "codec_augment": {"enabled": True, "p": 1},
    }
    collator = Collator(Processor(), cfg, training=True)
    py_state = random.getstate()
    np_state = np.random.get_state()
    a, ar = collator([row])
    b, br = Collator(Processor(), cfg, training=True)([row])
    assert torch.equal(a["input_features"], b["input_features"]) and ar == br
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert ar[0]["augmentation_audit"]["waveform_changed"]


def test_resume_seeds_independent_of_worker_scheduling():
    args = ({"ar": 10, "en": 10}, {"ar": 2, "en": 2}, 2, 4)
    full = list(StratifiedUpdateSampler(*args, augmentation_seeds=True))
    resumed = list(StratifiedUpdateSampler(*args, start=2, augmentation_seeds=True))
    assert full[4:] == resumed
    other = list(StratifiedUpdateSampler(*args, rank=1, world=2, augmentation_seeds=True))
    assert other[0][0][2] != full[0][0][2]


def test_configured_missing_noise_fails(tmp_path):
    cfg, row = fixture(tmp_path)
    cfg["augmentation"] = {
        "noise_injection": {"enabled": True, "noise_dir": str(tmp_path / "missing")}
    }
    with pytest.raises(ValueError, match="noise directory"):
        build_training_transforms(Processor(), cfg)


def test_full_target_budget_rechecked_after_join(tmp_path):
    cfg, row = fixture(tmp_path)

    class ContextSensitiveProcessor(Processor):
        def target(self, text, lang):
            # Simulate contextual tokenization making the joined target longer
            # than the isolated transcript + empty-header estimate.
            return [0] * (20 if "enword" in text else 4)

    batch, rows = Collator(ContextSensitiveProcessor(), cfg, training=True)([row])
    assert rows[0]["text"] == "anchor"
    assert rows[0]["augmentation_audit"]["reason"] == "full_target_budget"
    assert batch["input_features_mask"].sum() == 50


def test_speed_does_not_truncate_overlong_audio(tmp_path):
    cfg, row = fixture(tmp_path)
    cfg.update(online_code_switching=False, max_duration_seconds=0.5)
    cfg["augmentation"] = {"speed_perturb": {"enabled": True, "p": 1, "rate_range": [0.5, 0.5]}}
    batch, rows = Collator(Processor(), cfg, training=True)([row])
    assert batch["input_features_mask"].sum() == 50
    assert not rows[0]["augmentation_audit"]["waveform_changed"]


def test_frozen_plan_modification_rejected(tmp_path):
    cfg, row = fixture(tmp_path)
    plan = tmp_path / "cs/_state/plan.json"
    plan.write_text(plan.read_text() + " ")
    with pytest.raises(ValueError, match="frozen plan"):
        Collator(Processor(), cfg, training=True)


def test_quarantined_audio_cannot_reenter_through_code_switch_pool(tmp_path):
    cfg, row = fixture(tmp_path)
    exclusion = tmp_path / "quarantine.jsonl"
    # Different mount root: basename identity must still exclude the recordings.
    exclusion.write_text(
        "".join(json.dumps({"audio_filepath": f"/other/mount/en{i}.wav"}) + "\n" for i in range(2))
    )
    cfg["code_switch_exclude_manifest"] = str(exclusion)
    batch, rows = Collator(Processor(), cfg, training=True)([row])
    assert rows[0]["text"] == "anchor"
    assert rows[0]["augmentation_audit"]["reason"] == "no_fitting_partner"
    assert batch["input_features_mask"].sum() == 50


def test_missing_quarantine_manifest_fails_closed(tmp_path):
    cfg, row = fixture(tmp_path)
    cfg["code_switch_exclude_manifest"] = str(tmp_path / "missing.jsonl")
    with pytest.raises(FileNotFoundError):
        Collator(Processor(), cfg, training=True)
