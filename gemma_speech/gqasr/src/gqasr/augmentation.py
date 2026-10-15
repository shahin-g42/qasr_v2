"""Train-only QASR augmentations with GQASR target budgets and replayable RNG."""

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ._qasr.augmentation import build_augmenter
from ._qasr.code_switch import OnlineCodeSwitch


@contextmanager
def augmentation_rng(rows):
    """Stateless per-microbatch randomness, independent of worker/prefetch order."""
    if any("augmentation_seed" not in row for row in rows):
        raise ValueError("Training augmentation requires sampler-provided seeds")
    seed = int.from_bytes(
        hashlib.sha256(repr([r["augmentation_seed"] for r in rows]).encode()).digest()[:8], "big"
    )
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_training_transforms(processor, cfg):
    augmenter = build_augmenter(cfg.get("augmentation"), 16000)
    if augmenter and augmenter.noise_injection:
        noise = augmenter.noise_injection
        if noise.noise_dir and not noise._noise_files:
            raise ValueError(f"Configured noise directory has no usable audio: {noise.noise_dir}")
    switcher = None
    if cfg.get("online_code_switching"):
        # Pool text is always tokenized live; old tokenizer-specific pool eligibility
        # is retained, but old token counts are never used for GQASR budgets.
        switcher = OnlineCodeSwitch(cfg["code_switch_index"], processor.tokenizer, retokenize=True)
        excluded = cfg.get("code_switch_exclude_manifest")
        switcher._excluded_audio_names = set()
        if excluded:
            with Path(excluded).open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    audio = record.get("audio_filepath")
                    if not isinstance(audio, str) or not audio:
                        raise ValueError("Quarantine manifest requires audio_filepath on every row")
                    switcher._excluded_audio_names.add(Path(audio).name)
        if switcher.cfg["sample_rate"] != 16000:
            raise ValueError("Code-switch index must use 16 kHz audio")
        overhead = max(len(processor.target("", lang)) for lang in cfg["language_batch_counts"])
        budget = cfg["max_target_length"] - overhead
        if budget < 1:
            raise ValueError("Target budget cannot fit language header/EOS")
        switcher.cfg["max_target_tokens"] = budget
        switcher.cfg["max_seconds"] = min(switcher.cfg["max_seconds"], cfg["max_duration_seconds"])
    return augmenter, switcher


def smoke(config, model=None, batches=20):
    """Exercise real cluster audio/index/frontend without loading model weights."""
    from collections import Counter

    from .config import load_config
    from .data import Collator, ManifestIndex, StratifiedUpdateSampler, data_contract
    from .processing import GQASRProcessor

    if batches < 1:
        raise ValueError("batches must be positive")
    cfg = load_config(config)
    if model:
        cfg["base_model"] = model
    dataset = ManifestIndex(cfg["index_path"], "train")
    if dataset.metadata["data_contract"] != data_contract(cfg):
        raise ValueError("index/config mismatch; rebuild the manifest index")
    processor = GQASRProcessor(cfg["base_model"])
    fingerprint = hashlib.sha256(
        processor.tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest()
    if dataset.metadata["tokenizer_sha256"] != fingerprint:
        raise ValueError("tokenizer differs from manifest index")
    collator = Collator(processor, cfg, training=True)
    sampler = StratifiedUpdateSampler(
        dataset.counts,
        cfg["language_batch_counts"],
        cfg["batch_size"],
        batches,
        seed=cfg["data_seed"],
        augmentation_seeds=True,
    )
    counts = Counter()
    for i, keys in enumerate(sampler):
        if i >= batches:
            break
        batch, rows = collator([dataset[key] for key in keys])
        if not batch["input_features"].isfinite().all():
            raise ValueError("nonfinite augmented features")
        for row in rows:
            counts["examples"] += 1
            audit = row.get("augmentation_audit", {})
            for key in ("triggered", "applied", "waveform_changed", "specaugment_changed"):
                counts[key] += int(audit.get(key, False))
            if audit.get("reason"):
                counts["code_switch_" + audit["reason"]] += 1
        print(
            json.dumps({"event": "augmentation_smoke_progress", "batches": i + 1, **counts}),
            flush=True,
        )
    return {
        "kind": "audio augmentation integration smoke test, not speech quality evaluation",
        "batches": batches,
        "counts": dict(counts),
        "model_weights_loaded": False,
    }
