"""Inherit data, not machine-specific optimizer budgets, from a QASR stage recipe."""

import copy
import hashlib
import os
from dataclasses import fields
from pathlib import Path

import yaml
from qasr.config import TrainConfig

DATA_KEYS = {
    "train_manifest",
    "eval_manifest",
    "non_speech_manifest",
    "audio_root",
    "language",
    "punctuation",
    "min_duration_seconds",
    "max_duration_seconds",
    "max_target_length",
    "truncate_long_transcripts",
    "balanced_sampling",
    "sampling_strategy",
    "sampling_temperature",
    "language_batch_counts",
    "augmentation",
    "code_switch_index",
    "code_switch_probability",
    "eval_wer_global_language_budget",
}


def load_recipe(path, model=None, output=None, smoke=False, resume=None):
    path = Path(path).resolve()
    recipe = yaml.safe_load(path.read_text())
    allowed = {"data_recipe", "stage", "training", "robustness"}
    if not isinstance(recipe, dict) or set(recipe) - allowed:
        raise ValueError(f"AUD recipe accepts only {sorted(allowed)}")
    data_path = (path.parent / recipe["data_recipe"]).resolve()
    source = yaml.safe_load(data_path.read_text())
    stage = recipe.get("stage", "full")
    if stage not in ("adapter", "full", "decoder"):
        raise ValueError("Unknown training stage")
    values = {k: copy.deepcopy(v) for k, v in source.items() if k in DATA_KEYS}
    overrides = recipe.get("training", {})
    unknown = set(overrides) - {f.name for f in fields(TrainConfig)}
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    overrides = copy.deepcopy(overrides)
    if isinstance(overrides.get("augmentation"), dict) and isinstance(
        values.get("augmentation"), dict
    ):
        merged = copy.deepcopy(values["augmentation"])
        for key, value in overrides["augmentation"].items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
        overrides["augmentation"] = merged
    values.update(overrides)
    values.update(freeze_audio_tower=stage != "full", freeze_language_model=stage == "adapter")
    if model:
        values["model_name_or_path"] = str(Path(model).resolve())
    if output:
        values["output_dir"] = str(Path(output).resolve())
    if not values.get("model_name_or_path") or not values.get("output_dir"):
        raise ValueError("Supply --model and --output, or specify both in training options")
    if smoke:
        values["smoke_test"] = True
    if resume:
        values["resume_from_checkpoint"] = resume
    config = TrainConfig(**values)
    if config.ctc_loss_weight != 0:
        raise ValueError(
            "AUD preserves old CTC weights but currently trains the causal ASR loss only"
        )
    if config.fp16 or config.torch_compile:
        raise ValueError("AUD training currently supports bf16/fp32 without torch_compile")
    # QASR validates quotas per batch; AUD applies them across an accumulation window.
    virtual_batch = (
        sum(config.language_batch_counts.values())
        if config.language_batch_counts
        else config.per_device_train_batch_size
    )
    microbatch = config.per_device_train_batch_size
    if microbatch < 1 or config.gradient_accumulation_steps < 1:
        raise ValueError("Microbatch size and gradient accumulation must be positive")
    if virtual_batch % microbatch:
        raise ValueError("Microbatch size must divide the language quota window")
    if (microbatch * config.gradient_accumulation_steps) % virtual_batch:
        raise ValueError("An optimizer update must contain whole language quota windows")
    validation = copy.deepcopy(config)
    validation.per_device_train_batch_size = virtual_batch
    validation.validate()
    if config.deepspeed:
        config.deepspeed = str((path.parent / config.deepspeed).resolve())
    receipt = {
        "recipe": str(path),
        "recipe_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "data_recipe": str(data_path),
        "data_recipe_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "stage": stage,
        "quota_window": virtual_batch,
        "resolved": config.to_dict(),
    }
    robustness = copy.deepcopy(recipe.get("robustness") or {})
    if robustness:
        if set(robustness) - {"augmentation", "non_speech"}:
            raise ValueError("Unknown robustness settings")
        if robustness.get("non_speech") and config.non_speech_manifest_specs:
            raise ValueError("Use either controlled non-speech replacement or legacy manifests")
        assets = []
        ns = robustness.get("non_speech", {})
        ns["manifests"] = [os.path.expandvars(str(p)) for p in ns.get("manifests", [])]
        assets.extend(ns["manifests"])
        for settings in robustness.get("augmentation", {}).get("effects", {}).values():
            if "manifest" in settings:
                settings["manifest"] = os.path.expandvars(settings["manifest"])
                assets.append(settings["manifest"])
        receipt["robustness"] = robustness
        receipt["robustness_assets"] = {}
        for asset in assets:
            asset = Path(asset).expanduser().resolve()
            receipt["robustness_assets"][str(asset)] = hashlib.sha256(
                asset.read_bytes()
            ).hexdigest()
    config.aud_robustness = robustness
    return config, stage, receipt
