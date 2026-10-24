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
    allowed = {"data_recipe", "stage", "training", "robustness", "evaluation", "efficiency", "ctc"}
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
    evaluation = recipe.get("evaluation", {})
    if not isinstance(evaluation, dict) or set(evaluation) - {"exclude_languages"}:
        raise ValueError("evaluation accepts only exclude_languages")
    excluded = evaluation.get("exclude_languages", [])
    if not isinstance(excluded, list) or any(not isinstance(lang, str) for lang in excluded):
        raise ValueError("evaluation.exclude_languages must be a list of language codes")
    if excluded:
        manifests = values.get("eval_manifest")
        if not isinstance(manifests, dict) or set(excluded) - set(manifests):
            raise ValueError("Excluded evaluation languages must exist in eval_manifest")
        values["eval_manifest"] = {k: v for k, v in manifests.items() if k not in excluded}
        if not values["eval_manifest"]:
            raise ValueError("Keep at least one evaluation language")
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
    ctc = recipe.get("ctc", {})
    if not isinstance(ctc, dict) or set(ctc) - {"warmup_steps"}:
        raise ValueError("ctc accepts only warmup_steps")
    warmup = ctc.get("warmup_steps", 100)
    if type(warmup) is not int or warmup < 0:
        raise ValueError("ctc.warmup_steps must be a nonnegative integer")
    config.aud_ctc = dict(warmup_steps=warmup)
    if not 0 <= config.ctc_loss_weight <= 1:
        raise ValueError("ctc_loss_weight must be in [0, 1]")
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
    if virtual_batch % microbatch and microbatch % virtual_batch:
        raise ValueError(
            "Microbatch size must divide or be a multiple of the language quota window"
        )
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
    if config.ctc_loss_weight:
        receipt["ctc"] = config.aud_ctc
    robustness = copy.deepcopy(recipe.get("robustness", {}))
    if robustness is None:
        robustness = {}
    if not isinstance(robustness, dict):
        raise ValueError("robustness must be a mapping")
    for key in ("augmentation", "non_speech"):
        if key in robustness and robustness[key] is None:
            robustness.pop(key)
    if robustness:
        if set(robustness) - {"augmentation", "non_speech"}:
            raise ValueError("Unknown robustness settings")
        if robustness.get("non_speech") and config.non_speech_manifest_specs:
            raise ValueError("Use either controlled non-speech replacement or legacy manifests")
        assets = []

        def asset_path(value):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Robustness asset paths must be nonempty strings")
            expanded = os.path.expandvars(value)
            if "$" in expanded:
                raise ValueError(f"Unset environment variable in robustness path: {value}")
            asset = Path(expanded).expanduser()
            asset = asset if asset.is_absolute() else path.parent / asset
            return str(asset.resolve())

        ns = robustness.get("non_speech", {})
        if not isinstance(ns, dict) or not isinstance(ns.get("manifests", []), list):
            raise ValueError("non_speech requires a mapping with a list of manifests")
        ns["manifests"] = [asset_path(p) for p in ns.get("manifests", [])]
        assets.extend(ns["manifests"])
        augmentation = robustness.get("augmentation", {})
        if not isinstance(augmentation, dict) or not isinstance(
            augmentation.get("effects", {}), dict
        ):
            raise ValueError("augmentation and effects must be mappings")
        for settings in augmentation.get("effects", {}).values():
            if not isinstance(settings, dict):
                raise ValueError("Each augmentation effect must be a mapping")
            if "manifest" in settings:
                settings["manifest"] = asset_path(settings["manifest"])
                assets.append(settings["manifest"])
        receipt["robustness"] = robustness
        receipt["robustness_version"] = 5
        receipt["robustness_assets"] = {}
        for asset in assets:
            asset = Path(asset).expanduser().resolve()
            receipt["robustness_assets"][str(asset)] = hashlib.sha256(
                asset.read_bytes()
            ).hexdigest()
    config.aud_robustness = robustness
    from .efficiency import resolve_efficiency

    config.aud_efficiency = resolve_efficiency(recipe.get("efficiency"), path.parent)
    if recipe.get("efficiency"):
        receipt["efficiency"] = config.aud_efficiency
    return config, stage, receipt
