from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ManifestPaths = str | list[str]
ManifestConfig = ManifestPaths | dict[str, ManifestPaths]


def _expand_manifest_config(
    value: ManifestConfig | None,
    *,
    default_language: str,
    field_name: str,
    required: bool,
) -> list[tuple[str, str]]:
    if value is None:
        if required:
            raise ValueError(f"{field_name} must be set in the YAML file or on the command line")
        return []

    groups = value.items() if isinstance(value, dict) else [(default_language, value)]
    specs: list[tuple[str, str]] = []
    for language, paths_value in groups:
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"{field_name} language keys must be non-empty strings")
        if isinstance(paths_value, str):
            paths = [paths_value]
        elif isinstance(paths_value, list):
            paths = paths_value
        else:
            raise ValueError(
                f"{field_name}[{language!r}] must be a manifest path or a list of paths"
            )
        if not paths or any(not isinstance(path, str) or not path.strip() for path in paths):
            raise ValueError(f"{field_name}[{language!r}] must contain non-empty manifest paths")
        specs.extend((path, language) for path in paths)

    if required and not specs:
        raise ValueError(f"{field_name} must contain at least one manifest")
    return specs


@dataclass
class TrainConfig:
    model_name_or_path: str = (
        "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/initial"
    )
    train_manifest: ManifestConfig = ""
    eval_manifest: ManifestConfig | None = None
    audio_root: str | None = None
    output_dir: str = "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/run"

    language: str = "ar"
    punctuation: bool = True
    min_duration_seconds: float = 0.1
    max_duration_seconds: float = 30.0
    max_target_length: int = 512
    truncate_long_transcripts: bool = False
    validate_audio_paths: bool = False
    eval_split_ratio: float = 0.0
    eval_log_samples: int = 10
    eval_generation_max_new_tokens: int = 256

    smoke_test: bool = False
    smoke_test_max_steps: int = 2
    smoke_test_train_samples: int = 256
    smoke_test_eval_samples: int = 16

    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_steps: int = 90
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0

    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 3

    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True

    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    tf32: bool = True
    torch_compile: bool = False
    optim: str = "adamw_torch"

    deepspeed: str | None = None
    ddp_find_unused_parameters: bool = False
    report_to: str | list[str] = "wandb"
    wandb_project: str = "qasr"
    run_name: str | None = None
    seed: int = 42
    data_seed: int = 42

    resume_from_checkpoint: str | None = None
    attn_implementation: str | None = None
    freeze_audio_tower: bool = True
    freeze_language_model: bool = True

    # --- Phase A accuracy recipe (accuracy-focused defaults) ---
    # Language-balanced sampling: per-language draw weight proportional to
    # hours^sampling_temperature (Canary alpha/beta recipe). Keeps small
    # corpora (ml/hi) from being starved by naive concatenation.
    balanced_sampling: bool = True
    sampling_temperature: float = 0.5
    # "balanced"   -> LanguageBalancedSampler: hours^temperature, WITH
    #                 replacement. Cheap, but one epoch of steps leaves much of
    #                 the dominant language unseen.
    # "stratified" -> StratifiedLanguageSampler: every mini-batch holds an equal
    #                 share of each language AND every record of every language
    #                 is seen at least once per epoch (smaller languages repeat).
    #                 Requires per_device_train_batch_size % num_languages == 0.
    sampling_strategy: str = "balanced"
    # Per-group learning rates: encoder+projector at `learning_rate`, the
    # pretrained LLM body at `learning_rate * llm_lr_factor`, and the tied
    # token embeddings at `learning_rate * embed_lr_factor` (drift control).
    llm_lr_factor: float = 0.5
    embed_lr_factor: float = 0.25
    # Label smoothing for the causal LM loss (implemented via a custom
    # loss_function; transformers 5.14 does not smooth ForCausalLMLoss).
    label_smoothing: float = 0.1
    # Extra scheduler kwargs forwarded to TrainingArguments (e.g. WSD:
    # {"num_decay_steps": 20000, "min_lr_ratio": 0.1}).
    lr_scheduler_kwargs: dict[str, Any] | None = None
    # Non-speech / silence manifests for anti-hallucination training. Records
    # may carry blank transcripts (empty_target_ok); they are mixed into the
    # balanced sampler under their configured language tags.
    non_speech_manifest: ManifestConfig | None = None
    # CTC auxiliary loss weight on the encoder output (0 disables the head).
    # Training-time scaffold only; the head is discarded for inference.
    ctc_loss_weight: float = 0.0
    # Per-language WER/CER eval callback.
    eval_wer_samples_per_language: int = 100

    # Augmentation configuration. Each sub-key maps to an augmentation type.
    # All augmentations are disabled by default; enable them in the YAML config.
    augmentation: dict[str, Any] | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Configuration must be a YAML mapping: {config_path}")

        valid_keys = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid_keys)
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {', '.join(unknown)}")
        return cls(**values)

    def validate(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty fallback language code")
        _expand_manifest_config(
            self.train_manifest,
            default_language=self.language,
            field_name="train_manifest",
            required=True,
        )
        _expand_manifest_config(
            self.eval_manifest,
            default_language=self.language,
            field_name="eval_manifest",
            required=False,
        )
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.min_duration_seconds < 0:
            raise ValueError("min_duration_seconds must be non-negative")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError("max_duration_seconds must be greater than min_duration_seconds")
        if self.max_target_length < 2:
            raise ValueError("max_target_length must be at least 2")
        if not 0 <= self.eval_split_ratio < 1:
            raise ValueError("eval_split_ratio must be in the range [0, 1)")
        if self.eval_manifest_specs and self.eval_split_ratio:
            raise ValueError("Set either eval_manifest or eval_split_ratio, not both")
        if self.eval_log_samples < 0:
            raise ValueError("eval_log_samples must be non-negative")
        if self.eval_generation_max_new_tokens < 1:
            raise ValueError("eval_generation_max_new_tokens must be positive")
        if not isinstance(self.wandb_project, str) or not self.wandb_project.strip():
            raise ValueError("wandb_project must be a non-empty string")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        if self.per_device_train_batch_size < 1 or self.per_device_eval_batch_size < 1:
            raise ValueError("per-device batch sizes must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.smoke_test_max_steps < 1:
            raise ValueError("smoke_test_max_steps must be positive")
        if self.smoke_test_train_samples < 1:
            raise ValueError("smoke_test_train_samples must be positive")
        if self.smoke_test_eval_samples < 0:
            raise ValueError("smoke_test_eval_samples must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.sampling_strategy not in ("balanced", "stratified"):
            raise ValueError(
                "sampling_strategy must be 'balanced' or 'stratified', "
                f"got {self.sampling_strategy!r}"
            )
        if self.balanced_sampling and self.eval_split_ratio:
            # Both samplers index the UNSPLIT combined dataset; an eval split
            # shrinks the train dataset underneath them -> wrong-record fetches
            # and a mid-epoch IndexError. Fail here, not at step N.
            raise ValueError(
                "balanced/stratified sampling is incompatible with eval_split_ratio; "
                "provide eval_manifest instead (the sampler indexes the unsplit dataset)"
            )
        if self.balanced_sampling and self.sampling_strategy == "stratified":
            languages = {language for _, language in self.train_manifest_specs}
            languages.update(language for _, language in self.non_speech_manifest_specs)
            if languages and self.per_device_train_batch_size % len(languages):
                raise ValueError(
                    f"per_device_train_batch_size ({self.per_device_train_batch_size}) "
                    f"must be a multiple of the number of languages "
                    f"({len(languages)}: {', '.join(sorted(languages))}) under "
                    "sampling_strategy: stratified"
                )
        if not 0 < self.sampling_temperature <= 1:
            raise ValueError("sampling_temperature must be in the range (0, 1]")
        if not 0 < self.llm_lr_factor <= 1:
            raise ValueError("llm_lr_factor must be in the range (0, 1]")
        if not 0 < self.embed_lr_factor <= 1:
            raise ValueError("embed_lr_factor must be in the range (0, 1]")
        if not 0 <= self.label_smoothing < 0.2:
            raise ValueError("label_smoothing must be in the range [0, 0.2)")
        if self.lr_scheduler_kwargs is not None and not isinstance(self.lr_scheduler_kwargs, dict):
            raise ValueError("lr_scheduler_kwargs must be a mapping or null")
        if self.ctc_loss_weight < 0:
            raise ValueError("ctc_loss_weight must be non-negative")
        if self.eval_wer_samples_per_language < 0:
            raise ValueError("eval_wer_samples_per_language must be non-negative")
        _expand_manifest_config(
            self.non_speech_manifest,
            default_language=self.language,
            field_name="non_speech_manifest",
            required=False,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def train_manifest_paths(self) -> list[str]:
        return [path for path, _ in self.train_manifest_specs]

    @property
    def train_manifest_specs(self) -> list[tuple[str, str]]:
        return _expand_manifest_config(
            self.train_manifest,
            default_language=self.language,
            field_name="train_manifest",
            required=True,
        )

    @property
    def eval_manifest_paths(self) -> list[str]:
        return [path for path, _ in self.eval_manifest_specs]

    @property
    def eval_manifest_specs(self) -> list[tuple[str, str]]:
        return _expand_manifest_config(
            self.eval_manifest,
            default_language=self.language,
            field_name="eval_manifest",
            required=False,
        )

    @property
    def non_speech_manifest_specs(self) -> list[tuple[str, str]]:
        return _expand_manifest_config(
            self.non_speech_manifest,
            default_language=self.language,
            field_name="non_speech_manifest",
            required=False,
        )


def validate_local_checkpoint(model_name_or_path: str) -> None:
    """Fail fast, with the real reason, on a bad local checkpoint path.

    transformers turns a missing directory into a baffling HFValidationError
    (it assumes a Hub repo id), and an incomplete directory into a per-file
    OSError deep inside processor loading — both multiplied by every rank.
    Check the path and the artifact set up front instead.
    """
    if not model_name_or_path.startswith(("/", "./", "../", "~")):
        return  # Hub repo id — let transformers resolve it
    path = Path(model_name_or_path).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"model_name_or_path does not exist on this node: {model_name_or_path}\n"
            "Point it at the converted checkpoint (qasr.convert_weights output) "
            "or a previous phase's output_dir, and check the filesystem is "
            "mounted on every node."
        )
    missing = [name for name in ("config.json",) if not (path / name).is_file()]
    # transformers <5 saved the feature extractor as preprocessor_config.json;
    # 5.x folds it into processor_config.json. Either layout loads.
    if not any((path / name).is_file() for name in ("preprocessor_config.json", "processor_config.json")):
        missing.append("preprocessor_config.json or processor_config.json")
    if not any(path.glob("model*.safetensors")):
        missing.append("model*.safetensors")
    if missing:
        raise FileNotFoundError(
            f"Checkpoint at {model_name_or_path} is INCOMPLETE — missing: "
            f"{', '.join(missing)}.\n"
            "Likely an interrupted or failed conversion/save. Re-run the "
            "conversion (scripts/run_convert_weights.sh verifies completeness) "
            "or point at a known-good checkpoint. Do not train on this "
            "directory."
        )


def parse_config(argv: list[str] | None = None) -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the QASR Conformer-Qwen3 hybrid.")
    parser.add_argument("--config", required=True, help="Path to a YAML training configuration.")
    parser.add_argument("--train-manifest", nargs="+")
    parser.add_argument("--eval-manifest", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    config = TrainConfig.from_yaml(args.config)
    for name in ("train_manifest", "eval_manifest", "output_dir", "resume_from_checkpoint"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.smoke_test:
        config.smoke_test = True
    config.validate()
    return config


__all__ = ["TrainConfig", "parse_config", "validate_local_checkpoint"]
