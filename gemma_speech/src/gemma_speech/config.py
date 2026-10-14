"""Strict, serializable experiment configuration."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"


@dataclass
class ModelConfig:
    task: str = "asr"
    model_id: str = "google/embeddinggemma-2"
    revision: str = MODEL_REVISION
    local_files_only: bool = False
    width: int = 384
    layers: int = 4
    heads: int = 6
    dropout: float = 0.1
    train_backbone_layers: int = 0
    asr_upsample: int = 1
    intermediate_ctc_weight: float = 0.2
    use_semantic_text: bool = True
    mel_bins: int = 100
    mel_pack: int = 4
    mel_mean: float = -5.0
    mel_std: float = 2.0
    condition_dropout: float = 0.1
    duration_weight: float = 0.1
    asr_decoder: str = "ctc"
    qwen_model_id: str = "Qwen/Qwen3-ASR-0.6B"
    qwen_revision: str = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
    qwen_train_layers: int = 0
    qwen_audio_stride: int = 2

    def __post_init__(self):
        if self.task not in {"asr", "tts"}:
            raise ValueError("task must be asr or tts")
        if self.asr_decoder not in {"ctc", "qwen"}:
            raise ValueError("asr_decoder must be ctc or qwen")
        if self.qwen_train_layers < 0 or self.qwen_audio_stride not in {1, 2}:
            raise ValueError("invalid Qwen decoder layer count or audio stride")
        if self.width < 4 or self.heads < 1 or self.width % self.heads or self.width % 2:
            raise ValueError("width must be even and divisible by heads")
        if self.layers < 0 or (self.task == "tts" and self.layers == 0):
            raise ValueError("layers must be nonnegative (positive for TTS)")
        if self.train_backbone_layers < 0 or self.asr_upsample not in {1, 2}:
            raise ValueError("invalid backbone layer count or ASR upsample factor")
        for name in ("dropout", "condition_dropout", "intermediate_ctc_weight"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.mel_bins != 100 or self.mel_pack < 1 or self.mel_std <= 0:
            raise ValueError("Vocos contract requires 100 mel bins, positive pack and mel_std")
        if self.duration_weight <= 0:
            raise ValueError("duration_weight must be positive")


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train_manifest: str = ""
    validation_manifest: str = ""
    tokenizer: str = ""
    output_dir: str = ""
    seed: int = 42
    batch_size: int = 4
    accumulation_steps: int = 1
    max_steps: int = 5000
    samples_per_epoch: int = 0
    learning_rate: float = 3e-4
    backbone_learning_rate: float = 1e-5
    warmup_steps: int = 250
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "bf16"
    workers: int = 4
    max_audio_seconds: float = 30.0
    max_text_tokens: int = 1024
    eval_every: int = 250
    save_every: int = 500
    language_weights: dict[str, float] = field(default_factory=dict)
    require_multilingual: bool = True

    def __post_init__(self):
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        for name in (
            "batch_size",
            "accumulation_steps",
            "max_steps",
            "max_text_tokens",
            "eval_every",
            "save_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.workers < 0 or self.warmup_steps < 0 or self.samples_per_epoch < 0:
            raise ValueError("workers, warmup_steps and samples_per_epoch must be nonnegative")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("EmbeddingGemma 2 supports fp32/bf16; fp16 is unsafe")
        if (
            min(
                self.learning_rate,
                self.backbone_learning_rate,
                self.grad_clip,
                self.max_audio_seconds,
            )
            <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("invalid learning rate, clipping, duration or weight decay")
        if self.max_text_tokens > 8192:
            raise ValueError("text limit exceeds the documented 8192-token context")
        if any(v <= 0 for v in self.language_weights.values()):
            raise ValueError("all language weights must be positive")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        # All data/output paths are relative to the config, never to a launch node's cwd.
        for key in ("train_manifest", "validation_manifest", "tokenizer", "output_dir"):
            if key == "tokenizer" and data.get("model", {}).get("asr_decoder") == "qwen":
                data[key] = ""
                continue
            if not data.get(key):
                raise ValueError(f"config requires {key}")
            p = Path(data[key]).expanduser()
            data[key] = str((path.parent / p).resolve())
        for key in ("model_id", "qwen_model_id"):
            value = data.get("model", {}).get(key, "")
            if value.startswith(("./", "../", "~/")):
                data["model"][key] = str((path.parent / Path(value).expanduser()).resolve())
        return cls(**data)

    def to_dict(self):
        return asdict(self)
