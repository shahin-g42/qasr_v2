"""Strict, hashable training configuration loaded from TOML.

Unknown keys are rejected so a typo can never silently fall back to a default.
The identity hash covers everything that defines what is trained; log and
checkpoint locations and the resume pointer are excluded so a run can be
restarted into fresh directories after an interruption.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from jev_contracts import canonical_json

_ALLOWED_DTYPES = {"bfloat16": "bfloat16", "float32": "float32"}
_MAX_LENGTH_LIMIT = 8192


def _int_field(value: object, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer, not bool or float")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"..{maximum}" if maximum is not None else " or more"
        raise ValueError(f"{name} must be in {minimum}{bound}")
    return value


def _float_field(
    value: object, name: str, *, minimum: float, maximum: float | None = None,
    inclusive: bool = False,
) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not bool")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (not inclusive and value == minimum) or (
        maximum is not None and (value > maximum or (not inclusive and value == maximum))
    ):
        raise ValueError(f"{name} must lie strictly/weakly within ({minimum}, {maximum})")
    return value


def _path_field(value: object, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string path")
    return value


@dataclass(frozen=True)
class TrainConfig:
    """Every field participates in the run identity except the location fields."""

    snapshot: str
    release: str
    run_dir: str
    initialization: str = "autojev"
    seed: int = 0
    epochs: int = 1
    global_batch: int = 256
    micro_cap: int = 2
    token_budget: int = 32768
    max_length: int = 8192
    learning_rate: float = 2e-6
    lr_min_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    clip_norm: float = 1.0
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    device: str = "cuda"
    shard: int = 8
    replicate: int = 8
    foreach_optimizer: bool = True
    eval_every: int = 50
    checkpoint_every: int = 50
    keep_checkpoints: int = 2
    timeout_minutes: float = 30.0
    checkpoint_dir: str = ""
    resume: str = ""

    def __post_init__(self) -> None:
        _path_field(self.snapshot, "snapshot")
        _path_field(self.release, "release")
        _path_field(self.run_dir, "run_dir")
        if self.initialization not in {"autojev", "fresh"}:
            raise ValueError("initialization must be 'autojev' or 'fresh'")
        _int_field(self.seed, "seed", minimum=0)
        _int_field(self.epochs, "epochs", minimum=1)
        _int_field(self.global_batch, "global_batch", minimum=1)
        _int_field(self.micro_cap, "micro_cap", minimum=1)
        _int_field(self.token_budget, "token_budget", minimum=1)
        _int_field(self.max_length, "max_length", minimum=1, maximum=_MAX_LENGTH_LIMIT)
        if self.token_budget < self.max_length:
            raise ValueError("token_budget must fit at least one max_length example")
        _float_field(self.learning_rate, "learning_rate", minimum=0.0)
        _float_field(self.lr_min_ratio, "lr_min_ratio", minimum=0.0, maximum=1.0, inclusive=True)
        _float_field(self.warmup_ratio, "warmup_ratio", minimum=0.0, maximum=1.0)
        _float_field(self.weight_decay, "weight_decay", minimum=0.0, inclusive=True)
        _float_field(self.beta1, "beta1", minimum=0.0, maximum=1.0)
        _float_field(self.beta2, "beta2", minimum=0.0, maximum=1.0)
        _float_field(self.epsilon, "epsilon", minimum=0.0)
        _float_field(self.clip_norm, "clip_norm", minimum=0.0)
        _float_field(self.timeout_minutes, "timeout_minutes", minimum=0.0)
        if self.dtype not in _ALLOWED_DTYPES:
            raise ValueError("dtype must be 'bfloat16' or 'float32'")
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be a boolean")
        if type(self.foreach_optimizer) is not bool:
            raise ValueError("foreach_optimizer must be a boolean")
        _int_field(self.shard, "shard", minimum=1)
        _int_field(self.replicate, "replicate", minimum=1)
        _int_field(self.eval_every, "eval_every", minimum=0)
        _int_field(self.checkpoint_every, "checkpoint_every", minimum=1)
        _int_field(self.keep_checkpoints, "keep_checkpoints", minimum=1)
        if self.checkpoint_dir:
            _path_field(self.checkpoint_dir, "checkpoint_dir")
        if self.resume:
            _path_field(self.resume, "resume")

    @property
    def mesh_shape(self) -> tuple[int, int]:
        """(dp_replicate, dp_shard): the exact FSDP2 HSDP mesh dimensions."""
        return self.replicate, self.shard

    def to_dict(self) -> dict:
        return asdict(self)

    def identity(self) -> dict:
        """The canonical what-is-trained mapping; locations are excluded."""
        excluded = {"run_dir", "checkpoint_dir", "resume"}
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in excluded
        }

    @property
    def identity_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.identity()).encode("utf-8")).hexdigest()

    @classmethod
    def from_toml(cls, path: str | Path) -> TrainConfig:
        """Load a config file; unknown keys fail rather than train the wrong thing."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Training config not found: {path}")
        with path.open("rb") as handle:
            document = tomllib.load(handle)
        if type(document) is not dict:
            raise ValueError("Training config must be a TOML table")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(document) - known)
        if unknown:
            raise ValueError(f"Unknown training configuration keys: {unknown}")
        required = {"snapshot", "release", "run_dir"}
        missing = sorted(required - set(document))
        if missing:
            raise ValueError(f"Training configuration is missing required keys: {missing}")
        return cls(**document)
