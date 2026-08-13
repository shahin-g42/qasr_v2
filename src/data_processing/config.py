"""Pipeline configuration for the Arabic transcript processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PipelineConfig:
    """Configuration for the distributed Arabic transcript cleaning pipeline."""

    # vLLM server settings
    vllm_model: str = "Qwen/Qwen3.6-35B-A3B"
    vllm_host: str = "localhost"
    vllm_port: int = 8000
    vllm_tp: int = 8
    vllm_max_model_len: int = 16384
    vllm_max_num_seqs: int = 512
    vllm_gpu_memory_utilization: float = 0.93

    # Worker settings
    workers_per_node: int = 96
    batch_size: int = 32
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    request_timeout: float = 120.0
    checkpoint_interval: int = 100

    # Paths
    manifests_config: str = "configs/train_full_8node_filtered.yaml"
    output_dir: str = "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/cleaned_manifests"
    checkpoint_dir: str = "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/cleaning_checkpoints"
    shard_dir: str = "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/cleaning_shards"

    # Processing options
    language_filter: str = "ar"
    preserve_dialects: bool = True
    restore_diacritics: bool = True
    # itn_enabled gates ITN verification of the cleaner's output: date/time
    # field sanity, mis-assembled digit readouts, and numbers dropped from
    # the source in every language, plus Arabic spoken-number-word checks.
    # The cleaner prompt always asks for ITN; this only controls whether we
    # verify the result.
    itn_enabled: bool = True
    # NOTE: there is deliberately no preserve_code_switching flag. Keeping
    # code-switched words in the script they were spoken in is part of
    # verbatim fidelity (cleaner prompt rule 0), which nothing may turn off,
    # so its check runs unconditionally in both validators.
    # NOTE: punctuate is not wired — the cleaner prompt always requests
    # punctuation restoration. Kept for config compatibility.
    punctuate: bool = True
    validation_sample_rate: float = 0.05
    # Read missing durations from the audio header during cleaning.
    # Records that already carry a duration cost zero I/O (they are
    # filtered out before any filesystem access), so this only pays for
    # manifests that actually lack the field — and it keeps us from
    # needing a separate full backfill pass over the audio afterwards.
    probe_missing_durations: bool = True
    duration_probe_workers: int = 32
    # After a record is rejected, run one issue-driven correction pass that
    # repairs the SPECIFIC issues the validator flagged, then re-validate.
    # Rescues the "invalid but no correction offered" rejects at the cost of
    # up to 2 extra LLM calls per rejected record (reject path only).
    correction_enabled: bool = True
    # Enable Qwen3 thinking mode for the VALIDATOR only (A/B knob).
    # Adds ~500-2000 reasoning tokens per verdict (slower, more truncation
    # headroom used) — adopt only if it measurably cuts false rejections.
    validator_thinking: bool = False

    # Node distribution
    node_rank: int = 0
    num_nodes: int = 8

    # Single-manifest slicing (for spreading ONE huge manifest across nodes).
    # manifest_filter: comma-separated manifest stems to process (empty = all).
    # record_range: "START:END" slice of valid-record indices (END empty = EOF).
    # skip_processed: skip records already present in ANY prior output of the
    # manifest (cleaned/shard/rejected) — lets slices coexist with partial
    # full-manifest runs without duplicating work.
    manifest_filter: str = ""
    record_range: str = ""
    skip_processed: bool = False

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load configuration from a YAML file."""
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
        """Validate configuration values."""
        if self.workers_per_node < 1:
            raise ValueError("workers_per_node must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if not 0 <= self.validation_sample_rate <= 1:
            raise ValueError("validation_sample_rate must be in [0, 1]")
        if self.duration_probe_workers < 1:
            raise ValueError("duration_probe_workers must be positive")
        if self.vllm_port < 1 or self.vllm_port > 65535:
            raise ValueError("vllm_port must be a valid port number")
        if self.node_rank < 0 or self.node_rank >= self.num_nodes:
            raise ValueError(f"node_rank must be in [0, {self.num_nodes})")
        if not 0 < self.vllm_gpu_memory_utilization <= 1:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")
        self.parse_record_range()  # raises on malformed record_range

    def parse_record_range(self) -> tuple[int, int | None]:
        """Parse record_range 'START:END' into (start, end). END empty = EOF."""
        if not self.record_range:
            return 0, None
        raw = self.record_range.strip()
        start_s, sep, end_s = raw.partition(":")
        if not sep:
            raise ValueError(
                f"record_range must be 'START:END' (END empty for EOF): {raw!r}"
            )
        try:
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else None
        except ValueError as exc:
            raise ValueError(f"Invalid record_range {raw!r}: {exc}") from exc
        if start < 0 or (end is not None and end <= start):
            raise ValueError(f"Invalid record_range {raw!r}: need 0 <= START < END")
        return start, end

    @property
    def vllm_base_url(self) -> str:
        """Base URL for the vLLM OpenAI-compatible API."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a dictionary."""
        from dataclasses import asdict

        return asdict(self)


__all__ = ["PipelineConfig"]
