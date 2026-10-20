"""Register the user's fresh GQASR training checkpoint without changing old scores."""

from __future__ import annotations

import json
import os
from pathlib import Path

from gqasr.arabic_benchmark.runner import checkpoint_identity

from .core import digest

NAME = "gqasr-encoder12-b80-step7500"
ENTRY = {
    "source_type": "local_checkpoint",
    "backend": "gqasr",
    "display_name": "GQASR · encoder12_b80 · step 7,500",
    "max_new_tokens": 768,
}
DEFAULT_CHECKPOINT = "/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/gemma_speech/gqasr/outputs/encoder12_b80/checkpoint-00007500"
QASR_NAME = "qasr-v81-full-12node"
ENTRIES = {
    NAME: ENTRY,
    QASR_NAME: {
        "source_type": "local_checkpoint",
        "backend": "qasr",
        "display_name": "QASR · v81",
        "max_new_tokens": 768,
    },
}


def checkpoint_path(name=NAME):
    if name == QASR_NAME:
        return Path(os.environ.get(
            "QASR_REFRESH_CHECKPOINT",
            "/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/v81/full_12node",
        ))
    if name != NAME:
        raise ValueError("Unregistered local checkpoint")
    return Path(os.environ.get("GQASR_REFRESH_CHECKPOINT", DEFAULT_CHECKPOINT))


def checkpoint_metadata(path, name=NAME):
    """Check the inference artifact, never deserialize optimizer/training pickle."""
    path = Path(path)
    if name == QASR_NAME:
        files = checkpoint_identity(path, "qasr")
        return {
            **ENTRIES[name],
            "checkpoint_digest": digest(files),
            "training_step": None,
            "postprocessing": "native greedy Arabic decoding; preserve repetitions",
        }
    if name != NAME:
        raise ValueError("Unregistered local checkpoint")
    for filename in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "tokenizer/tokenizer_config.json",
        "tokenizer/tokenizer.json",
    ):
        if not (path / filename).is_file() or not (path / filename).stat().st_size:
            raise ValueError(f"Incomplete GQASR checkpoint: missing or empty {path / filename}")
    config = json.loads((path / "config.json").read_text())
    if (
        config.get("model_type") != "gqasr"
        or not {"acoustic_config", "decoder_config"} <= config.keys()
    ):
        raise ValueError("GQASR checkpoint lacks its acoustic/decoder configuration")
    provenance_path = path / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.is_file() else {}
    step = provenance.get("training", {}).get("completed_optimizer_steps")
    if step != 7500:
        raise ValueError(
            "This registered GQASR version requires provenance for optimizer step 7500"
        )
    # Reading the safetensors header detects incomplete weight writes without loading
    # tensors onto a GPU. Full strict key/shape validation happens in the smoke run.
    from safetensors import safe_open

    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as weights:
        if not list(weights.keys()):
            raise ValueError("GQASR checkpoint contains no tensors")
    files = checkpoint_identity(path, "gqasr")
    return {
        **ENTRY,
        "checkpoint_digest": digest(files),
        "training_step": step,
        "postprocessing": "native greedy Arabic decoding; preserve repetitions",
    }


def model_config(work, output, name=NAME):
    checkpoint = checkpoint_path(name)
    return {
        "backend": ENTRIES[name]["backend"],
        "checkpoint": str(checkpoint.resolve()),
        "work_dir": str(Path(work).resolve()),
        "output_dir": str(Path(output).resolve()),
        "precision": "bf16",
        "max_new_tokens": ENTRIES[name]["max_new_tokens"],
        "modes": ["arabic"],
        "task_size": 16,
        "start_when_ready": True,
        "simultaneous_model_loads": os.environ.get("PUBLIC_MODEL_LOAD_CONCURRENCY", "all"),
        "startup_timeout_seconds": 1800,
        "progress_timeout_seconds": 1800,
        "benchmark_model": checkpoint_metadata(checkpoint, name),
    }


def validate_registration(metadata, run, name=NAME):
    if name not in ENTRIES or any(metadata.get(k) != v for k, v in ENTRIES[name].items()):
        raise ValueError("Fresh GQASR registration differs")
    step = 7500 if name == NAME else None
    if metadata.get("training_step") != step or metadata.get("checkpoint_digest") != digest(
        run["checkpoint"]
    ):
        raise ValueError("Fresh GQASR checkpoint identity differs")
