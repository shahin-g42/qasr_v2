"""Immutable export: a sealed component bundle bound to its calibration.

Export never invents a temperature: it either receives an explicit one or reads
the sealed calibration artifact, whose checkpoint identity must match the
checkpoint being exported. Weights are reshared from the Distributed Checkpoint
into a full single-process model on CPU, so no GPU is required and the saved
directory is sealed by DecisionModel.save_pretrained's manifest as usual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from jev_contracts import canonical_json, verify_manifest

from .calibrate import load_calibration
from .distributed import materialize_empty
from .modeling import DecisionModel, build_meta_model

_FULL = StateDictOptions(full_state_dict=True, cpu_offload=False)


def _positive_temperature(value: object) -> float:
    if (type(value) not in (int, float) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("temperature must be a positive finite number, not bool")
    return float(value)


def checkpoint_identity(directory: str | Path) -> dict:
    """Verified identity of a sealed training checkpoint."""
    directory = Path(directory).absolute()
    manifest = verify_manifest(directory)
    if manifest.get("metadata", {}).get("kind") != "jev-checkpoint":
        raise ValueError(f"Not a sealed jev checkpoint: {directory}")
    descriptor = json.loads(
        (directory / "descriptor.json").read_text(encoding="utf-8"),
    )
    if descriptor.get("identity") != manifest["metadata"]["descriptor"]["identity"]:
        raise ValueError("Checkpoint descriptor and manifest identities disagree")
    return {
        "manifest": hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest(),
        "descriptor": descriptor,
    }


def _full_model(snapshot: str | Path, initialization: str) -> DecisionModel:
    model = build_meta_model(Path(snapshot), initialization)
    materialize_empty(model, "cpu")
    model.materialize_runtime_buffers("cpu")
    return model


def _assert_finite(model: DecisionModel) -> None:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError(f"Exported parameter {name!r} contains nonfinite values")


def load_model_from_checkpoint(
    checkpoint: str | Path, snapshot: str | Path, *, initialization: str,
) -> DecisionModel:
    """Reshard a sealed Distributed Checkpoint into one full CPU model."""
    model = _full_model(snapshot, initialization)
    state = {"model": get_model_state_dict(model, options=_FULL)}
    dcp_load(state, storage_reader=FileSystemReader(Path(checkpoint).absolute()))
    set_model_state_dict(model, state["model"], options=_FULL)
    model.materialize_runtime_buffers("cpu")
    _assert_finite(model)
    return model


def gather_full_state(model: Any) -> dict | None:
    """Rank-zero full CPU state from a live FSDP2 model; other ranks get None."""
    # Torch 2.13 folds rank0-only gathering into cpu_offload: with both options
    # set, rank 0 receives the full CPU state and every other rank an empty
    # mapping, which this wrapper collapses to None.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    state = get_model_state_dict(model, options=options)
    return state if state else None


def export_model(
    model: DecisionModel, destination: str | Path, *, temperature: float,
    metadata: dict | None = None,
) -> dict:
    """Seal the exported bundle; the caller owns the calibrated temperature."""
    scale = _positive_temperature(temperature)
    if metadata is not None:
        if type(metadata) is not dict:
            raise ValueError("Export metadata must be a mapping")
        model.metadata.update(metadata)
        canonical_json(model.metadata)
    model.temperature = scale
    model.save_pretrained(Path(destination))
    return json.loads(
        (Path(destination) / "decision_config.json").read_text(encoding="utf-8"),
    )


def export_from_state(
    destination: str | Path, snapshot: str | Path, *, initialization: str,
    full_state: dict, temperature: float, metadata: dict | None = None,
) -> dict:
    """Export a gathered full state dict without re-reading the checkpoint."""
    model = _full_model(snapshot, initialization)
    model.load_state_dict(full_state, strict=True, assign=True)
    model.materialize_runtime_buffers("cpu")
    _assert_finite(model)
    return export_model(model, destination, temperature=temperature, metadata=metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev-export",
        description="Export a sealed checkpoint as an immutable component bundle",
    )
    parser.add_argument("--checkpoint", required=True, help="Sealed jev checkpoint directory")
    parser.add_argument("--snapshot", required=True, help="Original pinned model snapshot")
    parser.add_argument("--initialization", default="autojev", choices=["autojev", "fresh"])
    parser.add_argument("--destination", required=True, help="Fresh directory for the bundle")
    parser.add_argument(
        "--calibration", default=None,
        help="Sealed calibration artifact; its checkpoint identity must match",
    )
    parser.add_argument(
        "--temperature", default=None,
        help="Explicit temperature; required exactly when --calibration is absent",
    )
    args = parser.parse_args(argv)
    identity = checkpoint_identity(args.checkpoint)
    if args.calibration:
        artifact = load_calibration(args.calibration)
        if artifact["identity"].get("checkpoint") != identity["manifest"]:
            raise SystemExit(
                "Calibration artifact was produced for a different checkpoint; refusing to export"
            )
        temperature = artifact["temperature"]["temperature"]
        metadata = {"calibration": artifact["identity"], "source_checkpoint": identity["manifest"]}
    else:
        if args.temperature is None:
            raise SystemExit("Provide --calibration or an explicit --temperature")
        temperature = _positive_temperature(float(args.temperature))
        metadata = {"source_checkpoint": identity["manifest"]}
    model = load_model_from_checkpoint(
        args.checkpoint, args.snapshot, initialization=args.initialization)
    exported = export_model(
        model, args.destination, temperature=temperature, metadata=metadata,
    )
    print(canonical_json({
        "destination": str(Path(args.destination).absolute()),
        "temperature": exported["temperature"], "codes": len(exported["codes"]),
        "source_checkpoint": identity["manifest"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
