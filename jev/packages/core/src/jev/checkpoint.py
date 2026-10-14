"""Optimizer-boundary checkpoints with immutable manifests and identity checks.

Every checkpoint is a fresh directory whose Distributed Checkpoint payload holds
exactly the model and optimizer state, sealed by a small descriptor and finally
by a contracts manifest. The manifest is the completion marker: a directory
without one is an interrupted save that must never be resumed. The run state
(cursor, LR schedule position, RNG) lives in the descriptor rather than the
DCP payload: plain mappings would be flattened into dotted leaves by DCP and
so cannot be reloaded through a single placeholder, whereas the descriptor is
JSON, manifest-verified, and human-auditable. RNG tensors are base64-encoded
uint8 buffers, so resume never depends on pickle compatibility. Resume verifies
the full identity (config, release, snapshot, prompt, mesh) before any state
is restored.
"""

from __future__ import annotations

import base64
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from jev_contracts import atomic_write_json, verify_manifest, write_manifest

DESCRIPTOR_NAME = "descriptor.json"
MANIFEST_NAME = "manifest.json"
DESCRIPTOR_SCHEMA = 1
_SHARDED = StateDictOptions(full_state_dict=False, cpu_offload=False)


def _encode_rng_tensor(tensor: torch.Tensor) -> dict:
    """A uint8 RNG buffer as strict JSON: shape plus base64 of the raw bytes."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        raise ValueError("RNG state tensors must be uint8 buffers")
    flat = tensor.detach().cpu().reshape(-1)
    return {
        "shape": [int(size) for size in tensor.shape],
        "data": base64.b64encode(bytes(flat.tolist())).decode("ascii"),
    }


def _decode_rng_tensor(encoded: dict) -> torch.Tensor:
    if type(encoded) is not dict or type(encoded.get("shape")) is not list:
        raise ValueError("Malformed encoded RNG tensor")
    shape = tuple(encoded["shape"])
    if any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Malformed RNG tensor shape")
    data = base64.b64decode(encoded.get("data", ""), validate=True)
    expected = 1
    for size in shape:
        expected *= size
    if len(data) != expected:
        raise ValueError("Encoded RNG buffer length disagrees with its shape")
    if not data:
        return torch.zeros(shape, dtype=torch.uint8)
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().reshape(shape)


def _encode_python_state(state: tuple) -> list:
    version, internal, gauss = state
    return [int(version), [int(value) for value in internal], gauss]


def _decode_python_state(value: list) -> tuple:
    version, internal, gauss = value
    return (int(version), tuple(int(item) for item in internal), gauss)


def capture_rng(device: Any) -> dict:
    """Every RNG stream the loop can consume, as a JSON-serializable mapping.

    The training plan itself never samples: permutation seeds are derived from
    content. These streams are the precautionary state for anything incidental,
    captured from rank zero's view and restored uniformly on every rank so a
    resumed world is exactly as deterministic as a fresh one.
    """
    target = torch.device(device)
    device_rng = (
        torch.cuda.get_rng_state(target) if target.type == "cuda" and torch.cuda.is_available()
        else torch.zeros(0, dtype=torch.uint8)
    )
    return {
        "torch_cpu": _encode_rng_tensor(torch.get_rng_state()),
        "device_rng": _encode_rng_tensor(device_rng),
        "python": _encode_python_state(random.getstate()),
    }


def restore_rng(state: dict, device: Any) -> None:
    """Restore captured streams; resume replays only these, never sampling order."""
    if type(state) is not dict:
        raise ValueError("RNG state must be a mapping")
    torch.set_rng_state(_decode_rng_tensor(state["torch_cpu"]))
    target = torch.device(device)
    if target.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_rng_state(_decode_rng_tensor(state["device_rng"]), target)
    random.setstate(_decode_python_state(state["python"]))


def _require_fresh(directory: Path) -> Path:
    if directory.exists():
        raise FileExistsError(f"Checkpoint directory already exists: {directory}")
    directory.mkdir(parents=True)
    return directory


def save_checkpoint(
    directory: str | Path, *, model: Any, optimizer: Any, run_state: dict,
    identity: dict, mesh_shape: tuple[int, int], update: int,
) -> dict:
    """Collectively save model, optimizer, and run state; seal the directory.

    dcp.save is a collective over the current process group. Only rank zero
    writes the descriptor and manifest afterward, and the barrier guarantees no
    rank resumes before the completion marker is visible to all.
    """
    if type(run_state) is not dict or not run_state:
        raise ValueError("run_state must be a nonempty mapping")
    if type(identity) is not dict or not identity:
        raise ValueError("identity must be a nonempty mapping")
    if (type(update) is not int or update < 0 or len(mesh_shape) != 2
            or any(type(value) is not int or value < 1 for value in mesh_shape)):
        raise ValueError("update must be a nonnegative int and mesh_shape two positive ints")
    directory = _require_fresh(Path(directory).absolute())
    state = {
        "model": get_model_state_dict(model, options=_SHARDED),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=_SHARDED),
    }
    dcp_save(state, storage_writer=FileSystemWriter(directory, sync_files=True, thread_count=1))
    descriptor = {
        "schema_version": DESCRIPTOR_SCHEMA, "kind": "jev-checkpoint", "update": update,
        "identity": identity, "mesh": list(mesh_shape), "run": run_state,
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "torch": str(torch.__version__),
    }
    if not dist.is_initialized() or dist.get_rank() == 0:
        atomic_write_json(directory / DESCRIPTOR_NAME, descriptor)
        files = [
            str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()
        ]
        write_manifest(directory, files, {"kind": "jev-checkpoint", "descriptor": descriptor})
    if dist.is_initialized():
        dist.barrier()
    return descriptor


def _read_descriptor(directory: Path) -> dict:
    path = directory / DESCRIPTOR_NAME
    if not path.is_file():
        raise ValueError(f"Checkpoint has no descriptor: {directory}")
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    if (type(descriptor) is not dict or descriptor.get("schema_version") != DESCRIPTOR_SCHEMA
            or descriptor.get("kind") != "jev-checkpoint"
            or type(descriptor.get("update")) is not int
            or type(descriptor.get("identity")) is not dict
            or type(descriptor.get("mesh")) is not list
            or type(descriptor.get("run")) is not dict or not descriptor["run"]):
        raise ValueError(f"Malformed checkpoint descriptor: {directory}")
    return descriptor


def _verify_identity(descriptor: dict, identity: dict, mesh_shape: tuple[int, int],
                     directory: Path) -> None:
    if descriptor["identity"] != identity:
        differing = sorted(
            key for key in set(descriptor["identity"]) | set(identity)
            if descriptor["identity"].get(key) != identity.get(key)
        )
        raise ValueError(
            f"Checkpoint identity mismatch in {differing}: refusing to resume {directory}"
        )
    if descriptor["mesh"] != list(mesh_shape):
        raise ValueError(
            f"Checkpoint mesh {descriptor['mesh']} differs from requested {list(mesh_shape)}; "
            "cross-world resharding is not qualified"
        )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if descriptor["world_size"] != world_size:
        raise ValueError(
            f"Checkpoint world size {descriptor['world_size']} differs from {world_size}; "
            "cross-world resume is not qualified"
        )


def load_checkpoint(
    directory: str | Path, *, model: Any, optimizer: Any,
    identity: dict, mesh_shape: tuple[int, int],
) -> dict:
    """Verify, then restore model, optimizer, and run state in place.

    The DCP payload is exactly {model, optimizer}: loading into a fresh
    optimizer may materialize zero moments with a zero-LR step that cannot
    change parameters, and the loaded optimizer state then replaces the
    placeholder including step counts. The run state is returned from the
    manifest-verified descriptor written at save time.
    """
    directory = Path(directory).absolute()
    manifest = verify_manifest(directory)
    if manifest.get("metadata", {}).get("kind") != "jev-checkpoint":
        raise ValueError(f"Not a sealed jev checkpoint: {directory}")
    descriptor = _read_descriptor(directory)
    _verify_identity(descriptor, identity, mesh_shape, directory)
    state = {
        "model": get_model_state_dict(model, options=_SHARDED),
        "optimizer": get_optimizer_state_dict(model, optimizer, options=_SHARDED),
    }
    dcp_load(state, storage_reader=FileSystemReader(directory))
    set_model_state_dict(model, state["model"], options=_SHARDED)
    set_optimizer_state_dict(model, optimizer, state["optimizer"], options=_SHARDED)
    return descriptor["run"]


def checkpoint_matches(directory: str | Path, *, update: int, identity: dict) -> bool:
    """True when a sealed checkpoint already stores exactly this update and identity."""
    directory = Path(directory)
    if type(update) is not int or update < 0 or type(identity) is not dict:
        raise ValueError("update must be a nonnegative int and identity a mapping")
    if not directory.is_dir() or not (directory / MANIFEST_NAME).is_file():
        return False
    manifest = verify_manifest(directory)
    if manifest.get("metadata", {}).get("kind") != "jev-checkpoint":
        return False
    descriptor = _read_descriptor(directory)
    return descriptor["update"] == update and descriptor["identity"] == identity


def find_checkpoints(root: str | Path) -> list[tuple[int, Path]]:
    """Sealed checkpoints only: a manifest is the proof a save completed."""
    root = Path(root)
    if not root.is_dir():
        return []
    found = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / MANIFEST_NAME).is_file():
            continue
        try:
            descriptor = _read_descriptor(directory)
        except (ValueError, OSError):
            continue
        found.append((descriptor["update"], directory))
    return sorted(found, key=lambda item: item[0])


def latest_checkpoint(root: str | Path) -> Path | None:
    """The highest-update sealed checkpoint, or None for a fresh start."""
    found = find_checkpoints(root)
    return found[-1][1] if found else None


def prune_checkpoints(root: str | Path, *, keep: int) -> None:
    """Delete the oldest sealed checkpoints beyond the newest `keep` ones.

    Only directories that carry both descriptor and manifest are ever removed,
    and the newest checkpoint always survives. Interrupted saves without a
    manifest are left for the operator to inspect.
    """
    if type(keep) is not int or keep < 1:
        raise ValueError("keep must be a positive integer")
    found = find_checkpoints(root)
    for _, directory in found[:-keep] if len(found) > keep else []:
        shutil.rmtree(directory)
