"""HSDP (FSDP2) setup, shard-aware initialization, and exact gradient semantics.

Mesh layout is (dp_replicate, dp_shard) so sharding stays node-local: FSDP2
treats mesh dim 0 as replication and dim 1 as sharding (HSDPMeshInfo in the
runtime), and torchrun assigns flat ranks node-major, making each dim-1 group
one node's GPUs. Managed gradients are averaged over the full mesh by the
FSDP2 default reducer; the unmanaged replicated readout is averaged manually
with the same divisor, so clipping and updates share one exact global-batch
gradient. Weight loading reads only each rank's dim-0 rows from safetensors:
no rank ever materializes a full parameter on the accelerator.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Replicate, Shard

from .modeling import _mtp_shapes, _Snapshot

if TYPE_CHECKING:
    from .modeling import DecisionModel

__all__ = [
    "assert_no_meta", "backend_for", "clip_gradients", "decoder_blocks",
    "enable_activation_checkpointing", "hsdp_mesh", "init_distributed",
    "init_sharded_weights", "materialize_empty", "parallelize",
    "replicate_size", "shard_range", "sync_replicated_gradients",
]


def backend_for(device_type: str) -> str:
    """The only qualified collectives backend for a device type."""
    if type(device_type) is not str:
        raise ValueError("device_type must be a string")
    if device_type == "cuda":
        return "nccl"
    if device_type == "cpu":
        return "gloo"
    raise ValueError("Only 'cuda' (NCCL) and 'cpu' (gloo) training devices are supported")


def init_distributed(*, device_type: str, timeout_minutes: float = 30.0) -> tuple[int, int, int]:
    """Join the torchrun-launched world; return (rank, world_size, local_rank)."""
    if (type(timeout_minutes) not in (int, float) or not 0 < timeout_minutes <= 24 * 60):
        raise ValueError("timeout_minutes must be a positive finite number of minutes")
    if not dist.is_available():
        raise RuntimeError("This torch build has no distributed collectives support")
    if not dist.is_initialized():
        dist.init_process_group(
            backend_for(device_type), timeout=timedelta(minutes=timeout_minutes),
        )
    rank, world_size = dist.get_rank(), dist.get_world_size()
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    if type(local_world_size) is not int or not 1 <= local_world_size <= world_size:
        raise ValueError("LOCAL_WORLD_SIZE must divide the world")
    if world_size % local_world_size:
        raise ValueError("world_size must be a multiple of the local world size")
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % local_world_size)))
    if not 0 <= local_rank < local_world_size:
        raise ValueError("LOCAL_RANK must index the local world")
    return rank, world_size, local_rank


def hsdp_mesh(
    *, device_type: str, shard: int, replicate: int, local_world_size: int,
) -> DeviceMesh:
    """Two-dimensional mesh (dp_replicate, dp_shard) with node-local sharding.

    FSDP2 shards over mesh dim 1 and replicates over dim 0, so shard groups are
    one node's GPUs exactly when shard == local_world_size and torchrun's
    node-major rank order is in effect. The layout is verified, not assumed.
    """
    if type(shard) is not int or type(replicate) is not int or shard < 1 or replicate < 1:
        raise ValueError("shard and replicate must be positive integers")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if shard * replicate != world_size:
        raise ValueError(f"shard*replicate={shard * replicate} must equal world_size={world_size}")
    if shard != local_world_size:
        raise ValueError("HSDP shard degree must equal the node-local GPU count")
    mesh = init_device_mesh(
        device_type, (replicate, shard), mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    shard_group = mesh.get_group("dp_shard")
    local_rank = int(os.environ.get("LOCAL_RANK", str(dist.get_rank() % shard)))
    if dist.get_rank(shard_group) != local_rank:
        raise ValueError(
            "Mesh shard groups are not node-local; verify node-major rank assignment "
            "(one torchrun process per GPU, LOCAL_RANK set)"
        )
    return mesh


def decoder_blocks(backbone: nn.Module) -> list[nn.Module]:
    """The complete decoder-block sequence; wrapping boundaries must be blocks."""
    language_model = getattr(backbone, "language_model", None)
    if language_model is None:
        return []
    layers = getattr(language_model, "layers", None)
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)):
        raise ValueError(
            "Unsupported backbone layout: language_model.layers must be the decoder block sequence"
        )
    return list(layers)


def enable_activation_checkpointing(model: DecisionModel) -> None:
    """Non-reentrant checkpointing at complete decoder-block boundaries."""
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )


def parallelize(
    model: DecisionModel, mesh: DeviceMesh, *, param_dtype: torch.dtype | None,
    reshard_after_forward: bool = True,
) -> DecisionModel:
    """fully_shard every decoder block and the backbone; keep the readout replicated.

    The tiny 255-row readout stays a dense FP32 parameter outside FSDP: sharding
    it would need uneven chunks, and its gradient is all-reduced manually with
    the same mean semantics FSDP2 applies to managed parameters.

    FSDP2 hooks the backward by walking the wrapped module's output for
    grad-requiring tensors and traverses only dict/list/tuple/NamedTuple/
    dataclass containers; the HF backbone's OrderedDict-based ModelOutput
    satisfies this. An opaque output would leave managed parameters
    gradientless until clip_gradients fails loudly on the missing grads.
    """
    if not isinstance(mesh, DeviceMesh):
        raise ValueError("parallelize requires an initialized DeviceMesh")
    if param_dtype not in (None, torch.bfloat16, torch.float32):
        raise ValueError("param_dtype must be None, bfloat16, or float32")
    kwargs: dict[str, Any] = {"mesh": mesh, "reshard_after_forward": reshard_after_forward}
    if param_dtype is not None:
        kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=torch.float32,
        )
    blocks = decoder_blocks(model.backbone)
    if blocks:
        container = model.backbone.language_model.layers
        for index, block in enumerate(blocks):
            container[index] = fully_shard(block, **kwargs)
    model.backbone = fully_shard(model.backbone, **kwargs)
    return model


def materialize_empty(model: nn.Module, device: Any) -> None:
    """Materialize meta parameters onto the device via Module.to_empty.

    After fully_shard, managed parameters are DTensor-backed objects whose
    tensor type forbids plain `.data` reassignment; to_empty is the mechanism
    FSDP2's own runtime prescribes for meta materialization and preserves the
    recorded mesh, placements, and global shapes. It also empties buffers with
    uninitialized storage, which is exactly why DecisionModel's contract
    requires materialize_runtime_buffers immediately afterward.
    """
    target = torch.device(device)
    if target.type not in ("cuda", "cpu"):
        raise ValueError("Only cuda/cpu materialization is supported")
    model.to_empty(device=target)


def assert_no_meta(model: nn.Module) -> None:
    """Fail loudly if any parameter or buffer survived materialization on meta."""
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    for name, tensor in tensors:
        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        if local.is_meta:
            raise ValueError(
                f"Tensor {name!r} is still on the meta device; materialize parameters and "
                "rebuild runtime buffers before sharded loading"
            )


def shard_range(rows_count: int, shard_size: int, shard_coord: int) -> tuple[int, int]:
    """Dim-0 rows for one shard under torch.chunk's ceil-based sharding.

    DTensor Shard(0) follows torch.chunk semantics: every chunk before the last
    holds ceil(rows/shards) rows. A shape mismatch against the actual local
    shard is reported by the caller as a qualification error, never silently
    re-derived.
    """
    if type(rows_count) is not int or rows_count < 1:
        raise ValueError("rows_count must be a positive integer")
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if type(shard_coord) is not int or not 0 <= shard_coord < shard_size:
        raise ValueError("shard_coord must index the shard dimension")
    chunk = -(-rows_count // shard_size)
    offset = shard_coord * chunk
    if offset >= rows_count:
        raise ValueError(
            "This shard coordinate holds no rows; sharding is more uneven than chunking allows"
        )
    return offset, min(chunk, rows_count - offset)


class _Layout:
    """Lazy, audited mapping from decision parameters to safetensors source rows.

    Naming, shape, dtype, and coverage rules mirror jev.modeling._state exactly;
    the world-size-one trainer test asserts both readers agree byte for byte.
    """

    def __init__(self, snapshot: _Snapshot, model: DecisionModel, initialization: str):
        if initialization not in {"autojev", "fresh"}:
            raise ValueError("initialization must be 'autojev' or 'fresh'")
        self.snapshot = snapshot
        self.initialization = initialization
        self.token_ids = list(model.token_ids)
        expected = {key: tuple(value.shape) for key, value in model.backbone.state_dict().items()}
        prefix = "model." if initialization == "fresh" else ""
        source_shapes = {prefix + key: shape for key, shape in expected.items()}
        text = model.backbone.config.text_config
        if initialization == "fresh":
            source_shapes["lm_head.weight"] = (text.vocab_size, text.hidden_size)
        index_name = "model.safetensors.index.json"
        if snapshot.exists(index_name):
            index = snapshot.json(index_name)
            weight_map = index.get("weight_map")
            if type(weight_map) is not dict or not weight_map or len(weight_map) > 100000:
                raise ValueError("Invalid/bounded weight_map required")
            paths = {}
            for key, filename in weight_map.items():
                if (not key or not isinstance(filename, str)
                        or not filename.endswith(".safetensors")):
                    raise ValueError("Invalid tensor name/shard filename in weight_map")
                paths[filename] = snapshot.path(filename)
            if len(paths) > 1024:
                raise ValueError("Too many checkpoint shards")
        else:
            paths = {"model.safetensors": snapshot.path("model.safetensors")}
            weight_map = None
        actual_shapes, dtypes = {}, {}
        for filename, path in paths.items():
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in actual_shapes:
                        raise ValueError(f"Duplicate tensor across shards: {key}")
                    if weight_map is not None and weight_map.get(key) != filename:
                        raise ValueError(f"Index/shard tensor mismatch: {key}")
                    tensor = handle.get_slice(key)
                    actual_shapes[key] = tuple(tensor.get_shape())
                    dtypes[key] = tensor.get_dtype()
        if weight_map is not None and set(weight_map) != set(actual_shapes):
            raise ValueError("Index names missing from actual shard tensors")
        mtp = {key for key in actual_shapes if key.startswith("mtp.")}
        if mtp and initialization == "fresh":
            if getattr(text, "mtp_num_hidden_layers", 1) != 1:
                raise ValueError("Only the audited single auxiliary MTP layout is supported")
            source_shapes.update(_mtp_shapes(text))
        missing = source_shapes.keys() - actual_shapes.keys()
        extra = actual_shapes.keys() - source_shapes.keys()
        if missing or extra:
            raise ValueError(
                f"Checkpoint tensor names mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        for key, shape in source_shapes.items():
            if actual_shapes[key] != shape:
                raise ValueError(
                    f"Checkpoint tensor shape mismatch: {key}: {actual_shapes[key]} != {shape}"
                )
            if dtypes[key] not in {"F16", "BF16", "F32", "F64"}:
                raise ValueError(f"Unsupported checkpoint tensor dtype: {key}: {dtypes[key]}")
        self._paths = paths
        self._handles: dict[str, Any] = {}
        self._file_of = (
            weight_map if weight_map is not None
            else {key: "model.safetensors" for key in actual_shapes}
        )
        self._prefix = prefix
        self._readout_handle = None
        if initialization == "autojev":
            self._readout_path = snapshot.path("readout.safetensors")

    def _handle(self, filename: str) -> Any:
        if filename not in self._handles:
            self._handles[filename] = safe_open(
                str(self._paths[filename]), framework="pt", device="cpu")
        return self._handles[filename]

    def rows(self, name: str, offset: int, size: int) -> torch.Tensor:
        """Exactly rows [offset, offset+size) of one parameter's source tensor."""
        if type(offset) is not int or type(size) is not int or offset < 0 or size < 1:
            raise ValueError("offset must be nonnegative and size positive")
        if name == "readout.weight":
            if self.initialization == "autojev":
                if self._readout_handle is None:
                    self._readout_handle = safe_open(
                        str(self._readout_path), framework="pt", device="cpu",
                    )
                return self._readout_handle.get_slice("weight")[offset:offset + size]
            handle = self._handle(self._file_of["lm_head.weight"])
            view = handle.get_slice("lm_head.weight")
            return torch.cat([view[i:i + 1] for i in self.token_ids[offset:offset + size]], dim=0)
        source = self._prefix + name.removeprefix("backbone.")
        return self._handle(self._file_of[source]).get_slice(source)[offset:offset + size]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.__exit__(None, None, None)
        self._handles.clear()
        if self._readout_handle is not None:
            self._readout_handle.__exit__(None, None, None)
            self._readout_handle = None

    def __enter__(self) -> _Layout:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def init_sharded_weights(
    model: DecisionModel, snapshot: Path, *, initialization: str,
) -> dict[str, int]:
    """Fill every rank's local shards by reading only its dim-0 source rows.

    Dense parameters (the readout) are loaded in full on every rank. After the
    copies, each replica verifies that all ranks holding the same shard coord
    read identical bytes; replicas that silently diverge would train apart.
    """
    source = _Snapshot(Path(snapshot))
    initialized: set[str] = set()
    rows_read = 0
    with _Layout(source, model, initialization) as layout:
        for name, param in sorted(model.named_parameters()):
            if isinstance(param, DTensor):
                mesh, shard_dim = _shard_placement(param)
                shard_size = mesh.size(shard_dim)
                shard_coord = dist.get_rank(mesh.get_group(shard_dim))
                offset, size = shard_range(int(param.shape[0]), shard_size, shard_coord)
                local = param.to_local()
                if tuple(local.shape) != (size,) + tuple(param.shape[1:]):
                    raise ValueError(
                        f"Local shard shape {tuple(local.shape)} for {name!r} does not match "
                        f"the expected chunk ({size}, {tuple(param.shape[1:])}); this "
                        "runtime's DTensor sharding differs from torch.chunk semantics "
                        "and must be qualified"
                    )
                rows = layout.rows(name, offset, size)
            else:
                rows = layout.rows(name, 0, int(param.shape[0]))
            if not bool(torch.isfinite(rows).all()):
                raise ValueError(f"Nonfinite source rows for parameter {name!r}")
            with torch.no_grad():
                target = param.to_local() if isinstance(param, DTensor) else param.data
                target.copy_(rows)
            initialized.add(name)
            rows_read += int(rows.shape[0])
    if initialized != {name for name, _ in model.named_parameters()}:
        raise ValueError("Sharded initialization did not cover every parameter exactly once")
    _verify_replica_shards(model)
    return {"parameters": len(initialized), "source_rows_read": rows_read}


def _shard_placement(param: DTensor) -> tuple[DeviceMesh, int]:
    mesh = param.device_mesh
    for dim, placement in enumerate(param.placements):
        if isinstance(placement, Shard):
            return mesh, dim
    raise ValueError("Managed parameter has no Shard placement on its mesh")


def _verify_replica_shards(model: nn.Module) -> None:
    """All ranks sharing a shard coordinate must hold byte-identical shards."""
    replicate_group, replicate_size = None, 0
    for _, param in model.named_parameters():
        if isinstance(param, DTensor):
            mesh = param.device_mesh
            for dim, placement in enumerate(param.placements):
                if isinstance(placement, Replicate):
                    replicate_group, replicate_size = mesh.get_group(dim), mesh.size(dim)
            break
    if replicate_group is None:
        return
    checksum = None
    for _, param in sorted(model.named_parameters()):
        local = (param.to_local() if isinstance(param, DTensor) else param.data).detach()
        value = local.double().sum()
        checksum = value if checksum is None else checksum + value
    if checksum is None or replicate_size < 2:
        return
    gathered = [torch.empty_like(checksum) for _ in range(replicate_size)]
    dist.all_gather(gathered, checksum, group=replicate_group)
    for other in gathered[1:]:
        if not bool(torch.equal(gathered[0], other)):
            raise ValueError(
                "Replica ranks read different source bytes for the same shards; verify the "
                "snapshot and shared storage before training"
            )


def replicate_size(grad: torch.Tensor) -> int:
    """How many identical copies of this gradient exist across the world."""
    if not isinstance(grad, DTensor):
        raise ValueError("replicate_size is defined for DTensor gradients")
    mesh = grad.device_mesh
    for dim, placement in enumerate(grad.placements):
        if isinstance(placement, Replicate):
            return int(mesh.size(dim))
    return 1


def sync_replicated_gradients(
    model: nn.Module, *, world_size: int, group: Any = None,
) -> list[str]:
    """Average dense (non-FSDP) parameter gradients over the full world.

    FSDP2's reducer averages managed gradients over the entire HSDP mesh; the
    unmanaged readout must be averaged with the same divisor or its updates
    would drift from the backbone by a factor of the world size.
    """
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if group is None:
        if not dist.is_initialized():
            raise ValueError("Gradient synchronization requires an initialized process group")
        group = dist.group.WORLD
    synced = []
    for name, param in model.named_parameters():
        if isinstance(param, DTensor) or param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
        param.grad.div_(world_size)
        synced.append(name)
    return synced


def clip_gradients(
    model: nn.Module, max_norm: float, *, world_size: int, group: Any = None,
) -> float:
    """Clip the true global gradient norm across managed and dense parameters.

    Managed shards exist once per (shard, replica) pair, so each rank's local
    squared norm is divided by its replicate factor before the world all-reduce;
    dense replicated gradients are divided by the world size. A trainable
    parameter without a gradient is a schedule bug and fails loudly.
    """
    if type(max_norm) not in (int, float) or not 0 < max_norm:
        raise ValueError("max_norm must be a positive finite number")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if group is None:
        if not dist.is_initialized():
            raise ValueError("Gradient clipping requires an initialized process group")
        group = dist.group.WORLD
    local_sq = None
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            if param.requires_grad:
                raise ValueError(f"Trainable parameter {name!r} has no gradient after backward")
            continue
        if isinstance(grad, DTensor):
            value = grad.to_local().double().pow(2).sum() / replicate_size(grad)
        else:
            value = grad.double().pow(2).sum() / world_size
        local_sq = value if local_sq is None else local_sq + value
    if local_sq is None:
        raise ValueError("No gradients were produced for clipping")
    dist.all_reduce(local_sq, op=dist.ReduceOp.SUM, group=group)
    total = float(local_sq.sqrt().item())
    if not total == total or total == float("inf"):
        raise ValueError("Nonfinite global gradient norm")
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)
    return total
