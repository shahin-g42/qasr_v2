"""Strict local Qwen3_5 decision construction and CPU-to-sharded state loading."""

from __future__ import annotations

import json
import math
import re
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from jev_contracts import (
    artifact_path,
    atomic_write_json,
    canonical_json,
    safe_relative_path,
    sha256_file,
    write_manifest,
)

from .codes import (
    AUTOJEV_MODEL,
    AUTOJEV_REVISION,
    AUTOJEV_SOURCE_REVISION,
    BASE_MODEL,
    BASE_REVISION,
    MAX_OPTIONS,
    discover_codes,
    verify_codes,
)
from .processing import MAX_LENGTH, PROMPT_VERSION, PreparedBatch, prepare_batch, validate_batch

PINS = {
    "autojev_revision": AUTOJEV_REVISION,
    "autojev_source_revision": AUTOJEV_SOURCE_REVISION,
    "base_revision": BASE_REVISION,
}
_JSON_LIMIT = 16 * 1024 * 1024


def _positive(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number, not bool")
    return float(value)


def _pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(path: Path) -> dict:
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Expected a regular JSON file: {path.name}")
    if path.stat().st_size > _JSON_LIMIT:
        raise ValueError(f"JSON file exceeds {_JSON_LIMIT} bytes: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs)
    if type(value) is not dict:
        raise ValueError(f"Expected a JSON object: {path.name}")
    canonical_json(value)
    return value


class _Snapshot:
    """Local, nonsymlink snapshot. Check each consumed file against any manifest.

    The caller must trust and keep the local snapshot immutable during loading;
    hashes detect corruption, not an attacker replacing both files and manifest.
    Meta construction checks configuration/processor files, never weight bytes.
    """

    def __init__(self, root: Path):
        self.root = Path(root).absolute()
        self.checked: set[str] = set()
        self.files: dict | None = None
        self.manifest: dict = {}
        for name in ("manifest.json", "release-manifest.json"):
            path = artifact_path(self.root, name)
            if path.exists():
                manifest = _json(path)
                if (type(manifest.get("schema_version")) is not int
                        or manifest["schema_version"] != 1):
                    raise ValueError(f"Unsupported manifest schema: {name}")
                files = manifest.get("files")
                if type(files) is not dict or not files:
                    raise ValueError(f"{name} must contain a nonempty files mapping")
                if self.files is not None:
                    raise ValueError("Ambiguous snapshot: two different manifest formats")
                self.files, self.manifest = files, manifest
                for relative, info in files.items():
                    safe_relative_path(relative)
                    if (type(info) is not dict or type(info.get("bytes")) is not int
                            or info["bytes"] < 0 or not isinstance(info.get("sha256"), str)
                            or re.fullmatch(r"[0-9a-f]{64}", info["sha256"]) is None):
                        raise ValueError(f"Invalid manifest entry: {relative}")
                if name == "release-manifest.json" and (
                    manifest.get("model_repository") != AUTOJEV_MODEL
                    or manifest.get("source_commit") != AUTOJEV_SOURCE_REVISION
                    or manifest.get("base_revision") != BASE_REVISION
                ):
                    raise ValueError("Release manifest does not match the audited source pins")

    def path(self, relative: str) -> Path:
        path = artifact_path(self.root, relative)
        if not stat.S_ISREG(path.stat().st_mode):
            raise ValueError(f"Expected a regular snapshot file: {relative}")
        if self.files is not None and relative not in self.checked:
            info = self.files.get(relative)
            if info is None:
                raise ValueError(f"Consumed file is absent from the manifest: {relative}")
            if path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
                raise ValueError(f"Snapshot integrity mismatch: {relative}")
            self.checked.add(relative)
        return path

    def json(self, relative: str) -> dict:
        return _json(self.path(relative))

    def exists(self, relative: str) -> bool:
        return artifact_path(self.root, relative).exists()

    def processor_files(self) -> None:
        # Transformers consumes these local text assets; forbid hidden symlink escapes.
        for path in self.root.iterdir():
            if path.suffix in {".json", ".jinja", ".txt", ".model"}:
                if path.name in {"manifest.json", "release-manifest.json"}:
                    continue
                self.path(path.name)
        templates = artifact_path(self.root, "chat_templates")
        if templates.exists():
            for path in templates.iterdir():
                self.path(f"chat_templates/{path.name}")


def _runtime():
    try:
        from transformers import AutoProcessor, Qwen3_5Config, Qwen3_5Model
    except ImportError as exc:
        raise ImportError("Install jev-core[train] with a qualified Qwen3_5 runtime") from exc
    return AutoProcessor, Qwen3_5Config, Qwen3_5Model


def _configuration(raw: dict, config_class: Any):
    if raw.get("model_type") != "qwen3_5" or not isinstance(raw.get("text_config"), dict):
        raise ValueError("Only the pinned multimodal Qwen3_5 backbone architecture is supported")
    if not isinstance(raw.get("vision_config"), dict) or raw.get("language_model_only", False):
        raise ValueError("The retained Qwen3_5 vision backbone is required")
    # Preserve architecture fields. Do not filter unknown kwargs to make a runtime appear to work.
    # Transformers' documented legacy qwen3_5 vision model_type remapping is the only rename.
    raw = json.loads(json.dumps(raw))
    if raw["vision_config"].get("model_type") == "qwen3_5":
        raw["vision_config"]["model_type"] = "qwen3_5_vision"
    try:
        config = config_class.from_dict(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "This runtime cannot represent the pinned Qwen3_5 config; qualify the candidate "
            "Transformers version rather than dropping unknown architecture fields"
        ) from exc
    config._attn_implementation = "sdpa"
    config.text_config.use_cache = False
    return config


class DecisionModel(nn.Module):
    def __init__(
        self, backbone: nn.Module, readout: nn.Linear, processor: Any,
        codes: Sequence[str], token_ids: Sequence[int], *, metadata: dict | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.codes, self.token_ids = verify_codes(processor.tokenizer, codes, token_ids)
        if not isinstance(readout, nn.Linear) or readout.out_features != MAX_OPTIONS:
            raise ValueError("readout must be Linear(hidden_size, 255, bias=False)")
        if readout.bias is not None:
            raise ValueError("The decision readout must not have a bias")
        self.backbone, self.readout, self.processor = backbone, readout, processor
        self.processor.tokenizer.padding_side = "left"
        self.metadata = json.loads(canonical_json(metadata or {}))
        self.metadata.setdefault("pins", dict(PINS))
        self.metadata.setdefault("prompt_version", PROMPT_VERSION)
        self.metadata.setdefault("max_length", MAX_LENGTH)
        self.temperature = _positive(temperature, "temperature")

    def freeze_vision(self) -> int:
        visual = getattr(self.backbone, "visual", None)
        if not isinstance(visual, nn.Module):
            raise ValueError("Expected Qwen3_5 backbone.visual; refusing to guess vision ownership")
        visual.requires_grad_(False)
        self.metadata["vision_frozen"] = True
        return sum(parameter.numel() for parameter in visual.parameters())

    def prepare(self, rows: Sequence[dict], max_length: int = MAX_LENGTH) -> PreparedBatch:
        # DTensor.device is the rank-local device; never use the processor's device or gather it.
        device = self.readout.weight.device
        if device.type == "meta":
            raise ValueError("Materialize/shard the meta model before preparing device inputs")
        return prepare_batch(self.processor, rows, self.codes, max_length, device=device)

    def forward(self, batch: PreparedBatch) -> torch.Tensor:
        validate_batch(batch)
        hidden = self.backbone(**batch.inputs, use_cache=False).last_hidden_state
        if hidden.ndim != 3 or hidden.shape[:2] != batch.inputs["input_ids"].shape:
            raise ValueError("Backbone must return [B, L, H] last_hidden_state")
        logits = self.readout(hidden[:, -1]).float()
        if (logits.shape != (len(batch.counts), MAX_OPTIONS)
                or not bool(torch.isfinite(logits).all())):
            raise ValueError("Decision logits must be finite [B,255]")
        mask = torch.arange(MAX_OPTIONS, device=logits.device)[None] >= torch.tensor(
            batch.counts, device=logits.device,
        )[:, None]
        return logits.masked_fill(mask, -1e9)

    @torch.inference_mode()
    def predict(
        self, rows: Sequence[dict], batch_size: int = 8, temperature: float | None = None,
    ) -> list[list[float]]:
        scale = _positive(self.temperature if temperature is None else temperature, "temperature")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        result = []
        try:
            for start in range(0, len(rows), batch_size):
                batch = self.prepare(rows[start:start + batch_size])
                logits = self(batch)
                for values, count in zip(logits, batch.counts, strict=True):
                    # Slice first: a huge valid temperature must not give padding probability mass.
                    valid = values[:count].double()
                    centered = valid - valid.max()
                    result.append((centered / scale).softmax(-1).cpu().tolist())
        finally:
            for module, training in modes:
                module.training = training
        return result

    def materialize_runtime_buffers(self, device: Any = "cpu") -> None:
        """Rebuild known nonpersistent RoPE buffers, including after trainer to_empty.

        Parameters are never materialized or reset here. Call after to_empty and
        before distributed state loading; these buffers are not checkpoint tensors.
        """
        for module in self.backbone.modules():
            names = module._non_persistent_buffers_set
            if not names:
                continue
            kind = type(module).__name__
            with torch.device(device):
                if kind == "Qwen3_5TextRotaryEmbedding":
                    replacement = type(module)(module.config, device=device)
                elif kind == "Qwen3_5VisionRotaryEmbedding":
                    replacement = type(module)(module.dim, module.theta)
                else:
                    if any(t is not None and t.is_meta for n, t in module._buffers.items()
                           if n in names):
                        raise ValueError(f"Unqualified nonpersistent buffer initialization: {kind}")
                    continue
            for name in names:
                if name not in replacement._buffers:
                    raise ValueError(f"Runtime buffer layout changed: {kind}.{name}")
                module._buffers[name] = replacement._buffers[name]

    def save_pretrained(self, destination: str | Path) -> None:
        """Write raw components to a NEW directory; manifest is the completion marker.

        No calibration, dtype conversion, source rewrite, license inference, or
        distributed gathering is performed. Export orchestration owns those steps.
        """
        verify_codes(self.processor.tokenizer, self.codes, self.token_ids)
        scale = _positive(self.temperature, "temperature")
        for tensor in self.state_dict().values():
            if tensor.is_meta or hasattr(tensor, "to_local"):
                raise ValueError("Save requires materialized full tensors, not meta/DTensor shards")
        metadata = json.loads(canonical_json(self.metadata))
        destination = Path(destination).absolute()
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifact_path(destination.parent, destination.name)
        destination.mkdir(exist_ok=False)
        # A failed save intentionally leaves an incomplete directory that cannot be reused.
        atomic_write_json(destination / "jev_components.json", {"format_version": 1})
        self.backbone.save_pretrained(
            str(destination), safe_serialization=True, max_shard_size="5GB",
        )
        save_file({"weight": self.readout.weight.detach().cpu().contiguous()},
                  str(destination / "readout.safetensors"))
        self.processor.save_pretrained(str(destination))
        atomic_write_json(destination / "decision_config.json", {
            "format_version": 1, "base_model": BASE_MODEL, "revision": BASE_REVISION,
            "codes": list(self.codes), "token_ids": list(self.token_ids), "temperature": scale,
            "metadata": metadata,
        })
        files = [str(path.relative_to(destination)) for path in destination.rglob("*")
                 if path.is_file()]
        write_manifest(destination, files, {"kind": "jev-model-components", "pins": dict(PINS)})


def _build(snapshot: _Snapshot, initialization: str) -> DecisionModel:
    if initialization not in {"autojev", "fresh"}:
        raise ValueError("initialization must be 'autojev' or 'fresh'")
    if snapshot.exists("jev_components.json") and not snapshot.exists("manifest.json"):
        raise ValueError("Incomplete component snapshot: completion manifest is missing")
    raw = snapshot.json("config.json")
    saved = (
        snapshot.json("decision_config.json") if snapshot.exists("decision_config.json") else None
    )
    if (initialization == "autojev") != (saved is not None):
        raise ValueError("autojev requires a decision artifact; fresh requires a full base")
    expected_pin = AUTOJEV_REVISION if initialization == "autojev" else BASE_REVISION
    if re.fullmatch(r"[0-9a-f]{40}", snapshot.root.name) and snapshot.root.name != expected_pin:
        raise ValueError("Snapshot directory revision does not match the initialization pin")
    if saved:
        if (type(saved.get("format_version")) is not int or saved["format_version"] != 1
                or saved.get("base_model") != BASE_MODEL or saved.get("revision") != BASE_REVISION):
            raise ValueError("Decision config format/base revision does not match the audited pin")
        metadata = saved.get("metadata", {})
        if type(metadata) is not dict:
            raise ValueError("Decision metadata must be a dictionary")
        if (metadata.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION
                or metadata.get("pins", PINS) != PINS
                or metadata.get("max_length", MAX_LENGTH) != MAX_LENGTH):
            raise ValueError("Saved prompt, input limit, or revision mapping differs")
    else:
        metadata = {}
        revision = raw.get("_commit_hash", snapshot.root.name)
        if revision != BASE_REVISION:
            raise ValueError(
                "Fresh loading requires the pinned HF snapshots/<revision> directory or an "
                "explicit matching _commit_hash in the trusted local config.json"
            )
    architectures = raw.get("architectures")
    expected_arch = "Qwen3_5Model" if saved else "Qwen3_5ForConditionalGeneration"
    if architectures != [expected_arch]:
        raise ValueError(f"Expected exactly the {expected_arch} architecture")
    processor_class, config_class, backbone_class = _runtime()
    snapshot.processor_files()
    processor = processor_class.from_pretrained(
        str(snapshot.root), local_files_only=True, trust_remote_code=False,
    )
    codes, ids = discover_codes(processor.tokenizer)
    if saved:
        verify_codes(processor.tokenizer, saved["codes"], saved["token_ids"])
    config = _configuration(raw, config_class)
    with torch.device("meta"):
        backbone = backbone_class(config)
        readout = nn.Linear(config.text_config.hidden_size, MAX_OPTIONS, bias=False)
    metadata = dict(metadata)
    metadata.update({
        "initialization": initialization, "pins": dict(PINS),
        "source_snapshot": str(snapshot.root), "runtime_certified": False,
        "state_mapping": "qwen3_5-model-prefix-drop-single-mtp-lm-row-readout-v1",
        "snapshot_integrity": "manifest" if snapshot.files is not None else "caller-trusted-local",
    })
    model = DecisionModel(backbone, readout, processor, codes, ids, metadata=metadata,
                          temperature=saved["temperature"] if saved else 1.0)
    model.materialize_runtime_buffers("cpu")
    model.freeze_vision()
    model.eval()
    return model


def build_meta_model(snapshot: Path, initialization: str = "autojev") -> DecisionModel:
    """Build only parameter metadata plus small CPU runtime buffers, never weights."""
    return _build(_Snapshot(snapshot), initialization)


def _mtp_shapes(config: Any) -> dict[str, tuple[int, ...]]:
    """Exact unused single-MTP layout in the pinned full Qwen artifact, not a wildcard."""
    h, intermediate = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    shapes = {
        "fc.weight": (h, 2 * h), "norm.weight": (h,),
        "pre_fc_norm_embedding.weight": (h,), "pre_fc_norm_hidden.weight": (h,),
        "layers.0.input_layernorm.weight": (h,), "layers.0.post_attention_layernorm.weight": (h,),
        "layers.0.mlp.down_proj.weight": (h, intermediate),
        "layers.0.mlp.gate_proj.weight": (intermediate, h),
        "layers.0.mlp.up_proj.weight": (intermediate, h),
        "layers.0.self_attn.k_norm.weight": (config.head_dim,),
        "layers.0.self_attn.q_norm.weight": (config.head_dim,),
        "layers.0.self_attn.k_proj.weight": (kv, h),
        "layers.0.self_attn.v_proj.weight": (kv, h),
        "layers.0.self_attn.q_proj.weight": (2 * q, h),
        "layers.0.self_attn.o_proj.weight": (h, q),
    }
    return {"mtp." + key: value for key, value in shapes.items()}


def _state(
    snapshot: _Snapshot, model: DecisionModel, initialization: str,
) -> dict[str, torch.Tensor]:
    expected = {key: tuple(value.shape) for key, value in model.backbone.state_dict().items()}
    prefix = "model." if initialization == "fresh" else ""
    source_shapes = {prefix + key: shape for key, shape in expected.items()}
    if initialization == "fresh":
        text = model.backbone.config.text_config
        source_shapes["lm_head.weight"] = (text.vocab_size, text.hidden_size)
    index_name = "model.safetensors.index.json"
    if snapshot.exists(index_name):
        index = snapshot.json(index_name)
        weight_map = index.get("weight_map")
        if type(weight_map) is not dict or not weight_map or len(weight_map) > 100000:
            raise ValueError("Invalid/bounded weight_map required")
        paths = {}
        for key, filename in weight_map.items():
            if not key or not isinstance(filename, str) or not filename.endswith(".safetensors"):
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
            f"Checkpoint tensor names mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for key, shape in source_shapes.items():
        if actual_shapes[key] != shape:
            raise ValueError(
                f"Checkpoint tensor shape mismatch: {key}: {actual_shapes[key]} != {shape}"
            )
        if dtypes[key] not in {"F16", "BF16", "F32", "F64"}:
            raise ValueError(f"Unsupported checkpoint tensor dtype: {key}: {dtypes[key]}")
    readout = None
    if initialization == "autojev":
        readout_path = snapshot.path("readout.safetensors")
        with safe_open(str(readout_path), framework="pt", device="cpu") as f:
            if set(f.keys()) != {"weight"}:
                raise ValueError("Readout file must contain exactly weight")
            if tuple(f.get_slice("weight").get_shape()) != tuple(model.readout.weight.shape):
                raise ValueError("Readout tensor shape mismatch")
            if f.get_slice("weight").get_dtype() not in {"F16", "BF16", "F32", "F64"}:
                raise ValueError("Unsupported readout tensor dtype")
            readout = f.get_tensor("weight")
    state = {}
    for path in paths.values():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source in handle.keys():
                if source in mtp:
                    continue
                if source == "lm_head.weight":
                    # Read only the selected vocabulary rows; never instantiate a full LM head.
                    view = handle.get_slice(source)
                    readout = torch.cat(
                        [view[i:i + 1] for i in model.token_ids], dim=0,
                    ).contiguous()
                else:
                    key = source.removeprefix(prefix) if prefix else source
                    state["backbone." + key] = handle.get_tensor(source)
    if readout is None or not bool(torch.isfinite(readout).all()):
        raise ValueError("Readout must contain finite weights")
    state["readout.weight"] = readout
    if state.keys() != model.state_dict().keys():
        raise ValueError("Decision state does not exactly cover the constructed model")
    return state


def load_full_state(snapshot: Path, initialization: str = "autojev") -> dict[str, torch.Tensor]:
    """Full CPU source-dtype tensors for rank-zero FSDP2 broadcast; no GPU copies."""
    source = _Snapshot(snapshot)
    model = _build(source, initialization)
    return _state(source, model, initialization)


def load_model(
    snapshot: Path, *, initialization: str = "autojev", device: Any = "cpu",
    dtype: torch.dtype = torch.float32, gradient_checkpointing: bool = False,
) -> DecisionModel:
    if dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
        raise ValueError("Expected a floating-point model dtype")
    if torch.device(device).type == "meta":
        raise ValueError("Use build_meta_model for sharded/meta initialization")
    source = _Snapshot(snapshot)
    model = _build(source, initialization)
    state = _state(source, model, initialization)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model.to(device=device, dtype=dtype)
    model.materialize_runtime_buffers(device)
    if gradient_checkpointing:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    return model
