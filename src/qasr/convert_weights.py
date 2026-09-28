from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import (
    AutoConfig,
    AutoProcessor,
    GenerationConfig,
)

from .configuration import QASRConfig
from .feature_extraction import QASRFeatureExtractor
from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor

LOGGER = logging.getLogger("qasr.convert")
DEFAULT_ENCODER = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_QWEN = "audarai/Audar-ASR-V1.2-Turbo"
DEFAULT_OUTPUT = "/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/output/initial"


def _load_exact(module: torch.nn.Module, state_dict: dict[str, torch.Tensor], name: str) -> None:
    incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"{name} transfer was incomplete; missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    loaded_state = module.state_dict()
    unequal = [key for key, tensor in state_dict.items() if not torch.equal(loaded_state[key], tensor)]
    if unequal:
        preview = ", ".join(unequal[:10])
        raise RuntimeError(f"{name} transfer changed {len(unequal)} tensor(s): {preview}")


def _resolve_safetensors(name_or_path: str) -> Path | list[Path]:
    """Resolve model weights to local path(s). Handles sharded checkpoints."""
    local_path = Path(name_or_path).expanduser()

    if local_path.is_file():
        return local_path

    if local_path.is_dir():
        # Check for sharded checkpoint
        index_file = local_path / "model.safetensors.index.json"
        if index_file.is_file():
            return _resolve_sharded_local(local_path, index_file)
        # Single file
        weights = local_path / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Expected model.safetensors or index at {local_path}")
        return weights

    # Hub model — download and check for sharding
    try:
        index_path = Path(hf_hub_download(name_or_path, filename="model.safetensors.index.json"))
        return _resolve_sharded_local(index_path.parent, index_path)
    except Exception:
        pass

    return Path(hf_hub_download(name_or_path, filename="model.safetensors"))


def _resolve_sharded_local(model_dir: Path, index_file: Path) -> list[Path]:
    """Resolve all shard files from a sharded checkpoint index."""
    with index_file.open() as f:
        index = json.load(f)
    shard_files = sorted(set(index["weight_map"].values()))
    paths = [model_dir / shard for shard in shard_files]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing shard file: {p}")
    LOGGER.info("Found sharded checkpoint: %d shards", len(paths))
    return paths


def _load_prefixed_state(weights_path: Path | list[Path], prefix: str) -> dict[str, torch.Tensor]:
    """Load tensors with a given prefix from single or sharded safetensors."""
    state: dict[str, torch.Tensor] = {}

    paths = weights_path if isinstance(weights_path, list) else [weights_path]
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            # safe_open is not iterable in this safetensors version; keys() is.
            for key in checkpoint.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = checkpoint.get_tensor(key)

    if not state:
        raise KeyError(f"No tensors beginning with {prefix!r} in checkpoint")
    return state


# Key renames: NeMo Fast-Conformer naming → HuggingFace Cohere encoder naming
# Both Cohere Transcribe and its Conformer variants use NeMo's Fast-Conformer,
# so the same renames apply.
_ENCODER_KEY_RENAMES = (
    (r"^pre_encode\.conv\.", "subsampling.layers."),
    (r"^pre_encode\.out\.", "subsampling.linear."),
    (r"\.self_attn\.linear_q(?=\.|$)", ".self_attn.q_proj"),
    (r"\.self_attn\.linear_k(?=\.|$)", ".self_attn.k_proj"),
    (r"\.self_attn\.linear_v(?=\.|$)", ".self_attn.v_proj"),
    (r"\.self_attn\.linear_out(?=\.|$)", ".self_attn.o_proj"),
    (r"\.self_attn\.linear_pos(?=\.|$)", ".self_attn.relative_k_proj"),
    (r"\.self_attn\.pos_bias_u(?=\.|$)", ".self_attn.bias_u"),
    (r"\.self_attn\.pos_bias_v(?=\.|$)", ".self_attn.bias_v"),
    (r"\.conv\.batch_norm(?=\.|$)", ".conv.norm"),
)


def _convert_encoder_state(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename NeMo Fast-Conformer tensors to the QASR encoder namespace.

    Works for all Cohere Conformer encoder variants since they share
    NeMo's Fast-Conformer implementation with identical key naming.
    """
    converted: dict[str, torch.Tensor] = {}
    for source_key, tensor in state_dict.items():
        target_key = source_key
        for pattern, replacement in _ENCODER_KEY_RENAMES:
            target_key = re.sub(pattern, replacement, target_key)
        if target_key in converted:
            raise RuntimeError(
                f"Encoder conversion mapped multiple tensors to {target_key!r}"
            )
        converted[target_key] = tensor
    return converted


# Possible encoder prefixes in different model checkpoints
_ENCODER_PREFIXES = (
    "model.encoder.",       # Cohere Transcribe (HF format)
    "encoder.",             # Alternative flat format
    "audio_tower.",         # Some multimodal models
)


def _load_encoder_weights(encoder_name_or_path: str) -> dict[str, torch.Tensor]:
    """Load and convert encoder weights, trying multiple prefix conventions."""
    weights_path = _resolve_safetensors(encoder_name_or_path)

    for prefix in _ENCODER_PREFIXES:
        try:
            raw_state = _load_prefixed_state(weights_path, prefix)
            LOGGER.info(
                "Found %d encoder tensors with prefix %r", len(raw_state), prefix
            )
            return _convert_encoder_state(raw_state)
        except KeyError:
            continue

    # If no prefix worked, list available top-level prefixes for debugging
    paths = weights_path if isinstance(weights_path, list) else [weights_path]
    top_keys: set[str] = set()
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():  # noqa: SIM118
                top_keys.add(key.split(".")[0] + "." + (key.split(".")[1] if "." in key else ""))
    raise KeyError(
        f"Could not find encoder tensors in {encoder_name_or_path}. "
        f"Tried prefixes: {_ENCODER_PREFIXES}. "
        f"Available top-level keys: {sorted(top_keys)[:20]}"
    )


def _encoder_feature_size(audio_config) -> int:
    """Mel-bin count for the feature extractor, robust to config schema.

    ``ParakeetEncoderConfig`` does not define ``feature_size`` — the attribute
    only exists when the source repo's config JSON happens to carry it as an
    extra key and the installed transformers preserves unknown keys through
    the sub-config round-trip. Reading it directly crashed conversion with
    AttributeError. ``num_mel_bins`` is class-defined and always present (and
    for this log-mel front end the two are the same number).
    """
    feature_size = getattr(audio_config, "feature_size", None)
    if feature_size is not None:
        return int(feature_size)
    return int(audio_config.num_mel_bins)


def convert_components(
    *,
    encoder_name_or_path: str,
    qwen_name_or_path: str,
    output_dir: str | Path,
    seed: int = 42,
    verify_reload: bool = False,
) -> Path:
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty conversion output: {output_path}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    dtype = torch.bfloat16

    # --- Load configs ---
    encoder_source_config = AutoConfig.from_pretrained(encoder_name_or_path)
    qwen_config_source = AutoConfig.from_pretrained(qwen_name_or_path)
    encoder_config = getattr(encoder_source_config, "encoder_config", encoder_source_config)

    # Extract text config from various multimodal config structures
    text_config = getattr(qwen_config_source, "text_config", None)
    if text_config is None:
        # Try nested structures (e.g., thinker_config.text_config)
        thinker_cfg = getattr(qwen_config_source, "thinker_config", None)
        if thinker_cfg is not None:
            text_config = getattr(thinker_cfg, "text_config", thinker_cfg)
        else:
            # Assume the config itself is the text config
            text_config = qwen_config_source

    # Extract special token IDs with fallbacks
    audio_token_id = getattr(qwen_config_source, "audio_token_id", None)
    if audio_token_id is None:
        audio_token_id = getattr(qwen_config_source, "audio_token_index", 151646)
    timestamp_token_id = getattr(qwen_config_source, "timestamp_token_id", None)
    if timestamp_token_id is None:
        timestamp_token_id = getattr(qwen_config_source, "timestamp_token_index", 151647)

    config = QASRConfig(
        audio_config=encoder_config.to_dict(),
        text_config=text_config.to_dict(),
        audio_token_id=int(audio_token_id),
        timestamp_token_id=int(timestamp_token_id),
        pad_token_id=int(qwen_config_source.pad_token_id),
        eos_token_id=qwen_config_source.eos_token_id,
        tie_word_embeddings=True,
        dtype="bfloat16",
        projector_hidden_act="gelu",
    )

    # --- Instantiate empty QASR model ---
    old_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = QASRForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(old_default_dtype)

    # --- Load encoder weights (Cohere Transcribe) ---
    LOGGER.info("Loading encoder from %s", encoder_name_or_path)
    encoder_state = _load_encoder_weights(encoder_name_or_path)
    LOGGER.info("Encoder: %d tensors after key conversion", len(encoder_state))
    _load_exact(model.model.audio_tower, encoder_state, "encoder")
    del encoder_state
    gc.collect()

    # --- Load decoder weights (Audar-ASR-V1.2-Turbo) ---
    LOGGER.info("Loading language model from %s", qwen_name_or_path)
    qwen_weights = _resolve_safetensors(qwen_name_or_path)

    # Try multiple prefix conventions for decoder weights
    _DECODER_PREFIXES = (
        "thinker.model.",         # Audar-ASR-V1.2-Turbo format
        "model.language_model.",  # Multimodal format (Qwen3-ASR)
        "model.",                 # Standard Qwen3 format
        "language_model.",        # Alternative flat format
    )
    language_state = None
    for prefix in _DECODER_PREFIXES:
        try:
            language_state = _load_prefixed_state(qwen_weights, prefix)
            LOGGER.info("Found %d decoder tensors with prefix %r", len(language_state), prefix)
            break
        except KeyError:
            continue
    if language_state is None:
        raise KeyError(f"Could not find decoder tensors in {qwen_name_or_path}. Tried: {_DECODER_PREFIXES}")
    _load_exact(model.model.language_model, language_state, "language model")
    del language_state

    # Try multiple lm_head prefixes
    lm_head_state = None
    for lm_prefix in ("thinker.lm_head.", "lm_head."):
        try:
            lm_head_state = _load_prefixed_state(qwen_weights, lm_prefix)
            LOGGER.info("Found lm_head with prefix %r", lm_prefix)
            break
        except KeyError:
            continue
    if lm_head_state is not None:
        _load_exact(model.lm_head, lm_head_state, "LM head")
        del lm_head_state
    else:
        LOGGER.info("Decoder checkpoint stores tied LM head via embed_tokens.weight")

    model.generation_config = GenerationConfig.from_pretrained(qwen_name_or_path)
    model.tie_weights()
    if model.lm_head.weight is not model.get_input_embeddings().weight:
        raise RuntimeError("Input embeddings and LM head are not tied after conversion")
    gc.collect()

    # --- Build processor ---
    LOGGER.info("Combining QASR features with the decoder tokenizer and chat template")
    qwen_processor = AutoProcessor.from_pretrained(qwen_name_or_path)

    # Create QASRFeatureExtractor explicitly (not from_pretrained which loads the native one)
    feature_extractor = QASRFeatureExtractor(
        feature_size=_encoder_feature_size(config.audio_config),
        sampling_rate=16000,
        hop_length=160,
        n_fft=512,
        win_length=400,
        preemphasis=0.97,
        padding_value=0.0,
        dither=1e-5,
        max_audio_clip_s=35.0,
        overlap_chunk_second=5.0,
    )

    # Extract processor attributes with fallbacks
    timestamp_segment_time = getattr(qwen_processor, "timestamp_segment_time", 0.04)
    chat_template = getattr(qwen_processor, "chat_template", None)

    processor = QASRProcessor(
        feature_extractor=feature_extractor,
        tokenizer=qwen_processor.tokenizer,
        chat_template=chat_template,
        timestamp_segment_time=timestamp_segment_time,
        subsampling_factor=config.audio_config.subsampling_factor,
        subsampling_conv_kernel_size=config.audio_config.subsampling_conv_kernel_size,
        subsampling_conv_stride=config.audio_config.subsampling_conv_stride,
    )

    # --- Save ---
    LOGGER.info("Saving initial QASR checkpoint to %s", output_path)
    model.save_pretrained(output_path, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(output_path)

    # --- Verify round-trip ---
    saved_config = QASRConfig.from_pretrained(output_path)
    if saved_config.audio_config.hidden_size != 1280:
        raise RuntimeError("Saved QASR encoder config did not round-trip")
    LOGGER.info("Config round-trip verified (hidden_size=%d)", saved_config.audio_config.hidden_size)

    if verify_reload:
        del model
        gc.collect()
        reloaded = QASRForConditionalGeneration.from_pretrained(
            output_path, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa",
        )
        if reloaded.lm_head.weight is not reloaded.get_input_embeddings().weight:
            raise RuntimeError("Reloaded model lost tied embeddings")
        LOGGER.info("Full QASR checkpoint reload verification passed")

    total_params = sum(p.numel() for p in model.parameters()) if not verify_reload else "verified"
    LOGGER.info("Conversion complete: %s (%s params)", output_path, total_params)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a QASR checkpoint from a Conformer encoder + LLM decoder."
    )
    parser.add_argument(
        "--encoder", default=DEFAULT_ENCODER,
        help=f"Encoder checkpoint (default: {DEFAULT_ENCODER})",
    )
    parser.add_argument(
        "--qwen", default=DEFAULT_QWEN,
        help=f"Decoder/LLM checkpoint (default: {DEFAULT_QWEN})",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-reload", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    convert_components(
        encoder_name_or_path=args.encoder,
        qwen_name_or_path=args.qwen,
        output_dir=args.output_dir,
        seed=args.seed,
        verify_reload=args.verify_reload,
    )
    # One-shot CLI hardening: the HF stack (hf_xet download threads in
    # particular) can leave non-daemon threads that stall interpreter
    # shutdown long after a successful conversion — the process prints
    # "Conversion complete" and then never returns to the shell. Everything
    # is saved, closed, and verified by this point, so exit unconditionally.
    # Failures still propagate normally: an exception above skips this line.
    logging.shutdown()
    os._exit(0)


if __name__ == "__main__":
    main()
