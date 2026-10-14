"""Offline integration verification, not a voice-quality evaluation."""

import json
from pathlib import Path

import numpy as np
import torch
from gqasr.bootstrap import digest
from safetensors import safe_open

from .base import GQTTSBase
from .speaker import audio_frames


def verify_base(model, device="cpu"):
    torch.set_num_threads(4)
    directory = Path(model)
    for name, expected in json.loads((directory / "checksums.json").read_text()).items():
        if digest(directory / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    with safe_open(directory / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
    if not keys or not all(k.startswith("gqtts_") for k in keys):
        raise ValueError("unexpected GQTTS tensor namespace")
    if any("speaker_encoder" in k for k in keys):
        raise ValueError("original speaker encoder remains in GQTTS")
    base = GQTTSBase.from_pretrained(directory, device)
    base.requires_grad_(False)
    base.gqtts_speaker_adapter.requires_grad_(True)
    wave16 = (0.05 * np.sin(np.arange(16000) * 2 * np.pi * 440 / 16000)).astype("float32")
    wave24 = (0.05 * np.sin(np.arange(24000) * 2 * np.pi * 440 / 24000)).astype("float32")
    with torch.no_grad():
        frames = audio_frames(base.gqtts_acoustic, base.frontend, wave16, device)
    valid = torch.ones(1, len(frames), device=device, dtype=torch.bool)
    vector = base.gqtts_speaker_adapter(frames[None], valid)
    loss = vector.square().mean()
    loss.backward()
    grad = float(base.gqtts_speaker_adapter.project[1].weight.grad.norm())
    if not torch.isfinite(vector).all() or not np.isfinite(grad) or grad <= 0:
        raise FloatingPointError("invalid speaker adapter output or gradient")
    with torch.no_grad():
        codec = base.api.model.speech_tokenizer
        codes = codec.encode(wave24, sr=24000).audio_codes[0]
        reconstructed, sample_rate = codec.decode([{"audio_codes": codes}])
        waves, generated_rate = base.synthesize(
            wave16, "Hello.", "English", max_new_tokens=4, do_sample=False, subtalker_dosample=False
        )
    if not len(reconstructed[0]) or not np.isfinite(reconstructed[0]).all():
        raise FloatingPointError("invalid codec reconstruction")
    if not len(waves[0]) or not np.isfinite(waves[0]).all():
        raise FloatingPointError("invalid short generation probe")
    return dict(
        kind="synthetic offline artifact verification, not speech quality evaluation",
        model_type="gqtts",
        parameters=sum(p.numel() for p in base.parameters()),
        tensor_count=len(keys),
        all_tensor_keys_use_gqtts=True,
        original_speaker_encoder_removed=True,
        checksums_verified=True,
        speaker_shape=list(vector.shape),
        adapter_gradient_norm=grad,
        codec_shape=list(codes.shape),
        codec_reconstruction_samples=len(reconstructed[0]),
        codec_sample_rate=sample_rate,
        generated_samples=len(waves[0]),
        sample_rate=generated_rate,
        speaker_adapter_trained=False,
        torch=str(torch.__version__),
        transformers=__import__("transformers").__version__,
    )
