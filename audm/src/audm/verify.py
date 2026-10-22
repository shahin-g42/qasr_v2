import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .export import sha256
from .modeling import AUDForConditionalGeneration
from .processing import AUDProcessor


def verify(directory, device="cpu", smoke=True):
    directory = Path(directory)
    checksum_file = directory / "checksums.json"
    if checksum_file.is_file():
        for name, expected in json.loads(checksum_file.read_text()).items():
            if sha256(directory / name) != expected:
                raise ValueError(f"Checksum mismatch: {name}")
    with safe_open(directory / "model.safetensors", framework="pt") as handle:
        if not all(k.startswith("aud_") for k in handle.keys()):
            raise ValueError("Weights outside the AUD namespace")
    model, info = AUDForConditionalGeneration.from_pretrained(
        directory, local_files_only=True, dtype="auto", output_loading_info=True
    )
    if any(
        info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Non-strict reload: {info}")
    processor = AUDProcessor.from_pretrained(directory)
    model.to(device).eval()
    report = {
        "variant": model.config.variant,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checksums": "verified" if checksum_file.is_file() else "not present (training checkpoint)",
        "offline_reload": True,
        "requires_alignment": model.config.requires_alignment,
        "architecture": model.architecture_report(),
    }
    if smoke:
        # Unequal waveforms exercise independent lengths and placeholder boundaries.
        rng = np.random.default_rng(42)
        waves = [rng.normal(0, 0.01, n).astype(np.float32) for n in (3200, 4960)]
        batch = processor.apply_transcription_request(waves, language=["en", "en"])
        batch = {k: v.to(device) for k, v in batch.items() if k in processor.model_input_names}
        with torch.no_grad():
            logits = model(**batch).logits
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite forward logits")
            tokens = model.generate(**batch, max_new_tokens=2, do_sample=False, use_cache=True)
        report["smoke"] = {
            "logits_shape": list(logits.shape),
            "generated_shape": list(tokens.shape),
            "device": device,
            "quality_evaluation": False,
        }
    return report


def distributed_check(expected_world_size):
    import os

    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    try:
        if dist.get_world_size() != expected_world_size:
            raise ValueError(f"Expected {expected_world_size} ranks, got {dist.get_world_size()}")
        value = torch.tensor([dist.get_rank() + 1.0], device=f"cuda:{local_rank}")
        dist.all_reduce(value)
        expected = expected_world_size * (expected_world_size + 1) / 2
        if value.item() != expected:
            raise ValueError("NCCL all-reduce produced an incorrect result")
        if dist.get_rank() == 0:
            return {"nccl": "passed", "world_size": expected_world_size, "sum": expected}
        return None
    finally:
        dist.destroy_process_group()
