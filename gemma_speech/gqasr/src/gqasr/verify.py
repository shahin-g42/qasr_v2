import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from .bootstrap import digest
from .modeling import GQASRForSpeechRecognition
from .processing import GQASRProcessor


def verify_base(directory):
    torch.set_num_threads(4)
    directory = Path(directory)
    checksums = json.loads((directory / "checksums.json").read_text())
    for name, expected in checksums.items():
        if digest(directory / name) != expected:
            raise ValueError(f"checksum mismatch: {name}")
    with safe_open(directory / "model.safetensors", framework="pt") as f:
        keys = list(f.keys())
        if not all(k.startswith("gqasr_") for k in keys):
            raise ValueError("non-GQASR tensor namespace")
    model = GQASRForSpeechRecognition.from_pretrained(directory)
    processor = GQASRProcessor(directory)
    classes = sorted({type(m).__name__ for m in model.modules()})
    if any("gemma" in c.lower() or "qwen" in c.lower() for c in classes):
        raise ValueError("upstream class identifier remains in the public module tree")
    wave = (0.05 * np.sin(np.arange(16000) / 16000 * 440 * 2 * np.pi)).astype("float32")
    batch = processor.batch([wave], ["مرحبا بالعالم"], ["ar"])
    loss = model(**batch)["loss"]
    loss.backward()
    norm = float(model.gqasr_bridge[1].weight.grad.norm())
    assert torch.isfinite(loss) and np.isfinite(norm) and norm > 0
    model.eval()
    predicted = model.generate_transcripts(
        **{k: v for k, v in batch.items() if k not in {"targets", "target_mask"}}, max_new_tokens=4
    )
    return {
        "kind": "synthetic offline artifact verification, not speech quality evaluation",
        "model_type": model.config.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "tensor_count": len(keys),
        "all_tensor_keys_use_gqasr": True,
        "core_classes": classes,
        "decoder_embeddings_tied": model.gqasr_decoder.get_input_embeddings().weight
        is model.gqasr_decoder.get_output_embeddings().weight,
        "finite_loss": float(loss.detach()),
        "bridge_gradient_norm": norm,
        "generated_tokens": predicted.shape[1],
        "checksums_verified": True,
        "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__,
    }
