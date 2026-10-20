"""Native QASR inference on the same prepared clips and decoding policy as GQASR.

QASR uses its saved 128-bin frontend, Conformer and text decoder. No conversion,
reference-conditioned prompts, speculative decoding or changes to model weights.
"""
import json
import math
import re
import struct
from pathlib import Path

from .common import sha256
from .worker import Recognizer


def checkpoint_files(directory):
    root = Path(directory).resolve()
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qasr":
        raise ValueError(f"Expected a native QASR checkpoint: {root}")
    if not any((root / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError(f"Missing saved QASR processor: {root}")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (root / name).is_file():
            raise ValueError(f"Missing saved QASR tokenizer file: {root / name}")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        if (root / "model.safetensors").exists():
            raise ValueError("Ambiguous QASR checkpoint: both single-file weights and a shard index exist")
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if not weight_map:
            raise ValueError("Empty QASR safetensors shard index")
        weights = [root / name for name in sorted(set(weight_map.values()))]
        for path in weights:
            if path.parent != root or path.suffix != ".safetensors":
                raise ValueError(f"Invalid checkpoint shard path: {path}")
    else:
        weights = [root / "model.safetensors"]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(f"Missing QASR weights: {path}")
    # Include all saved processor/tokenizer/generation settings; ignore optimizer
    # and Trainer state, which have no role in inference.
    assets = [root / name for name in (
        "config.json", "processor_config.json", "preprocessor_config.json",
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
        "generation_config.json", "model.safetensors.index.json",
    ) if (root / name).is_file()]
    assets += sorted((root / "chat_templates").glob("*.jinja"))
    return sorted(set(assets + weights)), weights


def checkpoint_identity(directory):
    files, _ = checkpoint_files(directory)
    root = Path(directory).resolve()
    return {str(path.relative_to(root)): sha256(path) for path in files}


def weight_memory(directory, precision):
    """Read only safetensors headers, not multi-GB tensor contents."""
    _, files = checkpoint_files(directory)
    elements = 0
    seen = set()
    for path in files:
        with path.open("rb") as f:
            size_bytes = f.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Invalid safetensors header: {path}")
            size = struct.unpack("<Q", size_bytes)[0]
            if size > 100_000_000 or size > path.stat().st_size - 8:
                raise ValueError(f"Invalid safetensors header size: {path}")
            header = json.loads(f.read(size))
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if name in seen:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            seen.add(name)
            elements += math.prod(entry["shape"])
    return {"stored_tensor_elements": elements,
            "estimated_weight_gib_per_worker": elements * (2 if precision == "bf16" else 4) / 2**30}


def check_memory(directory, precision, memory_gib, workers_per_gpu):
    profile = weight_memory(directory, precision)
    budget = .90 * memory_gib / workers_per_gpu
    # A necessary headroom check, not a measured peak or a promise that it fits.
    required = profile["estimated_weight_gib_per_worker"] + 2.0
    if required > budget:
        raise ValueError(
            f"QASR replicas exceed the memory budget: weights ~{profile['estimated_weight_gib_per_worker']:.2f} GiB "
            f"+ 2 GiB minimum activation/cache headroom per worker, but {workers_per_gpu} workers/GPU "
            f"allow {budget:.2f} GiB each. Lower WORKERS_PER_GPU (start with 4); run smoke before the full suite."
        )
    return {**profile, "workers_per_gpu": workers_per_gpu, "budget_gib_per_worker": budget,
            "minimum_headroom_gib": 2.0, "peak_memory_measured": False}


class QASRRecognizer(Recognizer):
    def __init__(self, checkpoint, device, precision, memory_fraction=1.0):
        import torch
        from qasr import QASRFeatureExtractor, QASRForConditionalGeneration, QASRProcessor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        self.torch, self.device, self.precision = torch, device, precision
        if device.startswith("cuda"):
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            check_memory(checkpoint, precision, torch.cuda.get_device_properties(0).total_memory / 2**30,
                         max(1, round(.90 / memory_fraction)))
        self.processor = QASRProcessor.from_pretrained(checkpoint, local_files_only=True)
        if not isinstance(self.processor.feature_extractor, QASRFeatureExtractor):
            raise ValueError("QASR benchmark requires its saved native QASRFeatureExtractor")
        if self.processor.feature_extractor.sampling_rate != 16000:
            raise ValueError("Prepared benchmark audio is 16 kHz; QASR frontend must match")
        self.processor.tokenizer.padding_side = "left"
        # Load directly in inference precision on CPU, then transfer once.
        self.model, loading = QASRForConditionalGeneration.from_pretrained(
            checkpoint, local_files_only=True, dtype=torch.bfloat16 if precision == "bf16" else torch.float32,
            attn_implementation="sdpa", output_loading_info=True,
        )
        bad = {key: loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
               if loading.get(key)}
        if bad:
            raise ValueError(f"QASR checkpoint did not load exactly: {bad}")
        for field in ("projector_pool_stride",):
            if getattr(self.processor, field) != getattr(self.model.config, field):
                raise ValueError(f"QASR processor/model {field} mismatch")
        self.model.eval().to(device)
        # Transformers 5.19 fills unset supplied generation fields from the
        # model's saved config. Reset that fallback as well as the per-call config.
        from transformers import GenerationConfig
        self.model.generation_config = GenerationConfig()
        self.eos_id = self.processor.tokenizer.eos_token_id
        if self.eos_id is None:
            raise ValueError("QASR tokenizer has no EOS token")

    def features(self, rows):
        import numpy as np

        waves = []
        for row in rows:
            if sha256(row["audio"]) != row["audio_sha256"]:
                raise ValueError(f"Prepared audio changed: {row['id']}")
            waves.append(np.load(row["audio"], allow_pickle=False))
        return self.features_from_audio(waves)

    def features_from_audio(self, waves):
        # No transcript or language hints enter this shared frontend pass.
        batch = self.processor.apply_transcription_request(
            audio=waves, language=None, sampling_rate=16000, return_tensors="pt",
            padding=True, truncation=False,
        )
        return {key: batch[key] for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")}

    def infer(self, rows, features, mode, max_new_tokens):
        torch = self.torch
        batch = {k: v.to(device=self.device, dtype=self.model.dtype if v.is_floating_point() else v.dtype)
                 for k, v in features.items()}
        if mode == "arabic":
            ids = self.processor.tokenizer.encode("language Arabic<asr_text>", add_special_tokens=False)
            ids = torch.tensor([ids], device=self.device).expand(len(rows), -1)
            batch["input_ids"] = torch.cat([batch["input_ids"], ids], 1)
            batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(ids)], 1)
        prompt_length = batch["input_ids"].shape[1]
        if prompt_length + max_new_tokens > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Audio prompt plus generation budget exceeds context; no truncation applied")
        # Fresh settings prevent checkpoint-specific penalties/forced tokens from
        # silently changing the greedy policy used by the GQASR benchmark.
        from transformers import GenerationConfig
        generation = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
            eos_token_id=self.eos_id, pad_token_id=self.eos_id, repetition_penalty=1.0,
        )
        with torch.inference_mode():
            output = self.model.generate(**batch, generation_config=generation)
        predictions = []
        for row, tokens in zip(rows, output[:, prompt_length:].tolist(), strict=True):
            ended = self.eos_id in tokens
            if ended:
                tokens = tokens[:tokens.index(self.eos_id) + 1]
            raw = self.processor.tokenizer.decode(tokens, skip_special_tokens=False)
            header = re.match(r"language\s+([^<]+)<asr_text>", raw)
            # Same output parsing as GQASRProcessor.decode; retain unexpected
            # special-token hallucinations rather than cleaning away errors.
            text = raw.split("<asr_text>", 1)[-1].split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()
            predictions.append({**row, "pred_text": text, "raw_output": raw, "mode": mode,
                                "forced_language": "Arabic" if mode == "arabic" else None,
                                "predicted_language": header.group(1).strip() if header else None,
                                "generated_tokens": len(tokens), "eos_seen": ended,
                                "generation_limit_hit": not ended and len(tokens) >= max_new_tokens})
        return predictions
