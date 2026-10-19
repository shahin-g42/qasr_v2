"""Native public-model inference on the exact prepared benchmark waveforms.

Only audio and an Arabic language hint reach a processor/model. Qwen output parsing
preserves repetitions; its convenience transcription decoder also edits repetitions.
Cohere uses its own chunk mapping. Whisper uses documented overlapping 30s chunks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import sha256
from .worker import Recognizer

BACKENDS = ("hf-qwen", "cohere", "whisper")
MODEL_TYPES = {"hf-qwen": "qwen3_asr", "cohere": "cohere_asr", "whisper": "whisper"}


def checkpoint_identity(directory, backend):
    root = Path(directory)
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != MODEL_TYPES[backend]:
        raise ValueError(f"Expected {MODEL_TYPES[backend]} checkpoint: {root}")
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and ".cache" not in p.relative_to(root).parts
    )
    weights = [p for p in files if p.suffix == ".safetensors"]
    if not weights or not (root / "preprocessor_config.json").is_file():
        raise ValueError("Public checkpoint requires safetensors weights and its saved frontend")
    if not any(p.name in {"tokenizer.json", "tokenizer.model", "vocab.json"} for p in files):
        raise ValueError("Public checkpoint tokenizer is missing")
    index = root / "model.safetensors.index.json"
    if index.exists():
        for name in set(json.loads(index.read_text())["weight_map"].values()):
            if Path(name).name != name or not (root / name).is_file():
                raise ValueError(f"Missing or unsafe checkpoint shard: {name}")
    return {p.relative_to(root).as_posix(): sha256(p) for p in files}


def generation_stats(tokens, eos_ids, budget):
    eos_ids = set(eos_ids if isinstance(eos_ids, (list, tuple, set)) else [eos_ids]) - {None}
    end = next((i + 1 for i, token in enumerate(tokens) if token in eos_ids), None)
    tokens = tokens[:end] if end is not None else tokens
    return {
        "generated_tokens": len(tokens),
        "eos_seen": end is not None,
        "generation_limit_hit": end is None and len(tokens) >= budget,
        "generation_limit_observable": True,
    }


def reassemble_chunks(texts, statistics, mapping, sample_count):
    if mapping is None:
        mapping = [(i, None) for i in range(len(texts))]
    if len(mapping) != len(texts) or len(texts) != len(statistics):
        raise ValueError("Cohere chunk outputs do not match the processor mapping")
    groups = [[] for _ in range(sample_count)]
    seen = set()
    for text, stats, (sample, chunk) in zip(texts, statistics, mapping, strict=True):
        if not isinstance(sample, int) or not 0 <= sample < sample_count or (sample, chunk) in seen:
            raise ValueError("Invalid or duplicate Cohere chunk identity")
        if chunk is not None and (not isinstance(chunk, int) or chunk < 0):
            raise ValueError("Invalid Cohere chunk ordinal")
        seen.add((sample, chunk))
        groups[sample].append((chunk, text, stats))
    result = []
    for group in groups:
        if not group or (len(group) > 1 and any(c is None for c, _, _ in group)):
            raise ValueError("Missing or inconsistent Cohere sample chunks")
        if group[0][0] is not None and sorted(c for c, _, _ in group) != list(range(len(group))):
            raise ValueError("Noncontiguous Cohere chunk ordinals")
        group.sort(key=lambda item: item[0] if item[0] is not None else 0)
        result.append(
            {
                "pred_text": " ".join(t.strip() for _, t, _ in group if t.strip()),
                "raw_output": [t for _, t, _ in group],
                "audio_chunks": len(group),
                "generated_tokens": sum(s["generated_tokens"] for _, _, s in group),
                "eos_seen": all(s["eos_seen"] for _, _, s in group),
                "generation_limit_hit": any(s["generation_limit_hit"] for _, _, s in group),
                "generation_limit_observable": True,
            }
        )
    return result


class PublicRecognizer(Recognizer):
    backend = None

    def __init__(self, checkpoint, device, precision, memory_fraction=1.0):
        import torch
        import transformers

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        self.torch, self.device = torch, device
        if device.startswith("cuda"):
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        classes = {
            "hf-qwen": "Qwen3ASRForConditionalGeneration",
            "cohere": "CohereAsrForConditionalGeneration",
            "whisper": "WhisperForConditionalGeneration",
        }
        self.processor = transformers.AutoProcessor.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False
        )
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        self.model, loading = getattr(transformers, classes[self.backend]).from_pretrained(
            checkpoint,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        bad = {
            k: loading.get(k)
            for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading.get(k)
        }
        if bad:
            raise ValueError(f"Public checkpoint did not load exactly: {bad}")
        self.model.eval().to(device)
        if self.processor.feature_extractor.sampling_rate != 16000:
            raise ValueError("Benchmark requires a native 16 kHz frontend")
        if self.backend == "hf-qwen":
            self.processor.tokenizer.padding_side = "left"
        if self.backend == "whisper":
            self.pipeline = transformers.pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=device,
                chunk_length_s=30,
                stride_length_s=(5, 5),
            )

    def features(self, rows):
        import numpy as np

        waves = np.empty(len(rows), dtype=object)
        for i, row in enumerate(rows):
            if sha256(row["audio"]) != row["audio_sha256"]:
                raise ValueError(f"Prepared audio changed: {row['id']}")
            wave = np.load(row["audio"], allow_pickle=False)
            if (
                wave.ndim != 1
                or wave.dtype != np.float32
                or not len(wave)
                or not np.isfinite(wave).all()
            ):
                raise ValueError(f"Invalid 16 kHz mono float32 prepared audio: {row['id']}")
            if abs(len(wave) / 16000 - row["actual_duration"]) > 1 / 16000:
                raise ValueError(f"Prepared duration mismatch: {row['id']}")
            waves[i] = wave
        # The common worker can slice this by sample before processor chunk expansion.
        return {"audio": waves}

    def infer(self, rows, features, mode, max_new_tokens):
        if mode != "arabic":
            raise ValueError("Public baseline comparisons require the Arabic language hint")
        waves = list(features["audio"])
        with self.torch.inference_mode():
            if self.backend == "whisper":
                # Pipeline stitching does not expose per-chunk generation token counts.
                if max_new_tokens > 440:
                    raise ValueError(
                        "Whisper token budget must leave room for its decoder prefix (<=440)"
                    )
                outputs = self.pipeline(
                    [{"array": w, "sampling_rate": 16000} for w in waves],
                    batch_size=len(waves),
                    generate_kwargs={
                        "language": "ar",
                        "task": "transcribe",
                        "do_sample": False,
                        "num_beams": 1,
                        "max_new_tokens": max_new_tokens,
                    },
                )
                values = [
                    {
                        "pred_text": r["text"],
                        "raw_output": r["text"],
                        "generated_tokens": None,
                        "generation_limit_hit": None,
                        "eos_seen": None,
                        "generation_limit_observable": False,
                    }
                    for r in outputs
                ]
            elif self.backend == "hf-qwen":
                inputs = self.processor.apply_transcription_request(
                    audio=waves,
                    language="Arabic",
                    prompt="",
                    processor_kwargs={
                        "return_tensors": "pt",
                        "padding": True,
                        "sampling_rate": 16000,
                        "truncation": False,
                    },
                )
                prompt_length = inputs["input_ids"].shape[1]
                limit = self.model.config.text_config.max_position_embeddings
                if prompt_length + max_new_tokens > limit:
                    raise ValueError(
                        "Qwen audio plus generation budget exceeds context; no audio truncated"
                    )
                inputs = inputs.to(self.device, self.model.dtype)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                )
                values = []
                for ids in outputs[:, prompt_length:].tolist():
                    stats = generation_stats(
                        ids, self.model.generation_config.eos_token_id, max_new_tokens
                    )
                    raw = self.processor.tokenizer.decode(
                        ids[: stats["generated_tokens"]], skip_special_tokens=False
                    )
                    text = (
                        raw.split("<asr_text>", 1)[-1]
                        .split("<|im_end|>", 1)[0]
                        .split("<|endoftext|>", 1)[0]
                        .strip()
                    )
                    values.append({"pred_text": text, "raw_output": raw, **stats})
            else:
                inputs = self.processor(
                    audio=waves,
                    sampling_rate=16000,
                    return_tensors="pt",
                    language="ar",
                    punctuation=True,
                )
                mapping = inputs.pop("audio_chunk_index", None)
                prompt_length = inputs["decoder_input_ids"].shape[1]
                inputs = inputs.to(self.device, self.model.dtype)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    repetition_penalty=1.0,
                )
                texts = self.processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                stats = [
                    generation_stats(
                        ids[prompt_length:],
                        self.model.generation_config.eos_token_id,
                        max_new_tokens,
                    )
                    for ids in outputs.tolist()
                ]
                values = reassemble_chunks(texts, stats, mapping, len(rows))
        if len(values) != len(rows) or any(not isinstance(v["pred_text"], str) for v in values):
            raise ValueError("Public model did not return one transcription per clip")
        return [
            {**row, **value, "mode": mode, "forced_language": "Arabic", "predicted_language": None}
            for row, value in zip(rows, values, strict=True)
        ]


class QwenRecognizer(PublicRecognizer):
    backend = "hf-qwen"


class CohereRecognizer(PublicRecognizer):
    backend = "cohere"


class WhisperRecognizer(PublicRecognizer):
    backend = "whisper"


def recognizer_class(backend):
    return {"hf-qwen": QwenRecognizer, "cohere": CohereRecognizer, "whisper": WhisperRecognizer}[
        backend
    ]
