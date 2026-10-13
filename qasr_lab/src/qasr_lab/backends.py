"""Lazy inference adapters. Each CLI invocation loads only its selected backend."""

from __future__ import annotations

import math
import platform
import time

from .checkpoints import checkpoint_identity, use_qasr_source
from .io import environment, file_hash, positive_number

LANGUAGES = {"ar": "Arabic", "en": "English", "zh": "Chinese", "hi": "Hindi", "ml": "Malayalam"}
MODEL_KEYS = {
    "id",
    "backend",
    "checkpoint",
    "revision",
    "device",
    "dtype",
    "max_new_tokens",
    "warmup_samples",
    "language_hints",
    "hash_weights",
    "qasr_source",
    "attn_implementation",
    "max_audio_seconds",
}


def validate_model_config(config):
    unknown = set(config) - MODEL_KEYS
    if unknown:
        raise ValueError(f"Unknown model settings: {sorted(unknown)}")
    for key in ("id", "backend", "checkpoint"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if config["backend"] not in ("qasr", "qasr-lab", "hf-qwen", "qwen-sdk", "cohere"):
        raise ValueError("backend must be qasr, qasr-lab, hf-qwen, qwen-sdk or cohere")
    result = {
        "device": "cpu",
        "dtype": "float32",
        "max_new_tokens": 768,
        "warmup_samples": 2,
        "language_hints": True,
        "hash_weights": True,
        "attn_implementation": "sdpa",
        "max_audio_seconds": 35.0,
        **config,
    }
    if result["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("Unsupported dtype")
    for key, minimum in (("max_new_tokens", 1), ("warmup_samples", 0)):
        if (
            isinstance(result[key], bool)
            or not isinstance(result[key], int)
            or result[key] < minimum
        ):
            raise ValueError(f"Invalid {key}")
    for key in ("language_hints", "hash_weights"):
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be boolean")
    positive_number(result["max_audio_seconds"], "max_audio_seconds")
    if result["backend"] == "cohere" and not result["language_hints"]:
        raise ValueError("Cohere requires an explicit language; use a hinted panel comparison")
    return result


def load_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    if not waveform.size or rate <= 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")
    waveform = waveform.mean(axis=1, dtype=np.float32)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        waveform = resample_poly(waveform, 16000 // divisor, rate // divisor)
    return np.ascontiguousarray(waveform, dtype=np.float32)


class HFBackend:
    def __init__(self, config):
        import torch

        self.config = validate_model_config(config)
        config = self.config
        self.torch = torch
        self.device = torch.device(config["device"])
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("The reproducible runner supports cpu or a single cuda device")
        self.dtype = getattr(torch, config["dtype"])
        self.metadata = {
            **config,
            "checkpoint_identity": checkpoint_identity(
                config["checkpoint"],
                revision=config.get("revision"),
                hash_weights=config["hash_weights"],
            ),
        }
        common = {"revision": config.get("revision"), "trust_remote_code": False}
        load = {**common, "dtype": self.dtype, "attn_implementation": config["attn_implementation"]}
        backend = config["backend"]
        if backend in ("qasr", "qasr-lab"):
            self.metadata["qasr_source"] = use_qasr_source(config.get("qasr_source"))
            from qasr import QASRConfig, QASRForConditionalGeneration, QASRProcessor

            model_config, _ = QASRConfig.get_config_dict(config["checkpoint"], **common)
            if backend == "qasr" and (
                model_config.get("model_type") == "qasr_lab"
                or model_config.get("lab_bridge_dim", 0)
                or model_config.get("lab_mask_pool_padding", False)
                or model_config.get("lab_ctc_vocabulary", [])
            ):
                raise ValueError("This checkpoint has lab extensions; select backend=qasr-lab")

            model_class = QASRForConditionalGeneration
            if backend == "qasr-lab":
                from .modeling import LabQASRForConditionalGeneration

                model_class = LabQASRForConditionalGeneration
            self.processor = QASRProcessor.from_pretrained(config["checkpoint"], **common)
            self.model = (
                model_class.from_pretrained(config["checkpoint"], **load).to(self.device).eval()
            )
        elif backend == "qwen-sdk":
            try:
                from qwen_asr import Qwen3ASRModel
            except ImportError as exc:
                raise ImportError(
                    "Use a separate environment with the official qwen-asr SDK installed"
                ) from exc
            load.pop("trust_remote_code")
            self.model = Qwen3ASRModel.from_pretrained(
                config["checkpoint"],
                device_map=str(self.device),
                max_inference_batch_size=1,
                max_new_tokens=config["max_new_tokens"],
                **load,
            )
            self.processor = None
        else:
            from transformers import (
                AutoProcessor,
                CohereAsrForConditionalGeneration,
                Qwen3ASRForConditionalGeneration,
            )

            self.processor = AutoProcessor.from_pretrained(config["checkpoint"], **common)
            model_class = (
                CohereAsrForConditionalGeneration
                if backend == "cohere"
                else Qwen3ASRForConditionalGeneration
            )
            self.model = (
                model_class.from_pretrained(config["checkpoint"], **load).to(self.device).eval()
            )
        self.metadata["hardware"] = {
            "device": str(self.device),
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "torch_threads": torch.get_num_threads(),
        }
        self.metadata["runtime_packages"] = environment()["packages"]
        self.metadata["decode_protocol"] = {
            "max_new_tokens": config["max_new_tokens"],
            "language_hints": config["language_hints"],
            "dtype": config["dtype"],
            "do_sample": False,
            "num_beams": 1,
            "sample_rate": 16000,
            "batch_size": 1,
            "attn_implementation": config["attn_implementation"],
        }
        self.metadata["timing_protocol"] = (
            {
                "scope": "preprocessing+encoder+decoder; file IO excluded; synchronized",
                "warmup_samples": config["warmup_samples"],
                "clock": "perf_counter",
            }
            if config["warmup_samples"]
            else None
        )

    def sync(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_memory(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        if self.device.type == "cuda":
            return self.torch.cuda.max_memory_allocated(self.device) / 1024**3
        return None

    def transcribe(self, waveform, language):
        torch = self.torch
        backend = self.config["backend"]
        language = language if self.config["language_hints"] else None
        with torch.inference_mode():
            if backend == "qwen-sdk":
                results = self.model.transcribe(
                    audio=(waveform, 16000), language=LANGUAGES.get(language, language)
                )
                return {
                    "hypothesis": results[0].text,
                    "generated_tokens": None,
                    "hit_token_limit": False,
                    "token_limit_observable": False,
                }
            if backend == "cohere":
                inputs = self.processor(
                    audio=waveform, sampling_rate=16000, language=language, return_tensors="pt"
                )
            else:
                inputs = self.processor.apply_transcription_request(
                    audio=waveform,
                    language=language,
                    sampling_rate=16000,
                    return_tensors="pt",
                )
            inputs = inputs.to(self.device, self.dtype)
            if inputs["input_features"].shape[0] != 1:
                raise ValueError(
                    "Audio was split into chunks; use pre-segmented audio for this panel"
                )
            inputs.pop("audio_chunk_index", None)
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
            if backend == "cohere":
                generated = output[0, inputs["decoder_input_ids"].shape[1] :]
                text = self.processor.decode(output, skip_special_tokens=True, language=language)[0]
            else:
                generated = output[0, inputs["input_ids"].shape[1] :]
                text = self.processor.decode(generated, return_format="transcription_only")
            eos = self.model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            truncated = (
                len(generated) >= self.config["max_new_tokens"]
                and int(generated[-1]) not in eos_ids
            )
            return {
                "hypothesis": text.strip(),
                "generated_tokens": len(generated),
                "hit_token_limit": truncated,
                "token_limit_observable": True,
            }


def evaluate_backend(backend, samples):
    warmup = min(backend.config["warmup_samples"], len(samples))
    for row in samples[:warmup]:
        if row.get("audio_sha256") and file_hash(row["audio_filepath"]) != row["audio_sha256"]:
            raise ValueError("Audio checksum differs from the frozen panel")
        audio = load_audio(row["audio_filepath"])
        if len(audio) / 16000 > backend.config["max_audio_seconds"]:
            raise ValueError("Warmup audio exceeds the configured duration limit")
        backend.transcribe(audio, row["language"])
    backend.sync()
    backend.reset_memory()
    for row in samples:
        try:
            if row.get("audio_sha256") and file_hash(row["audio_filepath"]) != row["audio_sha256"]:
                raise ValueError("Audio checksum differs from the frozen panel")
            waveform = load_audio(row["audio_filepath"])
            duration = len(waveform) / 16000
            if not 0.01 <= duration <= backend.config["max_audio_seconds"]:
                raise ValueError(f"Decoded duration outside [0.01, max_audio_seconds]: {duration}")
            if abs(duration - row["duration"]) > max(0.05, 0.01 * duration):
                raise ValueError("Audio duration differs from frozen manifest; rebuild the panel")
            backend.sync()
            started = time.perf_counter()
            prediction = backend.transcribe(waveform, row["language"])
            backend.sync()
            yield {
                "id": row["id"],
                "status": "ok",
                "audio_seconds": duration,
                "latency_seconds": time.perf_counter() - started,
                **prediction,
            }
        except Exception as exc:
            # Failures remain explicit rows. Comparisons reject incomplete reports.
            yield {"id": row["id"], "status": "error", "error": f"{type(exc).__name__}: {exc}"}
