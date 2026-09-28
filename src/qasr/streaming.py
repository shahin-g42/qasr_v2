from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import torch

from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor

PCM_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class StreamingConfig:
    """Audio and inference settings shared by the WebSocket sessions."""

    sample_rate: int = 16_000
    partial_interval_seconds: float = 0.75
    min_audio_seconds: float = 0.5
    window_seconds: float = 30.0
    max_new_tokens: int = 128

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.partial_interval_seconds <= 0:
            raise ValueError("partial_interval_seconds must be positive")
        if self.min_audio_seconds <= 0:
            raise ValueError("min_audio_seconds must be positive")
        if self.window_seconds < self.min_audio_seconds:
            raise ValueError("window_seconds must be at least min_audio_seconds")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

    @property
    def partial_interval_samples(self) -> int:
        return max(1, round(self.partial_interval_seconds * self.sample_rate))

    @property
    def min_audio_samples(self) -> int:
        return max(1, round(self.min_audio_seconds * self.sample_rate))

    @property
    def window_samples(self) -> int:
        return max(1, round(self.window_seconds * self.sample_rate))


class PCM16Buffer:
    """A bounded buffer for mono, little-endian signed 16-bit PCM."""

    def __init__(self, max_samples: int) -> None:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = int(max_samples)
        self._pcm = bytearray()
        self.total_samples = 0

    def append(self, chunk: bytes) -> None:
        if len(chunk) % PCM_SAMPLE_WIDTH_BYTES:
            raise ValueError("PCM16 chunks must contain a whole number of samples")
        if not chunk:
            return
        self._pcm.extend(chunk)
        added_samples = len(chunk) // PCM_SAMPLE_WIDTH_BYTES
        self.total_samples += added_samples
        max_bytes = self.max_samples * PCM_SAMPLE_WIDTH_BYTES
        if len(self._pcm) > max_bytes:
            del self._pcm[: len(self._pcm) - max_bytes]

    @property
    def buffered_samples(self) -> int:
        return len(self._pcm) // PCM_SAMPLE_WIDTH_BYTES

    @property
    def is_windowed(self) -> bool:
        return self.total_samples > self.buffered_samples

    def waveform(self) -> np.ndarray:
        if not self._pcm:
            return np.empty(0, dtype=np.float32)
        pcm = np.frombuffer(bytes(self._pcm), dtype="<i2")
        return np.ascontiguousarray(pcm.astype(np.float32) / 32768.0)


def merge_windowed_transcript(previous: str, current: str) -> str:
    """Merge two overlapping rolling-window hypotheses without repeating text.

    The newest hypothesis wins inside the overlap, which lets partial decoding
    correct recent words while retaining text that has fallen out of the audio
    window. Space-delimited languages are merged by words; unsegmented text is
    merged by characters.
    """
    previous = previous.strip()
    current = current.strip()
    if not previous:
        return current
    if not current:
        return previous

    uses_words = any(character.isspace() for character in previous + current)
    old_parts = previous.split() if uses_words else list(previous)
    new_parts = current.split() if uses_words else list(current)
    old_normalized = [_normalize_part(part) for part in old_parts]
    new_normalized = [_normalize_part(part) for part in new_parts]

    # The rolling window removes audio from the beginning, so its new text
    # should start somewhere inside the tail of the previous hypothesis.
    minimum_overlap = 2 if uses_words else 4
    best: tuple[int, int] | None = None
    for old_start in range(len(old_parts)):
        maximum = min(len(old_parts) - old_start, len(new_parts))
        overlap = 0
        while (
            overlap < maximum
            and old_normalized[old_start + overlap] == new_normalized[overlap]
        ):
            overlap += 1
        if overlap >= minimum_overlap and (best is None or overlap > best[1]):
            best = (old_start, overlap)

    if best is None:
        # A very short or heavily revised hypothesis is safer to replace than
        # to concatenate, which would visibly duplicate the entire window.
        return current

    old_start, _ = best
    merged = old_parts[:old_start] + new_parts
    return " ".join(merged) if uses_words else "".join(merged)


def _normalize_part(value: str) -> str:
    normalized = "".join(
        character for character in value.casefold() if character.isalnum()
    )
    return normalized or value.casefold()


class QASRTranscriber:
    """Thin, thread-safe inference wrapper around a loaded QASR checkpoint."""

    def __init__(self, processor: QASRProcessor, model: Any) -> None:
        self.processor = processor
        self.model = model
        self.sample_rate = int(processor.feature_extractor.sampling_rate)
        self._lock = Lock()

    @property
    def lock(self) -> Lock:
        """Lock serializing inference on the shared model.

        Exposed so callers that run the same model through another decoding
        path (e.g. EAGLE speculative decoding) can serialize with
        :meth:`transcribe` instead of racing it.
        """
        return self._lock

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        dtype: str = "auto",
        device: str = "auto",
    ) -> QASRTranscriber:
        processor = QASRProcessor.from_pretrained(model_path)
        torch_dtype = _resolve_dtype(dtype, device)
        model_kwargs: dict[str, Any] = {
            "dtype": torch_dtype,
            "attn_implementation": "sdpa",
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        model = QASRForConditionalGeneration.from_pretrained(
            model_path,
            **model_kwargs,
        ).eval()
        if device != "auto":
            model.to(device)
        return cls(processor=processor, model=model)

    def transcribe(
        self,
        waveform: np.ndarray,
        *,
        language: str | None,
        max_new_tokens: int,
    ) -> str:
        if waveform.ndim != 1:
            raise ValueError("Streaming QASR expects one-dimensional mono audio")
        if waveform.size == 0:
            return ""

        # Transformers generation and the shared model cache are not safe to
        # mutate concurrently. Keeping this lock here also protects callers
        # that use the transcriber outside the bundled ASGI server.
        with self._lock, torch.inference_mode():
            inputs = self.processor.apply_transcription_request(
                audio=np.ascontiguousarray(waveform, dtype=np.float32),
                language=language,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            ).to(self.model.device, self.model.dtype)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
            return self.processor.decode(
                generated_ids[0],
                return_format="transcription_only",
            ).strip()


def _resolve_dtype(dtype: str, device: str) -> torch.dtype:
    choices = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype != "auto":
        try:
            return choices[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype!r}") from exc
    if device == "cpu" or not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


__all__ = [
    "PCM16Buffer",
    "QASRTranscriber",
    "StreamingConfig",
    "merge_windowed_transcript",
]
