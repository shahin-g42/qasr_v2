from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


class AudioLoadingError(RuntimeError):
    """Raised when an audio sample cannot be decoded safely."""


class AudioDurationError(ValueError):
    """Raised when decoded audio falls outside the configured duration range."""


def load_mono_audio(path: str | Path, target_sampling_rate: int) -> np.ndarray:
    """Decode audio as float32 mono and resample it with a polyphase filter."""
    audio_path = Path(path)
    try:
        waveform, source_sampling_rate = sf.read(
            audio_path,
            dtype="float32",
            always_2d=True,
        )
    except Exception as exc:
        raise AudioLoadingError(f"Could not decode {audio_path}: {exc}") from exc

    if source_sampling_rate <= 0:
        raise AudioLoadingError(f"Invalid sampling rate {source_sampling_rate} in {audio_path}")
    if waveform.shape[0] == 0:
        raise AudioLoadingError(f"Audio file is empty: {audio_path}")

    waveform = waveform.mean(axis=1, dtype=np.float32)
    if source_sampling_rate != target_sampling_rate:
        divisor = math.gcd(source_sampling_rate, target_sampling_rate)
        waveform = resample_poly(
            waveform,
            up=target_sampling_rate // divisor,
            down=source_sampling_rate // divisor,
        ).astype(np.float32, copy=False)

    waveform = np.ascontiguousarray(waveform, dtype=np.float32)
    if not np.isfinite(waveform).all():
        raise AudioLoadingError(f"Audio contains NaN or infinite values: {audio_path}")
    return waveform


def validate_audio_duration(
    waveform: np.ndarray,
    *,
    path: str | Path,
    sampling_rate: int,
    min_audio_seconds: float,
    max_audio_seconds: float,
) -> None:
    """Ensure decoded audio, rather than manifest metadata, is in range."""
    actual = waveform.shape[0] / sampling_rate
    min_samples = int(round(min_audio_seconds * sampling_rate))
    if waveform.shape[0] < min_samples:
        raise AudioDurationError(
            f"{path} is {actual:.3f}s after decoding, shorter than the configured "
            f"{min_audio_seconds:.3f}s minimum"
        )

    max_samples = int(round(max_audio_seconds * sampling_rate))
    if waveform.shape[0] > max_samples:
        raise AudioDurationError(
            f"{path} is {actual:.3f}s after decoding, longer than the configured "
            f"{max_audio_seconds:.3f}s limit"
        )

__all__ = [
    "AudioDurationError",
    "AudioLoadingError",
    "load_mono_audio",
    "validate_audio_duration",
]
