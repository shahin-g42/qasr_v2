"""Audio and spectrogram augmentation for QASR training.

Provides four augmentation types that can be composed:

1. **SpecAugment** — Time and frequency masking on extracted features.
2. **SpeedPerturbation** — Resample waveforms at slightly different speeds.
3. **NoiseInjection** — Mix background noise into waveforms.
4. **CodecAugmentation** — Simulate telephone/codec compression artifacts.

Waveform-level augmentations (speed, noise, codec) are applied *before* feature
extraction so the Cohere encoder sees naturally distorted spectrograms.
SpecAugment is applied *after* feature extraction on the 128-bin features.

Usage in training config (YAML)::

    augmentation:
      spec_augment:
        enabled: true
        num_time_masks: 2
        max_time_mask_ratio: 0.05
        num_freq_masks: 2
        max_freq_mask_ratio: 0.15
        p: 0.5
      speed_perturb:
        enabled: true
        rates: [0.9, 1.0, 1.1]
        p: 0.5
      noise_injection:
        enabled: true
        noise_dir: /path/to/musan/noise
        snr_range: [5.0, 20.0]
        p: 0.3
      codec_augment:
        enabled: true
        p: 0.2
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("qasr")


# ---------------------------------------------------------------------------
# SpecAugment (feature-level)
# ---------------------------------------------------------------------------


class SpecAugment:
    """Apply time and frequency masking to log-filterbank features.

    Operates on tensors of shape ``(batch, time, freq)`` with a corresponding
    ``attention_mask`` of shape ``(batch, time)`` so that padding frames are
    never masked and remain zero.

    Each sample gets a random intensity: the number of masks and their widths
    are sampled uniformly per sample, creating variable-strength augmentation.
    """

    def __init__(
        self,
        num_time_masks: int = 2,
        max_time_mask_ratio: float = 0.05,
        num_freq_masks: int = 2,
        max_freq_mask_ratio: float = 0.15,
        p: float = 0.5,
    ) -> None:
        if not 0.0 < max_time_mask_ratio < 1.0:
            raise ValueError("max_time_mask_ratio must be in (0, 1)")
        if not 0.0 < max_freq_mask_ratio < 1.0:
            raise ValueError("max_freq_mask_ratio must be in (0, 1)")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.num_time_masks = num_time_masks
        self.max_time_mask_ratio = max_time_mask_ratio
        self.num_freq_masks = num_freq_masks
        self.max_freq_mask_ratio = max_freq_mask_ratio
        self.p = p

    @torch.no_grad()
    def __call__(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpecAugment with per-sample random intensity.

        Args:
            features: ``(batch, time, freq)`` feature tensor.
            attention_mask: ``(batch, time)`` boolean/float mask where 1 = valid.

        Returns:
            Augmented features with the same shape.
        """
        if random.random() > self.p:
            return features

        features = features.clone()
        batch_size, _max_time, num_freq = features.shape

        for b in range(batch_size):
            valid_length = int(attention_mask[b].sum().item())
            if valid_length <= 1:
                continue

            # Per-sample random intensity: how many masks to apply (1 to max)
            n_time_masks = random.randint(1, self.num_time_masks)
            n_freq_masks = random.randint(1, self.num_freq_masks)

            # Per-sample intensity factor (0.3 to 1.0) scales mask widths
            intensity = random.uniform(0.3, 1.0)

            # Time masking with random widths per sample
            max_t = max(1, int(valid_length * self.max_time_mask_ratio * intensity))
            for _ in range(n_time_masks):
                t_width = random.randint(0, max_t)
                if t_width == 0:
                    continue
                t_start = random.randint(0, valid_length - t_width)
                features[b, t_start : t_start + t_width, :] = 0.0

            # Frequency masking with random widths per sample
            max_f = max(1, int(num_freq * self.max_freq_mask_ratio * intensity))
            for _ in range(n_freq_masks):
                f_width = random.randint(0, max_f)
                if f_width == 0:
                    continue
                f_start = random.randint(0, num_freq - f_width)
                features[b, :valid_length, f_start : f_start + f_width] = 0.0

        return features


# ---------------------------------------------------------------------------
# Speed Perturbation (waveform-level)
# ---------------------------------------------------------------------------


class SpeedPerturbation:
    """Resample waveforms at slightly different speeds.

    Speed changes alter both the duration and pitch of the audio. For ASR
    training this is acceptable and acts as a strong regularizer. The pitch
    shift is small enough (±10%) that it does not significantly affect
    intelligibility.

    Each sample gets a random speed factor sampled from a continuous range
    around 1.0, providing variable-strength perturbation per sample.

    .. note::
        Speed perturbation changes the waveform length, which cascades into
        different feature frame counts and audio token counts. The collator
        must re-validate the audio contract after applying this augmentation.
    """

    def __init__(
        self,
        rates: list[float] | None = None,
        rate_range: tuple[float, float] | None = None,
        p: float = 0.5,
    ) -> None:
        # If rate_range is provided, use continuous sampling; otherwise use discrete rates
        self.rate_range = rate_range  # e.g., (0.85, 1.15) for continuous
        self.rates = rates or [0.9, 1.0, 1.1]
        self.p = p
        if rate_range is None and 1.0 not in self.rates:
            LOGGER.warning("SpeedPerturbation rates do not include 1.0 (no-op).")

    def _sample_rate(self) -> float:
        """Sample a random speed factor for this sample."""
        if self.rate_range is not None:
            # Continuous uniform sampling within range
            return random.uniform(self.rate_range[0], self.rate_range[1])
        else:
            # Discrete sampling from provided rates
            return random.choice(self.rates)

    def __call__(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Apply speed perturbation to a mono waveform.

        Args:
            waveform: 1-D float32 numpy array.
            sampling_rate: Sample rate of the waveform.

        Returns:
            Resampled waveform (length may differ).
        """
        if random.random() > self.p:
            return waveform

        rate = self._sample_rate()
        if abs(rate - 1.0) < 1e-6:
            return waveform

        # Use linear interpolation for resampling (fast, good enough for ±15%)
        original_length = len(waveform)
        new_length = round(original_length / rate)
        if new_length <= 0:
            return waveform

        # torch-based resampling for quality
        waveform_t = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)
        resampled = torch.nn.functional.interpolate(
            waveform_t,
            size=new_length,
            mode="linear",
            align_corners=False,
        )
        return resampled.squeeze().numpy()


# ---------------------------------------------------------------------------
# Noise Injection (waveform-level)
# ---------------------------------------------------------------------------


class NoiseInjection:
    """Mix background noise into waveforms at a random SNR.

    Noise clips are loaded from a directory of WAV files (e.g., MUSAN noise).
    If no noise directory is provided, synthetic noise (Gaussian) is used
    as a fallback.

    Each sample gets a random SNR and random noise type (if multiple noise
    files are available), providing variable-strength noise injection.
    """

    def __init__(
        self,
        noise_dir: str | Path | None = None,
        snr_range: tuple[float, float] = (5.0, 20.0),
        noise_types: list[str] | None = None,
        p: float = 0.3,
        sampling_rate: int = 16_000,
    ) -> None:
        self.noise_dir = Path(noise_dir) if noise_dir else None
        self.snr_range = snr_range
        self.noise_types = noise_types or ["gaussian"]  # fallback
        self.p = p
        self.sampling_rate = sampling_rate
        self._noise_files: dict[str, list[Path]] = {}  # type -> files

        if self.noise_dir and self.noise_dir.is_dir():
            # Scan for noise files organized by type (subdirectories) or flat
            for subdir in self.noise_dir.iterdir():
                if subdir.is_dir():
                    files = sorted(
                        p for p in subdir.rglob("*")
                        if p.suffix.lower() in (".wav", ".flac", ".mp3")
                    )
                    if files:
                        self._noise_files[subdir.name] = files
            # If no subdirectories, treat all files as one type
            if not self._noise_files:
                files = sorted(
                    p for p in self.noise_dir.rglob("*")
                    if p.suffix.lower() in (".wav", ".flac", ".mp3")
                )
                if files:
                    self._noise_files["default"] = files
            if self._noise_files:
                self.noise_types = list(self._noise_files.keys())
                total_files = sum(len(v) for v in self._noise_files.values())
                LOGGER.info(
                    "NoiseInjection loaded %d noise files (%s) from %s",
                    total_files, ", ".join(self.noise_types), self.noise_dir,
                )
            else:
                LOGGER.warning("Noise directory %s contains no audio files.", self.noise_dir)

    def _load_noise(self, length: int, noise_type: str | None = None) -> np.ndarray:
        """Load or generate a noise clip of the given length."""
        # Try to load from noise files
        if self._noise_files:
            # Pick a random noise type if available
            if noise_type and noise_type in self._noise_files:
                candidates = self._noise_files[noise_type]
            else:
                # Pick from all available files
                all_files = []
                for files in self._noise_files.values():
                    all_files.extend(files)
                candidates = all_files

            if candidates:
                noise_path = random.choice(candidates)
                try:
                    import soundfile as sf

                    noise, sr = sf.read(str(noise_path), dtype="float32")
                    if noise.ndim > 1:
                        noise = noise.mean(axis=1)
                    if sr != self.sampling_rate:
                        ratio = self.sampling_rate / sr
                        new_len = int(len(noise) * ratio)
                        noise_t = torch.from_numpy(noise).float().unsqueeze(0).unsqueeze(0)
                        noise = (
                            torch.nn.functional.interpolate(
                                noise_t, size=new_len, mode="linear", align_corners=False
                            )
                            .squeeze()
                            .numpy()
                        )
                    # Tile or truncate to match target length
                    if len(noise) < length:
                        repeats = (length // len(noise)) + 1
                        noise = np.tile(noise, repeats)
                    noise = noise[:length]
                    return noise
                except Exception as exc:
                    LOGGER.debug("Failed to load noise file %s: %s", noise_path, exc)

        # Fallback: Gaussian noise with random amplitude
        amplitude = random.uniform(0.005, 0.02)
        return np.random.randn(length).astype(np.float32) * amplitude

    def __call__(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Mix noise into the waveform at a random SNR.

        Args:
            waveform: 1-D float32 numpy array.
            sampling_rate: Sample rate.

        Returns:
            Noisy waveform with the same length.
        """
        if random.random() > self.p:
            return waveform

        # Per-sample random SNR and noise type
        snr_db = random.uniform(self.snr_range[0], self.snr_range[1])
        noise_type = random.choice(self.noise_types) if self.noise_types else None
        noise = self._load_noise(len(waveform), noise_type)

        # Compute RMS
        signal_power = np.mean(waveform ** 2)
        noise_power = np.mean(noise ** 2)

        if signal_power < 1e-10 or noise_power < 1e-10:
            return waveform

        # Scale noise to target SNR
        snr_linear = 10 ** (snr_db / 20)
        scale = np.sqrt(signal_power / (noise_power * snr_linear ** 2))
        noisy = waveform + scale * noise

        # Normalize to prevent clipping
        peak = np.max(np.abs(noisy))
        if peak > 1.0:
            noisy = noisy / peak * 0.95

        return noisy.astype(np.float32)


# ---------------------------------------------------------------------------
# Codec Augmentation (waveform-level)
# ---------------------------------------------------------------------------


class CodecAugmentation:
    """Simulate telephone / low-bitrate codec compression artifacts.

    Uses a simple approach: bandpass filter (simulating telephone bandwidth)
    + quantization noise (simulating low-bitrate encoding). This avoids
    requiring external codec libraries while still providing meaningful
    degradation for robustness training.

    Each sample gets random codec parameters: bandpass range, quantization
    bits, and blend ratio, creating variable-strength degradation.
    """

    def __init__(
        self,
        p: float = 0.2,
        sampling_rate: int = 16_000,
    ) -> None:
        self.p = p
        self.sampling_rate = sampling_rate

    def __call__(self, waveform: np.ndarray, sampling_rate: int) -> np.ndarray:
        """Apply codec-like degradation with per-sample random intensity.

        Args:
            waveform: 1-D float32 numpy array.
            sampling_rate: Sample rate.

        Returns:
            Degraded waveform with the same length.
        """
        if random.random() > self.p:
            return waveform

        waveform_t = torch.from_numpy(waveform).float()

        # Per-sample random codec parameters
        n = len(waveform_t)
        spectrum = torch.fft.rfft(waveform_t)
        freqs = torch.fft.rfftfreq(n, d=1.0 / sampling_rate)

        # 1. Random bandpass filter (narrower = more degraded)
        #    Wide: 100-5000 Hz (slight degradation)
        #    Narrow: 300-3000 Hz (telephone quality)
        low_cut = random.uniform(100.0, 500.0)
        high_cut = random.uniform(2500.0, 5000.0)
        mask = (freqs >= low_cut) & (freqs <= high_cut)
        spectrum = spectrum * mask.float()
        filtered = torch.fft.irfft(spectrum, n=n)

        # 2. Random quantization noise (fewer bits = more degraded)
        #    High quality: 14-16 bits
        #    Low quality: 6-10 bits
        bits = random.randint(6, 16)
        levels = 2 ** bits
        quantized = torch.round(filtered * (levels / 2)) / (levels / 2)

        # 3. Random blend ratio (how much of the degraded signal to mix)
        #    Light: 0.2-0.4 (mostly original)
        #    Heavy: 0.6-0.9 (mostly degraded)
        alpha = random.uniform(0.2, 0.9)
        result = (1 - alpha) * waveform_t + alpha * quantized

        return result.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for all augmentations."""

    spec_augment: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "num_time_masks": 2,
        "max_time_mask_ratio": 0.05,
        "num_freq_masks": 2,
        "max_freq_mask_ratio": 0.15,
        "p": 0.5,
    })
    speed_perturb: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "rates": [0.9, 1.0, 1.1],
        "p": 0.5,
    })
    noise_injection: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "noise_dir": None,
        "snr_range": [5.0, 20.0],
        "p": 0.3,
    })
    codec_augment: dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "p": 0.2,
    })


class AudioAugmenter:
    """Compose waveform-level augmentations.

    SpecAugment is handled separately in the collator because it operates
    on extracted features, not waveforms.
    """

    def __init__(self, config: AugmentationConfig, sampling_rate: int = 16_000) -> None:
        self.config = config
        self.sampling_rate = sampling_rate

        # Waveform-level augmentations (applied in order)
        self.speed_perturb: SpeedPerturbation | None = None
        if config.speed_perturb.get("enabled", False):
            rate_range_cfg = config.speed_perturb.get("rate_range")
            rate_range = tuple(rate_range_cfg) if rate_range_cfg else None
            self.speed_perturb = SpeedPerturbation(
                rates=config.speed_perturb.get("rates", [0.9, 1.0, 1.1]),
                rate_range=rate_range,
                p=config.speed_perturb.get("p", 0.5),
            )
            mode = f"continuous={rate_range}" if rate_range else f"discrete={config.speed_perturb.get('rates')}"
            LOGGER.info("SpeedPerturbation enabled: %s, p=%.2f", mode, config.speed_perturb.get("p", 0.5))

        self.noise_injection: NoiseInjection | None = None
        if config.noise_injection.get("enabled", False):
            self.noise_injection = NoiseInjection(
                noise_dir=config.noise_injection.get("noise_dir"),
                snr_range=tuple(config.noise_injection.get("snr_range", [5.0, 20.0])),
                p=config.noise_injection.get("p", 0.3),
                sampling_rate=sampling_rate,
            )
            LOGGER.info("NoiseInjection enabled: snr=%s, p=%.2f",
                        config.noise_injection.get("snr_range"), config.noise_injection.get("p", 0.3))

        self.codec_augment: CodecAugmentation | None = None
        if config.codec_augment.get("enabled", False):
            self.codec_augment = CodecAugmentation(
                p=config.codec_augment.get("p", 0.2),
                sampling_rate=sampling_rate,
            )
            LOGGER.info("CodecAugmentation enabled: p=%.2f", config.codec_augment.get("p", 0.2))

        # SpecAugment (feature-level, applied after feature extraction)
        self.spec_augment: SpecAugment | None = None
        if config.spec_augment.get("enabled", False):
            self.spec_augment = SpecAugment(
                num_time_masks=config.spec_augment.get("num_time_masks", 2),
                max_time_mask_ratio=config.spec_augment.get("max_time_mask_ratio", 0.05),
                num_freq_masks=config.spec_augment.get("num_freq_masks", 2),
                max_freq_mask_ratio=config.spec_augment.get("max_freq_mask_ratio", 0.15),
                p=config.spec_augment.get("p", 0.5),
            )
            LOGGER.info("SpecAugment enabled: time_masks=%d, freq_masks=%d, p=%.2f",
                        config.spec_augment.get("num_time_masks", 2),
                        config.spec_augment.get("num_freq_masks", 2),
                        config.spec_augment.get("p", 0.5))

    @property
    def has_waveform_augmentation(self) -> bool:
        return any([self.speed_perturb, self.noise_injection, self.codec_augment])

    @property
    def has_spec_augmentation(self) -> bool:
        return self.spec_augment is not None

    def augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Apply all waveform-level augmentations in sequence.

        Order: speed → noise → codec. Speed is applied first because it
        changes the waveform length; noise and codec preserve length.
        """
        if not self.has_waveform_augmentation:
            return waveform

        if self.speed_perturb is not None:
            waveform = self.speed_perturb(waveform, self.sampling_rate)

        if self.noise_injection is not None:
            waveform = self.noise_injection(waveform, self.sampling_rate)

        if self.codec_augment is not None:
            waveform = self.codec_augment(waveform, self.sampling_rate)

        return waveform

    def augment_features(
        self,
        features: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SpecAugment to extracted features."""
        if self.spec_augment is None:
            return features
        return self.spec_augment(features, attention_mask)


def build_augmenter(
    augmentation: dict[str, Any] | None,
    sampling_rate: int,
) -> AudioAugmenter | None:
    """Build an :class:`AudioAugmenter` from a training-config ``augmentation`` dict.

    Each sub-dict (``spec_augment``, ``speed_perturb``, ``noise_injection``,
    ``codec_augment``) is merged over the :class:`AugmentationConfig` defaults,
    so YAMLs only need to spell out the keys they change. Returns ``None``
    when no augmentation section is configured.
    """
    if not augmentation:
        return None
    aug_config = AugmentationConfig()
    for aug_name, aug_params in augmentation.items():
        if hasattr(aug_config, aug_name) and isinstance(aug_params, dict):
            merged = getattr(aug_config, aug_name).copy()
            merged.update(aug_params)
            setattr(aug_config, aug_name, merged)
    return AudioAugmenter(aug_config, sampling_rate=sampling_rate)


__all__ = [
    "AudioAugmenter",
    "AugmentationConfig",
    "CodecAugmentation",
    "NoiseInjection",
    "SpecAugment",
    "SpeedPerturbation",
    "build_augmenter",
]
