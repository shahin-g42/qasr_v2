from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .audio import load_mono_audio, validate_audio_duration
from .augmentation import AudioAugmenter
from .utils import get_projector_pool_output_lengths, get_subsampling_output_lengths


@dataclass
class QASRDataCollator:
    processor: Any
    language: str
    min_audio_seconds: float
    max_audio_seconds: float
    max_target_length: int
    augmenter: AudioAugmenter | None = None

    def __post_init__(self) -> None:
        self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)

    def _load_waveform(self, feature: dict[str, Any]):
        waveform = feature.get("waveform")
        if waveform is None:
            try:
                waveform = load_mono_audio(feature["audio_path"], self.sampling_rate)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load {feature['audio_path']} "
                    f"(manifest line {feature.get('line_number', '?')}): {exc}"
                ) from exc
        validate_audio_duration(
            waveform,
            path=feature["audio_path"],
            sampling_rate=self.sampling_rate,
            min_audio_seconds=self.min_audio_seconds,
            max_audio_seconds=self.max_audio_seconds,
        )
        return waveform

    def _validate_audio_contract(self, batch: dict[str, torch.Tensor], batch_size: int) -> None:
        input_features = batch["input_features"]
        feature_mask = batch["input_features_mask"]
        if input_features.ndim != 3 or input_features.shape[0] != batch_size:
            raise RuntimeError(
                "The feature extractor split a supervised sample into multiple chunks. "
                "Segment long recordings before training."
            )
        if input_features.shape[2] != 128:
            raise RuntimeError(
                f"Expected time-major 128-bin features, got {tuple(input_features.shape)}"
            )
        expected_tokens = get_subsampling_output_lengths(
            feature_mask.sum(-1),
            subsampling_factor=self.processor.subsampling_factor,
            kernel_size=self.processor.subsampling_conv_kernel_size,
            stride=self.processor.subsampling_conv_stride,
        )
        pool_stride = int(getattr(self.processor, "projector_pool_stride", 1) or 1)
        if pool_stride > 1:
            expected_tokens = get_projector_pool_output_lengths(
                expected_tokens, pool_stride=pool_stride
            )
        audio_tokens = batch["input_ids"].eq(self.processor.audio_token_id).sum(-1)
        if not torch.equal(expected_tokens.to(audio_tokens.device), audio_tokens):
            raise RuntimeError(
                "Audio placeholder counts do not match Cohere encoder output lengths: "
                f"expected {expected_tokens.tolist()}, got {audio_tokens.tolist()}"
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty batch")
        transcript_lengths = [
            len(
                self.processor.tokenizer(
                    feature["text"],
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]
            )
            for feature in features
        ]
        if any(length > self.max_target_length for length in transcript_lengths):
            raise ValueError(
                "Transcript exceeds max_target_length: "
                f"{transcript_lengths} > {self.max_target_length}"
            )
        waveforms = [self._load_waveform(feature) for feature in features]

        # Apply waveform-level augmentations (speed, noise, codec)
        if self.augmenter is not None and self.augmenter.has_waveform_augmentation:
            waveforms = [self.augmenter.augment_waveform(w) for w in waveforms]

        try:
            batch = self.processor.prepare_training_batch(
                audio=waveforms,
                text=[feature["text"] for feature in features],
                language=[feature.get("language") or self.language for feature in features],
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
        except ValueError as exc:
            locations = ", ".join(
                f"{Path(item['audio_path']).name}:{item.get('line_number', '?')}" for item in features
            )
            raise ValueError(f"Could not prepare supervised QASR batch ({locations}): {exc}") from exc
        batch = dict(batch)

        # Apply SpecAugment (feature-level) after feature extraction
        if self.augmenter is not None and self.augmenter.has_spec_augmentation:
            batch["input_features"] = self.augmenter.augment_features(
                batch["input_features"],
                batch["input_features_mask"],
            )

        self._validate_audio_contract(batch, len(features))
        allowed = {"input_ids", "attention_mask", "input_features", "input_features_mask", "labels"}
        return {name: value for name, value in batch.items() if name in allowed}

    def generation_batch(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Cannot collate an empty generation batch")
        waveforms = [self._load_waveform(feature) for feature in features]
        batch = dict(
            self.processor.apply_transcription_request(
                audio=waveforms,
                language=[feature.get("language") or self.language for feature in features],
                sampling_rate=self.sampling_rate,
                padding=True,
                return_tensors="pt",
            )
        )
        self._validate_audio_contract(batch, len(features))
        allowed = {"input_ids", "attention_mask", "input_features", "input_features_mask"}
        return {name: value for name, value in batch.items() if name in allowed}


__all__ = ["QASRDataCollator"]
