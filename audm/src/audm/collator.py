"""QASR augmentations and code-switching, adapted to each AUD audio contract."""

import numpy as np
import torch
from qasr.collator import QASRDataCollator


def spectral_mask_waveform(wave, augmenter):
    """Raw-audio counterpart of SpecAugment: mask STFT bins, preserve phase, reconstruct.

    This is waveform spectral masking, not Moonshine's native training recipe.
    It is used only in training; no mel frontend is inserted into Moonshine.
    """
    tensor = torch.as_tensor(wave, dtype=torch.float32)
    window = torch.hann_window(400)
    spectrum = torch.stft(
        tensor, n_fft=400, hop_length=160, window=window, pad_mode="constant", return_complex=True
    )
    mask = augmenter.augment_features(
        torch.ones(1, spectrum.shape[1], spectrum.shape[0]),
        torch.ones(1, spectrum.shape[1], dtype=torch.bool),
    )[0].T
    result = torch.istft(
        spectrum * mask, n_fft=400, hop_length=160, window=window, length=len(wave)
    )
    return result.numpy().astype(np.float32, copy=False)


class AUDDataCollator(QASRDataCollator):
    def _validate_audio_contract(self, batch, batch_size):
        mask = batch["input_features_mask"]
        if mask.shape[0] != batch_size:
            raise RuntimeError("Audio was split into extra supervised examples")
        expected = self.processor.audio_token_lengths(mask)
        actual = batch["input_ids"].eq(self.processor.audio_token_id).sum(-1)
        if not torch.equal(actual, expected.to(actual.device)):
            raise RuntimeError(
                f"Audio token lengths differ: {actual.tolist()} vs {expected.tolist()}"
            )
        if batch["input_ids"].shape[1] > self.processor.config.text_config.max_position_embeddings:
            raise ValueError("Audio and text exceed decoder context; segment the recording")

    def __call__(self, features):
        if not features:
            raise ValueError("Empty training batch")
        audits = []
        if self.code_switcher is not None:
            mixed = [self.code_switcher.augment(f, self.code_switch_probability) for f in features]
            features, audits = [m[0] for m in mixed], [m[1] for m in mixed]
        if self.audit_callback is not None:
            self.audit_callback(features, audits)
        if getattr(self, "non_speech_pool", None) is not None:
            features = self.non_speech_pool.replace(features)
        # Keep the stage-02/index contract: this limit applies to transcript
        # tokens. Language markers and EOS are additional supervised tokens.
        if any(
            len(self.processor.tokenizer.encode(f["text"], add_special_tokens=False))
            > self.max_target_length
            for f in features
        ):
            raise ValueError("Transcript exceeds max_target_length")
        waves = [self._load_waveform(f) for f in features]
        augmenter = self.augmenter
        if augmenter is not None and augmenter.has_waveform_augmentation:
            augmented = []
            for wave, feature in zip(waves, features, strict=True):
                kind = (
                    feature.get("audio_type") if self.processor.config.audio_type_version else None
                )
                supports_types = getattr(augmenter, "supports_audio_types", False)
                if kind in ("music", "noise", "silence") and not supports_types:
                    augmented.append(wave)
                    continue
                augmented.append(
                    augmenter.augment_waveform(
                        wave,
                        min_audio_seconds=self.min_audio_seconds,
                        max_audio_seconds=self.max_audio_seconds,
                        **({"audio_type": kind} if supports_types else {}),
                    )
                )
            waves = augmented
        variant = self.processor.config.variant
        raw_audio = variant == "audflash" and self.processor.config.flash_encoder == "moonshine"
        if raw_audio and augmenter is not None and augmenter.has_spec_augmentation:
            waves = [
                w
                if self.processor.config.audio_type_version
                and f.get("audio_type") in ("music", "noise", "silence")
                and not getattr(augmenter, "supports_audio_types", False)
                else spectral_mask_waveform(w, augmenter)
                for w, f in zip(waves, features, strict=True)
            ]
        batch = dict(
            self.processor.prepare_training_batch(
                audio=waves,
                text=[f["text"] for f in features],
                language=[f.get("language") or self.language for f in features],
                audio_type=[f.get("audio_type", "speech") for f in features]
                if self.processor.config.audio_type_version
                else None,
            )
        )
        if not raw_audio and augmenter is not None and augmenter.has_spec_augmentation:
            mel = batch["input_features"]
            if variant == "audturbo":
                mel = mel.transpose(1, 2)
            spec_mask = batch["input_features_mask"].clone()
            if self.processor.config.audio_type_version:
                for i, feature in enumerate(features):
                    if feature.get("audio_type") == "silence" or (
                        feature.get("audio_type") in ("music", "noise")
                        and not getattr(augmenter, "supports_audio_types", False)
                    ):
                        spec_mask[i] = 0
            mel = augmenter.augment_features(mel, spec_mask)
            batch["input_features"] = mel.transpose(1, 2) if variant == "audturbo" else mel
        self._validate_audio_contract(batch, len(features))
        allowed = set(self.processor.model_input_names) | {"labels"}
        return {k: v for k, v in batch.items() if k in allowed}
