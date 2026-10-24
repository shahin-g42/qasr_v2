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


def gemma_specaugment(features, mask, augmenter):
    """Use GQASR's per-frequency mean of original valid frames as the fill.

    Sample the unchanged policy on ones to preserve clean gates and mask draws.
    Never compute the mean from already-masked features or include padding.
    """
    keep = augmenter.augment_features(torch.ones_like(features), mask).bool()
    valid = mask.bool().unsqueeze(-1)
    fill = features.masked_fill(~valid, 0).sum(dim=1, keepdim=True)
    fill = fill / valid.sum(dim=1, keepdim=True).clamp_min(1)
    selected = ~keep & valid
    return torch.where(selected, fill, features)


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
        if getattr(self, "non_speech_pool", None) is not None:
            features = self.non_speech_pool.replace(features)
        audits = []
        if self.code_switcher is not None:
            mixed = [
                self.code_switcher.augment(f, self.code_switch_probability)
                if f["text"].strip()
                else (f, {})
                for f in features
            ]
            features, audits = [m[0] for m in mixed], [m[1] for m in mixed]
        if self.audit_callback is not None:
            self.audit_callback(features, audits)
        # Keep the stage-02/index contract: this limit applies to transcript
        # tokens. Language markers and EOS are additional supervised tokens.
        if any(
            len(
                f["transcript_ids"]
                if "transcript_ids" in f and not audits
                else self.processor.tokenizer.encode(f["text"], add_special_tokens=False)
            )
            > self.max_target_length
            for f in features
        ):
            raise ValueError("Transcript exceeds max_target_length")
        waves = [self._load_waveform(f) for f in features]
        augmenter = self.augmenter
        if augmenter is not None and augmenter.has_waveform_augmentation:
            augmented = []
            for wave, feature in zip(waves, features, strict=True):
                kind = feature.get("audio_type") or (
                    "speech" if feature["text"].strip() else "non_speech"
                )
                supports_types = getattr(augmenter, "supports_audio_types", False)
                if kind in ("music", "noise", "silence", "non_speech") and not supports_types:
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
                if not f["text"].strip() and not getattr(augmenter, "supports_audio_types", False)
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
            for i, feature in enumerate(features):
                if feature.get("audio_type") == "silence" or (
                    not feature["text"].strip()
                    and not getattr(augmenter, "supports_audio_types", False)
                ):
                    spec_mask[i] = 0
            if variant == "audflash" and self.processor.config.flash_encoder == "gemma4":
                mel = gemma_specaugment(mel, spec_mask, augmenter)
            else:
                mel = augmenter.augment_features(mel, spec_mask)
            batch["input_features"] = mel.transpose(1, 2) if variant == "audturbo" else mel
        self._validate_audio_contract(batch, len(features))
        allowed = set(self.processor.model_input_names) | {"labels"}
        if self.processor.config.ctc_version:
            sequences = [
                self.processor.encode_ctc(
                    f["text"], f.get("transcript_ids") if not audits else None
                )
                if f.get("audio_type", "speech") == "speech" and f["text"].strip()
                else []
                for f in features
            ]
            targets = torch.full(
                (len(features), max(1, max(map(len, sequences)))), -100, dtype=torch.long
            )
            for i, ids in enumerate(sequences):
                targets[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            batch["ctc_targets"] = targets
            batch["ctc_input_lengths"] = self.processor.audio_token_lengths(
                batch["input_features_mask"]
            )
            allowed |= {"ctc_targets", "ctc_input_lengths"}
        return {k: v for k, v in batch.items() if k in allowed}
