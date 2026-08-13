from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType, is_librosa_available, logging
from transformers.utils.import_utils import requires

if is_librosa_available():
    import librosa


EPSILON = 1e-5
LOG_ZERO_GUARD_VALUE = 2**-24
LOGGER = logging.get_logger(__name__)


@requires(backends=("torch", "librosa"))
class QASRFeatureExtractor(SequenceFeatureExtractor):
    """Extract normalized 128-bin log-Mel features for the QASR encoder."""

    model_input_names: ClassVar[list[str]] = ["input_features", "attention_mask"]

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        padding_value: float = 0.0,
        dither: float = 1e-5,
        max_audio_clip_s: float = 35.0,
        overlap_chunk_second: float = 5.0,
        min_energy_window_samples: int = 1_600,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.dither = dither
        self.max_audio_clip_s = max_audio_clip_s
        self.overlap_chunk_second = overlap_chunk_second
        self.min_energy_window_samples = min_energy_window_samples
        mel_filters = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=feature_size,
            fmin=0.0,
            fmax=sampling_rate / 2,
            norm="slaney",
        )
        self.mel_filters = torch.from_numpy(mel_filters).to(torch.float32)

    def _find_split_point_energy(
        self,
        waveform: torch.Tensor,
        start_idx: int,
        end_idx: int,
    ) -> int:
        segment = waveform[start_idx:end_idx]
        if segment.shape[0] <= self.min_energy_window_samples:
            return (start_idx + end_idx) // 2

        min_energy = float("inf")
        quietest_idx = start_idx
        upper = segment.shape[0] - self.min_energy_window_samples
        for index in range(0, upper, self.min_energy_window_samples):
            window = segment[index : index + self.min_energy_window_samples]
            energy = torch.sqrt(torch.mean(window * window)).item()
            if energy < min_energy:
                min_energy = energy
                quietest_idx = start_idx + index
        return quietest_idx

    def _split_audio_chunks_energy(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        chunk_size = max(1, round(self.max_audio_clip_s * self.sampling_rate))
        boundary_context_size = max(
            1,
            round(self.overlap_chunk_second * self.sampling_rate),
        )
        total_samples = waveform.shape[0]
        if total_samples <= chunk_size:
            return [waveform]

        chunks: list[torch.Tensor] = []
        index = 0
        while index < total_samples:
            if index + chunk_size >= total_samples:
                chunks.append(waveform[index:total_samples])
                break

            search_start = max(index, index + chunk_size - boundary_context_size)
            search_end = min(index + chunk_size, total_samples)
            split_point = (
                index + chunk_size
                if search_end <= search_start
                else self._find_split_point_energy(waveform, search_start, search_end)
            )
            split_point = max(index + 1, min(split_point, total_samples))
            chunks.append(waveform[index:split_point])
            index = split_point
        return chunks

    def _apply_dither(
        self,
        waveform: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if self.dither <= 0:
            return waveform
        generator = torch.Generator(device=waveform.device)
        for index in range(waveform.shape[0]):
            valid_samples = min(
                int(audio_lengths[index].item()),
                waveform.shape[1],
            )
            if valid_samples <= 0:
                continue
            generator.manual_seed(valid_samples)
            noise = torch.randn(
                valid_samples,
                dtype=waveform.dtype,
                device=waveform.device,
                generator=generator,
            )
            waveform[index, :valid_samples] += self.dither * noise
        return waveform

    def _torch_extract_fbank_features(
        self,
        waveform: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """Extract mel features. Returns (batch, 128, time) frequency-major format."""
        window = torch.hann_window(self.win_length, periodic=False, device=device)
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            pad_mode="constant",
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
        mel_spec = self.mel_filters.to(device) @ magnitudes
        mel_spec = torch.log(mel_spec + LOG_ZERO_GUARD_VALUE)
        # Return (batch, 128, time) - frequency-major for Qwen3ASR compatibility
        return mel_spec

    def __call__(
        self,
        raw_speech: np.ndarray | list[float] | list[np.ndarray] | list[list[float]],
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_tensors: str | TensorType | None = "pt",
        return_attention_mask: bool | None = True,
        padding: str | None = "longest",
        max_length: int | None = None,
        sampling_rate: int | None = None,
        do_normalize: bool | None = None,
        device: str = "cpu",
        return_token_timestamps: bool | None = None,
        chunk_long_audio: bool = True,
        **kwargs: Any,
    ) -> BatchFeature:
        # Ignore unused params but keep return_attention_mask for compatibility
        del do_normalize, return_token_timestamps, kwargs
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"{self.__class__.__name__} expects {self.sampling_rate} Hz audio, "
                f"not {sampling_rate} Hz"
            )
        if sampling_rate is None:
            LOGGER.warning(
                "Pass sampling_rate to %s to prevent silent resampling errors.",
                self.__class__.__name__,
            )

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.from_numpy(raw_speech)
        elif isinstance(raw_speech, list) and (
            not raw_speech or isinstance(raw_speech[0], (float, int))
        ):
            raw_speech = torch.tensor(raw_speech)
        elif isinstance(raw_speech, (list, tuple)):
            raw_speech = [
                torch.from_numpy(speech) if isinstance(speech, np.ndarray) else torch.tensor(speech)
                for speech in raw_speech
            ]

        is_batched_tensor = isinstance(raw_speech, torch.Tensor) and raw_speech.ndim > 1
        if is_batched_tensor and raw_speech.ndim > 2:
            LOGGER.warning("Only mono audio is supported; averaging the channel dimension.")
            raw_speech = raw_speech.mean(-1)

        if is_batched_tensor or isinstance(raw_speech, (list, tuple)):
            waveforms = []
            for speech in raw_speech:
                if speech.ndim > 1:
                    LOGGER.warning("Only mono audio is supported; averaging the channel dimension.")
                    speech = speech.mean(-1)
                waveforms.append(speech.to(torch.float32))
        else:
            waveforms = [raw_speech.to(torch.float32)]

        chunked_waveforms: list[torch.Tensor] = []
        if chunk_long_audio:
            fast_path_threshold_s = max(
                0.0,
                self.max_audio_clip_s - self.overlap_chunk_second,
            )
            for speech in waveforms:
                duration_s = speech.shape[0] / self.sampling_rate
                if duration_s <= fast_path_threshold_s:
                    chunked_waveforms.append(speech)
                else:
                    chunked_waveforms.extend(self._split_audio_chunks_energy(speech))
        else:
            # Training path: audio is pre-validated by the collator and may be
            # slightly longer than max_audio_clip_s after speed perturbation.
            # The encoder handles variable lengths; chunking is not needed.
            chunked_waveforms = list(waveforms)

        raw_features = [speech[:, None] for speech in chunked_waveforms]
        audio_lengths = [len(speech) for speech in raw_features]
        padded_inputs = self.pad(
            BatchFeature({"input_features": raw_features, "audio_lengths": audio_lengths}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_features = padded_inputs.input_features.squeeze(-1)
        input_features = self._apply_dither(input_features, padded_inputs.audio_lengths)

        if self.preemphasis is not None:
            time_mask = torch.arange(input_features.shape[1], device=input_features.device)[
                None, :
            ] < padded_inputs.audio_lengths[:, None]
            input_features = torch.cat(
                [
                    input_features[:, :1],
                    input_features[:, 1:] - self.preemphasis * input_features[:, :-1],
                ],
                dim=1,
            )
            input_features = input_features.masked_fill(~time_mask, 0.0)

        input_features = self._torch_extract_fbank_features(input_features, device)
        # input_features is (batch, 128, time) - frequency-major
        # torch.stft runs with center=False, so the exact frame count for a
        # waveform of T samples is floor((T - n_fft) / hop) + 1. Using
        # floor(T / hop) here is a slightly conservative estimate for the
        # attention mask: it never marks padding frames as valid.
        feature_lengths = torch.floor_divide(
            padded_inputs.audio_lengths,
            self.hop_length,
        )
        # attention_mask is (batch, time)
        attention_mask = torch.arange(input_features.shape[2], device=device)[
            None, :
        ] < feature_lengths[:, None]

        # Normalize over time dimension (dim=2 for frequency-major)
        mask = attention_mask.unsqueeze(1)  # (batch, 1, time)
        masked_features = input_features * mask
        mean = masked_features.sum(dim=2) / feature_lengths.unsqueeze(-1)  # (batch, 128)
        mean = mean.unsqueeze(2)  # (batch, 128, 1)
        # Clamp the denominator: a single-frame utterance (feature_lengths==1)
        # would otherwise divide by zero and poison the normalization.
        variance = ((masked_features - mean) ** 2 * mask).sum(dim=2) / (
            feature_lengths - 1
        ).clamp(min=1).unsqueeze(-1)
        std = torch.sqrt(variance).unsqueeze(2)  # (batch, 128, 1)
        input_features = (input_features - mean) / (std + EPSILON)
        input_features *= mask

        # Permute to time-major (batch, time, num_mel_bins) to match the
        # CohereAsrFeatureExtractor convention and what the Cohere/Parakeet
        # encoder expects at its input (see modeling.get_audio_features).
        input_features = input_features.permute(0, 2, 1).contiguous()

        # Always return tensors and ensure attention_mask is present
        if return_tensors is None:
            return_tensors = "pt"
        return BatchFeature(
            data={
                "input_features": input_features,
                "attention_mask": attention_mask.to(torch.int64),
            },
            tensor_type=return_tensors,
        )


__all__ = ["QASRFeatureExtractor"]
