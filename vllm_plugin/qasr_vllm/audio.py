"""QASR audio front end and encoder tower, free of any vLLM import.

Everything here must reproduce training numerics exactly:

- :class:`QASRMelExtractor` is a line-for-line port of
  ``qasr.feature_extraction.QASRFeatureExtractor`` (pre-emphasis, deterministic
  length-seeded dither, centered STFT, slaney mel, per-utterance mean/variance
  normalization, ``floor(samples / hop)`` valid-frame count), minus chunking,
  which the training processor disables.
- :class:`QASRAudioTower` wraps transformers' ``ParakeetEncoder`` (the module
  QASR trains, with identical parameter names) plus the QASR projector.

Kept vLLM-free so ``tests/test_qasr_vllm_audio.py`` can assert parity against
the training code on a laptop.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

LOG_ZERO_GUARD_VALUE = 2**-24
EPSILON = 1e-5


def subsampling_output_lengths(
    input_lengths: torch.Tensor,
    *,
    subsampling_factor: int = 8,
    kernel_size: int = 3,
    stride: int = 2,
) -> torch.Tensor:
    """Encoder frame count after the Conformer's strided 2-D convs (``qasr.utils``)."""
    lengths = input_lengths.to(dtype=torch.long)
    padding = (kernel_size - 1) // 2
    for _ in range(int(math.log2(subsampling_factor))):
        lengths = torch.div(lengths + 2 * padding - kernel_size, stride, rounding_mode="floor") + 1
    return lengths


def _mel_filters(sampling_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
    try:
        import librosa

        filters = librosa.filters.mel(
            sr=sampling_rate, n_fft=n_fft, n_mels=n_mels, fmin=0.0,
            fmax=sampling_rate / 2, norm="slaney",
        )
    except ImportError:
        # Same slaney-scale/slaney-norm bank; transformers tests it against librosa.
        from transformers.audio_utils import mel_filter_bank

        filters = mel_filter_bank(
            num_frequency_bins=n_fft // 2 + 1, num_mel_filters=n_mels,
            min_frequency=0.0, max_frequency=sampling_rate / 2,
            sampling_rate=sampling_rate, norm="slaney", mel_scale="slaney",
        ).T
    return torch.from_numpy(np.asarray(filters)).to(torch.float32)


class QASRMelExtractor:
    """Single-utterance port of ``QASRFeatureExtractor`` (training defaults)."""

    def __init__(
        self,
        feature_size: int = 128,
        sampling_rate: int = 16_000,
        hop_length: int = 160,
        n_fft: int = 512,
        win_length: int = 400,
        preemphasis: float = 0.97,
        dither: float = 1e-5,
    ) -> None:
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.preemphasis = preemphasis
        self.dither = dither
        self.mel_filters = _mel_filters(sampling_rate, n_fft, feature_size)

    def num_frames(self, num_samples: int) -> int:
        """Valid mel frames for a waveform (the training attention-mask length)."""
        return num_samples // self.hop_length

    def __call__(self, waveform: np.ndarray | torch.Tensor) -> tuple[torch.Tensor, int]:
        """Return ``(features[valid_frames, 128] float32, valid_frames)`` for one mono waveform."""
        speech = torch.as_tensor(np.asarray(waveform), dtype=torch.float32)
        if speech.ndim > 1:
            speech = speech.mean(-1)
        speech = speech.clone()
        num_samples = int(speech.shape[0])
        if num_samples < self.hop_length:
            raise ValueError(f"audio too short: {num_samples} samples < one {self.hop_length}-sample hop")

        if self.dither > 0:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(num_samples)
            speech += self.dither * torch.randn(num_samples, dtype=torch.float32, generator=generator)
        if self.preemphasis is not None:
            speech = torch.cat([speech[:1], speech[1:] - self.preemphasis * speech[:-1]])

        window = torch.hann_window(self.win_length, periodic=False)
        stft = torch.stft(
            speech[None, :], self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window, return_complex=True,
            pad_mode="constant",
        )
        magnitudes = torch.view_as_real(stft)
        magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1)).pow(2)
        mel = torch.log(self.mel_filters @ magnitudes + LOG_ZERO_GUARD_VALUE)[0]  # (128, T)

        frames = self.num_frames(num_samples)
        mask = (torch.arange(mel.shape[1]) < frames)[None, :]
        masked = mel * mask
        mean = masked.sum(dim=1, keepdim=True) / frames
        variance = ((masked - mean) ** 2 * mask).sum(dim=1, keepdim=True) / max(frames - 1, 1)
        mel = (mel - mean) / (torch.sqrt(variance) + EPSILON)
        mel = mel * mask
        # Frames past `frames` are zeroed and masked in training; dropping them
        # is equivalent because the encoder zero-pads/masks beyond the length.
        return mel.T[:frames].contiguous(), frames


class QASRProjector(nn.Module):
    """Mirror of ``qasr.modeling.QASRMultiModalProjector`` (same parameter names)."""

    def __init__(self, encoder_size: int, text_size: int, depth: int, act: str) -> None:
        super().__init__()
        from transformers.activations import ACT2FN

        self.input_layernorm = nn.LayerNorm(encoder_size) if depth > 1 else None
        self.linear_1 = nn.Linear(encoder_size, encoder_size)
        self.act = ACT2FN[act]
        self.hidden_layers = (
            nn.ModuleList(
                nn.Sequential(nn.LayerNorm(encoder_size), nn.Linear(encoder_size, encoder_size))
                for _ in range(depth - 1)
            )
            if depth > 1
            else None
        )
        self.linear_2 = nn.Linear(encoder_size, text_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layernorm(audio_features) if self.input_layernorm is not None else audio_features
        hidden = self.act(self.linear_1(hidden))
        if self.hidden_layers is not None:
            for block in self.hidden_layers:
                hidden = hidden + self.act(block(hidden))
        return self.linear_2(hidden)


class QASRAudioTower(nn.Module):
    """Parakeet/Cohere Conformer + QASR projector: features -> decoder embeddings.

    Parameter names mirror the checkpoint under ``model.``:
    ``encoder.*`` <- ``model.audio_tower.*`` and
    ``projector.*`` <- ``model.multi_modal_projector.*``.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from transformers.models.parakeet.modeling_parakeet import ParakeetEncoder

        if int(getattr(config, "projector_pool_stride", 1)) != 1:
            raise NotImplementedError("projector_pool_stride > 1 is not supported by the vLLM plugin")
        audio_config = config.audio_config
        self.subsampling_factor = int(audio_config.subsampling_factor)
        self.kernel_size = int(audio_config.subsampling_conv_kernel_size)
        self.stride = int(audio_config.subsampling_conv_stride)
        self.encoder = ParakeetEncoder(audio_config)
        self.projector = QASRProjector(
            int(audio_config.hidden_size),
            int(config.text_config.hidden_size),
            int(getattr(config, "projector_depth", 1)),
            config.projector_hidden_act,
        )

    def num_tokens(self, num_frames: int) -> int:
        return int(subsampling_output_lengths(
            torch.tensor([num_frames]), subsampling_factor=self.subsampling_factor,
            kernel_size=self.kernel_size, stride=self.stride,
        )[0])

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Encode a list of ``(frames_i, 128)`` utterances; return one embedding per utterance."""
        param = next(self.encoder.parameters())
        lengths = torch.tensor([f.shape[0] for f in features], device=param.device)
        batch = torch.zeros(len(features), int(lengths.max()), features[0].shape[-1],
                            device=param.device, dtype=param.dtype)
        for i, f in enumerate(features):
            batch[i, : f.shape[0]] = f.to(device=param.device, dtype=param.dtype)
        mask = (torch.arange(batch.shape[1], device=param.device)[None, :] < lengths[:, None]).to(torch.long)
        hidden = self.encoder(input_features=batch, attention_mask=mask).last_hidden_state
        out_lengths = subsampling_output_lengths(
            lengths, subsampling_factor=self.subsampling_factor,
            kernel_size=self.kernel_size, stride=self.stride,
        ).tolist()
        return [self.projector(hidden[i, :n]) for i, n in enumerate(out_lengths)]
