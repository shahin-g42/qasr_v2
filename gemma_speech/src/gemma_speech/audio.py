"""24 kHz log-magnitude mels compatible with vocos-mel-24khz."""

import math

import torch
from torch import nn


class LogMel(nn.Module):
    sample_rate = 24000
    hop_length = 256

    def __init__(self):
        super().__init__()
        # Torchaudio/Vocos defaults: HTK, no area normalization, 100 filters, power=1.
        frequencies = torch.linspace(0, 12000, 513)
        mel_points = torch.linspace(0, 2595 * math.log10(1 + 12000 / 700), 102)
        hz = 700 * (10 ** (mel_points / 2595) - 1)
        differences = hz[1:] - hz[:-1]
        slopes = hz[None] - frequencies[:, None]
        filters = torch.minimum(
            -slopes[:, :-2] / differences[:-1], slopes[:, 2:] / differences[1:]
        ).clamp_min(0)
        self.register_buffer("filters", filters)
        self.register_buffer("window", torch.hann_window(1024))

    def forward(self, waveform):
        spectrum = torch.stft(
            waveform.float(),
            n_fft=1024,
            hop_length=256,
            win_length=1024,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        ).abs()
        return (spectrum.transpose(-1, -2) @ self.filters).clamp_min(1e-7).log()


def load_vocoder(directory, device):
    """Explicit local asset path makes exported inference usable offline."""
    from pathlib import Path

    from vocos import Vocos

    directory = Path(directory)
    model = Vocos.from_hparams(str(directory / "config.yaml"))
    model.load_state_dict(
        torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True),
        strict=True,
    )
    return model.to(device).eval()
