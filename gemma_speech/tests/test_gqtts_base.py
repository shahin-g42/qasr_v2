"""Artifact contract tests; the cluster verify stage exercises pretrained modules."""

from types import SimpleNamespace

import pytest
import torch
from gqtts.base import GQTTSBase
from torch import nn

from gqtts import base as base_module


class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.talker = nn.Linear(4, 4)
        self.speaker_encoder = nn.Linear(4, 4)

    def load_speech_tokenizer(self, tokenizer):
        self.speech_tokenizer = tokenizer

    def load_generate_config(self, defaults):
        self.generate_config = defaults


class Codec(nn.Linear):
    def __init__(self, config):
        super().__init__(4, 4)
        self.config = config


def test_gqtts_artifact_roundtrip_excludes_teacher(tmp_path, monkeypatch):
    import transformers

    config = dict(
        model_type="gqtts",
        format_version=1,
        acoustic_config={},
        generator_config={},
        codec_config={},
        adapter_dimensions=dict(input_dim=4, width=4, output_dim=4),
        generate_defaults={},
    )
    for name in ("Gemma4AudioConfig", "Qwen3TTSConfig", "Qwen3TTSTokenizerV2Config"):
        monkeypatch.setattr(base_module, name, lambda **kwargs: kwargs)
    monkeypatch.setattr(base_module, "Gemma4AudioModel", lambda config: nn.Linear(4, 4))
    monkeypatch.setattr(base_module, "Qwen3TTSForConditionalGeneration", Generator)
    monkeypatch.setattr(base_module, "Qwen3TTSTokenizerV2Model", Codec)
    for cls in (
        transformers.AutoFeatureExtractor,
        base_module.Gemma4AudioFeatureExtractor,
        base_module.Qwen3TTSProcessor,
    ):
        monkeypatch.setattr(cls, "from_pretrained", lambda *a, **kw: SimpleNamespace())
    original = GQTTSBase(config)
    path = tmp_path / "base"
    original.save_pretrained(path)
    restored = GQTTSBase.from_pretrained(path)
    assert restored.gqtts_generator.speaker_encoder is None
    assert restored.api.model.speech_tokenizer.model is restored.gqtts_codec
    for key, value in original.state_dict().items():
        assert key.startswith("gqtts_") and "speaker_encoder" not in key
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        original.save_pretrained(path)
