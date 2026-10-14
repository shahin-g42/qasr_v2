"""Standalone GQTTS base: acoustic encoder, speaker adapter, generator and codec."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_model, save_model
from torch import nn
from transformers import Gemma4AudioConfig, Gemma4AudioFeatureExtractor, Gemma4AudioModel

from gemma_speech._vendor.qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSProcessor,
)
from gemma_speech._vendor.qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
)
from gemma_speech._vendor.qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Model,
)
from gemma_speech._vendor.qwen_tts.inference.qwen3_tts_model import (
    Qwen3TTSModel,
    VoiceClonePromptItem,
)
from gemma_speech._vendor.qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

from .speaker import SpeakerAdapter, audio_frames

_RENAMED_CLASSES = {}


def public_class_name(original):
    """Keep acoustic, generator and codec implementations in distinct namespaces."""
    name = original.__name__
    for prefix, replacement in (
        ("Gemma4Audio", "GQTTSAcoustic"),
        ("Gemma4", "GQTTSAcousticPrimitive"),
        ("Qwen3TTSTokenizer", "GQTTSCodec"),
        ("Qwen3TTS", "GQTTSGenerator"),
    ):
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return "GQTTSSpeakerAdapter" if original is SpeakerAdapter else None


def name_modules(module):
    for child in module.modules():
        original = type(child)
        name = public_class_name(original)
        if name is None:
            continue
        renamed = _RENAMED_CLASSES.get(original)
        if renamed is None:
            existing = globals().get(name)
            if existing is not None:
                raise ValueError(f"GQTTS class alias collision: {name} for {original}")
            renamed = type(name, (original,), {"__module__": __name__})
            _RENAMED_CLASSES[original] = renamed
            globals()[name] = renamed
        # Never substitute the implementation of another source class.
        if renamed.__bases__ != (original,):
            raise TypeError(f"GQTTS alias has incorrect implementation: {name}")
        child.__class__ = renamed


class GQTTSBase(nn.Module):
    def __init__(self, config, acoustic=None, generator=None, codec=None, adapter=None):
        super().__init__()
        if config.get("model_type") != "gqtts" or config.get("format_version") != 1:
            raise ValueError("unsupported GQTTS base format")
        self.config = config
        self.gqtts_acoustic = (
            acoustic
            if acoustic is not None
            else Gemma4AudioModel(Gemma4AudioConfig(**config["acoustic_config"]))
        )
        self.gqtts_generator = (
            generator
            if generator is not None
            else Qwen3TTSForConditionalGeneration(Qwen3TTSConfig(**config["generator_config"]))
        )
        # Teacher speaker encoder is not part of the replacement inference model.
        self.gqtts_generator.speaker_encoder = None
        self.gqtts_codec = (
            codec
            if codec is not None
            else Qwen3TTSTokenizerV2Model(Qwen3TTSTokenizerV2Config(**config["codec_config"]))
        )
        self.gqtts_speaker_adapter = (
            adapter if adapter is not None else SpeakerAdapter(**config["adapter_dimensions"])
        )
        name_modules(self)

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "config.json").write_text(json.dumps(self.config, indent=2) + "\n")
        save_model(
            self,
            str(directory / "model.safetensors"),
            metadata={"format": "pt", "model_type": "gqtts"},
        )

    @classmethod
    def from_pretrained(cls, directory, device="cpu"):
        from transformers import AutoFeatureExtractor

        directory = Path(directory)
        model = cls(json.loads((directory / "config.json").read_text()))
        load_model(model, str(directory / "model.safetensors"), strict=True)
        model.to(device).eval()
        tokenizer = Qwen3TTSTokenizer()
        tokenizer.model = model.gqtts_codec
        tokenizer.config = model.gqtts_codec.config
        tokenizer.device = torch.device(device)
        tokenizer.feature_extractor = AutoFeatureExtractor.from_pretrained(
            directory / "codec_frontend", local_files_only=True
        )
        model.gqtts_generator.load_speech_tokenizer(tokenizer)
        defaults = model.config["generate_defaults"]
        model.gqtts_generator.load_generate_config(defaults)
        processor = Qwen3TTSProcessor.from_pretrained(
            directory / "processor", local_files_only=True
        )
        model.api = Qwen3TTSModel(model.gqtts_generator, processor, defaults)
        model.frontend = Gemma4AudioFeatureExtractor.from_pretrained(
            directory / "acoustic_frontend", local_files_only=True
        )
        return model

    @torch.no_grad()
    def synthesize(self, reference_wave16, text, language="Auto", **kwargs):
        device = next(self.parameters()).device
        frames = audio_frames(self.gqtts_acoustic, self.frontend, reference_wave16, device)
        valid = torch.ones(1, len(frames), device=device, dtype=torch.bool)
        vector = self.gqtts_speaker_adapter(frames[None], valid)[0]
        prompt = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=vector,
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text=None,
        )
        return self.api.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=[prompt],
            non_streaming_mode=True,
            **kwargs,
        )
