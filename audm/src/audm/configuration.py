"""Local, versioned configuration; upstream sources are provenance, not dependencies."""

from transformers import Gemma4AudioConfig, PreTrainedConfig, Qwen3Config
from transformers.models.moonshine_streaming.configuration_moonshine_streaming import (
    MoonshineStreamingEncoderConfig,
)
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)


class AUDTextConfig(Qwen3Config):
    model_type = "aud_text"


class AUDCohereConfig(ParakeetEncoderConfig):
    model_type = "aud_cohere"


class AUDOmniConfig(Qwen3OmniMoeAudioEncoderConfig):
    model_type = "aud_omni"


class AUDMoonshineConfig(MoonshineStreamingEncoderConfig):
    model_type = "aud_moonshine"


class AUDGemma4Config(Gemma4AudioConfig):
    model_type = "aud_gemma4_audio"


ENCODERS = {"audpro": AUDCohereConfig, "audturbo": AUDOmniConfig, "audflash": AUDMoonshineConfig}


def clean_config(value):
    data = value.to_dict() if isinstance(value, PreTrainedConfig) else dict(value or {})
    for key in ("model_type", "architectures", "auto_map", "_name_or_path", "transformers_version"):
        data.pop(key, None)
    return data


class AUDConfig(PreTrainedConfig):
    model_type = "aud"
    sub_configs = {"text_config": AUDTextConfig, "audio_config": PreTrainedConfig}

    def __init__(
        self,
        variant="audpro",
        audio_config=None,
        text_config=None,
        audio_token_id=151676,
        projector_depth=1,
        projector_hidden_act="gelu",
        projector_pool_stride=1,
        adapter_width=384,
        flash_encoder="moonshine",
        flash_audio_stride=1,
        ctc_vocab_size=0,
        prompt_prefix=None,
        prompt_suffix=None,
        max_audio_seconds=35.0,
        requires_alignment=False,
        format_version=1,
        audio_type_version=0,
        audio_type_token_ids=None,
        **kwargs,
    ):
        if variant not in ENCODERS or format_version != 1:
            raise ValueError("Unsupported AUD variant or format version")
        if audio_type_version not in (0, 1):
            raise ValueError("Unsupported audio-type target version")
        self.audio_type_version = audio_type_version
        self.audio_type_token_ids = dict(audio_type_token_ids or {})
        if audio_type_version == 1:
            from .audio_types import TOKENS

            ids = list(self.audio_type_token_ids.values())
            if (
                set(self.audio_type_token_ids) != set(TOKENS)
                or len(set(ids)) != 4
                or any(type(i) is not int or i < 0 for i in ids)
            ):
                raise ValueError("Invalid audio-type token map")
        elif self.audio_type_token_ids:
            raise ValueError("Audio-type IDs require version 1")
        self.variant = variant
        if flash_encoder not in ("moonshine", "gemma4") or flash_audio_stride not in (1, 2):
            raise ValueError("Unsupported Flash encoder or temporal stride")
        self.flash_encoder = flash_encoder
        self.flash_audio_stride = int(flash_audio_stride)
        audio_cls = (
            AUDGemma4Config
            if variant == "audflash" and flash_encoder == "gemma4"
            else ENCODERS[variant]
        )
        self.audio_config = audio_cls(**clean_config(audio_config))
        self.text_config = AUDTextConfig(**clean_config(text_config))
        self.audio_token_id = int(audio_token_id)
        self.projector_depth = int(projector_depth)
        self.projector_hidden_act = projector_hidden_act
        self.projector_pool_stride = int(projector_pool_stride)
        self.adapter_width = int(adapter_width)
        self.ctc_vocab_size = int(ctc_vocab_size)
        self.prompt_prefix = list(prompt_prefix or [])
        self.prompt_suffix = list(prompt_suffix or [])
        self.max_audio_seconds = float(max_audio_seconds)
        self.requires_alignment = bool(requires_alignment)
        self.format_version = format_version
        if min(self.projector_depth, self.projector_pool_stride, self.adapter_width) < 1:
            raise ValueError("Projector dimensions/stride must be positive")
        if variant != "audpro" and self.projector_pool_stride != 1:
            raise ValueError("Turbo and Flash exports preserve the encoder's native token rate")
        kwargs.setdefault("pad_token_id", self.text_config.pad_token_id)
        kwargs.setdefault("eos_token_id", self.text_config.eos_token_id)
        kwargs["tie_word_embeddings"] = self.text_config.tie_word_embeddings
        kwargs.setdefault("is_encoder_decoder", False)
        super().__init__(**kwargs)
