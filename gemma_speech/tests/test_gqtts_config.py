"""Configuration-only regressions: no model construction or weights required."""

import json

import torch

from gemma_speech._vendor.qwen_tts.core.models.configuration_qwen3_tts import (
    Qwen3TTSConfig,
    Qwen3TTSSpeakerEncoderConfig,
)


def test_speaker_config_accepts_serialized_metadata():
    config = Qwen3TTSSpeakerEncoderConfig(dtype="float32", enc_dim=1024, sample_rate=24000)
    restored = Qwen3TTSSpeakerEncoderConfig(**json.loads(config.to_json_string()))
    assert restored.dtype == torch.float32
    assert restored.enc_dim == config.enc_dim
    assert restored.enc_channels == config.enc_channels
    assert restored.sample_rate == config.sample_rate


def test_full_generator_config_json_roundtrip():
    rope = {"rope_type": "default", "mrope_section": [24, 20, 20], "interleaved": True}
    config = Qwen3TTSConfig(
        speaker_encoder_config={"dtype": "float32", "enc_dim": 1024},
        talker_config={
            "rope_scaling": rope,
            "rope_parameters": dict(rope, rope_theta=10000),
            "code_predictor_config": {"hidden_size": 1024, "rope_theta": 321000.0},
        },
        tokenizer_type="qwen3_tts_tokenizer_12hz",
        tts_model_type="base",
        tts_model_size="0b6",
    )
    for _ in range(3):
        config = Qwen3TTSConfig(**json.loads(json.dumps(config.to_dict())))
        assert config.speaker_encoder_config.enc_dim == 1024
        assert config.speaker_encoder_config.dtype == torch.float32
        assert config.talker_config.rope_scaling == rope
        assert config.talker_config.rope_parameters["mrope_section"] == rope["mrope_section"]
        assert config.talker_config.rope_parameters["interleaved"] is True
        assert config.talker_config.code_predictor_config.hidden_size == 1024
        assert config.talker_config.code_predictor_config.rope_theta == 321000.0
        assert config.tokenizer_type == "qwen3_tts_tokenizer_12hz"
