"""Inference-side config for QASR checkpoints (``model_type: "qasr"``).

Parses the ``config.json`` written by ``QASRForConditionalGeneration.save_pretrained``
without importing the training package, whose pinned transformers version the
vLLM image may not match.
"""

from __future__ import annotations

from typing import Any, ClassVar

from transformers import PretrainedConfig, Qwen3Config
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig


class QASRVllmConfig(PretrainedConfig):
    model_type = "qasr"
    sub_configs: ClassVar[dict[str, type]] = {"audio_config": ParakeetEncoderConfig, "text_config": Qwen3Config}

    def __init__(
        self,
        audio_config: dict[str, Any] | ParakeetEncoderConfig | None = None,
        text_config: dict[str, Any] | Qwen3Config | None = None,
        audio_token_id: int = 151676,
        projector_hidden_act: str = "gelu",
        projector_depth: int = 1,
        projector_pool_stride: int = 1,
        tie_word_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        if isinstance(audio_config, dict):
            audio_config = {k: v for k, v in audio_config.items() if k != "model_type"}
            audio_config = ParakeetEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = ParakeetEncoderConfig()
        if isinstance(text_config, dict):
            text_config = {k: v for k, v in text_config.items() if k != "model_type"}
            text_config = Qwen3Config(**text_config)
        elif text_config is None:
            text_config = Qwen3Config()
        # The checkpoint stores only embed_tokens (lm_head is tied at the top
        # level). vLLM builds the decoder from text_config alone, so the tie
        # must live there or it would allocate an untrained lm_head.
        text_config.tie_word_embeddings = bool(tie_word_embeddings)
        self.audio_config = audio_config
        self.text_config = text_config
        self.audio_token_id = int(audio_token_id)
        self.projector_hidden_act = projector_hidden_act
        self.projector_depth = int(projector_depth)
        self.projector_pool_stride = int(projector_pool_stride)
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, *args: Any, **kwargs: Any) -> Qwen3Config:
        return self.text_config
