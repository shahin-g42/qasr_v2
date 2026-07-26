from __future__ import annotations

from typing import Any

from transformers import Qwen3ASRConfig
from transformers.models.parakeet.configuration_parakeet import ParakeetEncoderConfig

# The Cohere Transcribe encoder shares the same Conformer architecture that
# transformers registers under the "parakeet_encoder" model type.  Expose a
# project-local alias so all QASR code references "Cohere" consistently.
CohereEncoderConfig = ParakeetEncoderConfig


class QASRConfig(Qwen3ASRConfig):
    """Configuration for a Cohere Conformer encoder connected to the Qwen3-ASR LLM."""

    model_type = "qasr"

    def __init__(
        self,
        *,
        audio_config: dict[str, Any] | CohereEncoderConfig | None = None,
        text_config: dict[str, Any] | None = None,
        projector_hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        if audio_config is None:
            audio_config = CohereEncoderConfig(
                hidden_size=1280,
                num_hidden_layers=48,
                num_attention_heads=8,
                intermediate_size=5120,
                hidden_act="silu",
                attention_bias=True,
                convolution_bias=True,
                conv_kernel_size=9,
                subsampling_factor=8,
                subsampling_conv_channels=256,
                num_mel_bins=128,
                subsampling_conv_kernel_size=3,
                subsampling_conv_stride=2,
                dropout=0.0,
                dropout_positions=0.0,
                layerdrop=0.0,
                activation_dropout=0.0,
                attention_dropout=0.0,
                max_position_embeddings=5000,
                scale_input=False,
            )
        super().__init__(audio_config=audio_config, text_config=text_config, **kwargs)
        self.projector_hidden_act = projector_hidden_act
        self._validate_hybrid_config()

    def _validate_hybrid_config(self) -> None:
        # transformers registers the Cohere Conformer encoder under the
        # "parakeet_encoder" model type; accept it as the canonical identifier.
        if self.audio_config.model_type != "parakeet_encoder":
            raise ValueError(
                "QASR requires a Cohere Conformer encoder "
                f"(audio_config.model_type='parakeet_encoder'), got "
                f"{self.audio_config.model_type!r}"
            )
        if self.audio_config.num_mel_bins != 128:
            raise ValueError("QASR requires 128-bin Conformer input features")
        factor = int(self.audio_config.subsampling_factor)
        if factor < 1 or factor & (factor - 1):
            raise ValueError("audio_config.subsampling_factor must be a positive power of two")
        if self.text_config.hidden_size <= 0 or self.audio_config.hidden_size <= 0:
            raise ValueError("Encoder and text hidden sizes must be positive")


__all__ = ["CohereEncoderConfig", "QASRConfig"]
