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
        projector_depth: int = 1,
        projector_pool_stride: int = 1,
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
        # Depth of the encoder->LLM bridge MLP. 1 reproduces the legacy
        # two-layer projector (linear_1 -> act -> linear_2); >= 2 adds an
        # input LayerNorm and residual hidden blocks for a stronger bridge.
        # Default is 1 for checkpoint compatibility: set projector_depth=2 in
        # the model's config.json before the Phase 1 deep-projector retrain.
        self.projector_depth = int(projector_depth)
        # Optional strided-conv token downsampling inside the projector
        # (Tier B). 1 keeps the encoder's native 12.5 Hz token rate; 2 halves
        # LLM prefill cost. Any value > 1 requires processor/length plumbing
        # (QASRProcessor.projector_pool_stride) to stay in sync.
        self.projector_pool_stride = int(projector_pool_stride)
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
        if self.projector_depth < 1:
            raise ValueError("projector_depth must be at least 1")
        if self.projector_pool_stride < 1:
            raise ValueError("projector_pool_stride must be at least 1")


__all__ = ["CohereEncoderConfig", "QASRConfig"]
