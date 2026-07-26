from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.activations import ACT2FN
from transformers.models.qwen3_asr.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRModel,
    Qwen3ASRPreTrainedModel,
)

from .configuration import QASRConfig


class QASRMultiModalProjector(nn.Module):
    """Project Cohere encoder states into the Qwen3 token embedding space."""

    def __init__(self, config: QASRConfig) -> None:
        super().__init__()
        encoder_size = int(config.audio_config.hidden_size)
        text_size = int(config.text_config.hidden_size)
        self.linear_1 = nn.Linear(encoder_size, encoder_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(encoder_size, text_size)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(audio_features)))


class QASRModel(Qwen3ASRModel):
    """Qwen3-ASR body with its Whisper encoder replaced by the Cohere Conformer."""

    config_class = QASRConfig

    def __init__(self, config: QASRConfig) -> None:
        # Calling Qwen3ASRModel.__init__ would allocate the discarded Whisper tower.
        Qwen3ASRPreTrainedModel.__init__(self, config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.multi_modal_projector = QASRMultiModalProjector(config)
        self.post_init()

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        input_features_mask: torch.LongTensor | None,
        **kwargs: Any,
    ):
        if input_features.ndim != 3 or input_features.shape[-1] != self.config.audio_config.num_mel_bins:
            raise ValueError(
                "QASR input_features must have shape (batch, frames, 128), got "
                f"{tuple(input_features.shape)}"
            )
        if input_features_mask is None:
            raise ValueError("QASR requires input_features_mask so padded encoder states are not injected")
        if input_features_mask.shape != input_features.shape[:2]:
            raise ValueError(
                "input_features_mask must match the batch and frame dimensions: "
                f"{tuple(input_features_mask.shape)} vs {tuple(input_features.shape[:2])}"
            )
        audio_output = self.audio_tower(
            input_features=input_features,
            attention_mask=input_features_mask,
            **kwargs,
        )
        hidden_states = audio_output.last_hidden_state
        output_mask = getattr(audio_output, "attention_mask", None)
        if output_mask is None:
            valid_hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        else:
            if output_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "Cohere encoder output mask and hidden states disagree: "
                    f"{tuple(output_mask.shape)} vs {tuple(hidden_states.shape[:2])}"
                )
            # With device_map="auto" the encoder mask can land on a different
            # GPU than the hidden states; align devices before boolean indexing.
            valid_hidden_states = hidden_states[
                output_mask.to(device=hidden_states.device, dtype=torch.bool)
            ]
        audio_output.pooler_output = self.multi_modal_projector(valid_hidden_states)
        return audio_output


class QASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Causal ASR model combining a Cohere Conformer encoder and Qwen3 language model."""

    config_class = QASRConfig
    # The encoder block class is registered under its transformers library name.
    _no_split_modules = ["ParakeetEncoderBlock", "Qwen3DecoderLayer"]

    def __init__(self, config: QASRConfig) -> None:
        Qwen3ASRPreTrainedModel.__init__(self, config)
        self.model = QASRModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        # A frozen encoder has no activation graph to retain, so checkpointing
        # it only causes an unnecessary recomputation attempt (and reentrant
        # checkpointing requires a grad-bearing input). The frozen decoder must
        # remain checkpointed because gradients still flow through its inputs
        # into the trainable multimodal projector.
        if not any(parameter.requires_grad for parameter in self.model.audio_tower.parameters()):
            self.model.audio_tower.gradient_checkpointing_disable()


__all__ = [
    "QASRForConditionalGeneration",
    "QASRModel",
    "QASRMultiModalProjector",
]
