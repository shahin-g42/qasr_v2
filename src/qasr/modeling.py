from __future__ import annotations

from typing import Any, ClassVar

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
from .utils import get_projector_pool_output_lengths


class QASRMultiModalProjector(nn.Module):
    """Project Cohere encoder states into the Qwen3 token embedding space.

    ``projector_depth == 1`` reproduces the legacy bridge exactly
    (``linear_1 -> act -> linear_2``) so old checkpoints load unchanged.
    ``projector_depth >= 2`` prepends an input LayerNorm and inserts
    ``depth - 1`` residual hidden blocks, the Voxtral-style deeper adapter:
    ``LayerNorm -> linear_1 -> act -> [LayerNorm -> Linear -> act] x (depth-1)
    -> linear_2``. Naming keeps ``linear_1``/``linear_2`` first/last so a
    legacy checkpoint loads the matching weights with ``strict=False`` while
    the new blocks start from fresh initialization.
    """

    def __init__(self, config: QASRConfig) -> None:
        super().__init__()
        encoder_size = int(config.audio_config.hidden_size)
        text_size = int(config.text_config.hidden_size)
        depth = int(getattr(config, "projector_depth", 1))
        if depth < 1:
            raise ValueError(f"projector_depth must be >= 1, got {depth}")
        self.input_layernorm = nn.LayerNorm(encoder_size) if depth > 1 else None
        self.linear_1 = nn.Linear(encoder_size, encoder_size)
        self.act = ACT2FN[config.projector_hidden_act]
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


class QASRPoolingConv(nn.Module):
    """Strided 1-D conv reducing the audio token rate along the time axis."""

    KERNEL_SIZE = 3

    def __init__(self, hidden_size: int, stride: int) -> None:
        super().__init__()
        self.stride = int(stride)
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=self.KERNEL_SIZE,
            stride=self.stride,
            padding=(self.KERNEL_SIZE - 1) // 2,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pool ``hidden_states`` (batch, time, dim) and recompute masks/lengths."""
        pooled = self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)
        output_lengths = get_projector_pool_output_lengths(
            lengths, pool_stride=self.stride, kernel_size=self.KERNEL_SIZE
        )
        output_mask = (
            torch.arange(pooled.shape[1], device=pooled.device).unsqueeze(0)
            < output_lengths.unsqueeze(1)
        )
        return pooled, output_mask, output_lengths


class QASRModel(Qwen3ASRModel):
    """Qwen3-ASR body with its Whisper encoder replaced by the Cohere Conformer."""

    config_class = QASRConfig

    def __init__(self, config: QASRConfig) -> None:
        # Calling Qwen3ASRModel.__init__ would allocate the discarded Whisper tower.
        Qwen3ASRPreTrainedModel.__init__(self, config)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.multi_modal_projector = QASRMultiModalProjector(config)
        pool_stride = int(getattr(config, "projector_pool_stride", 1))
        self.pooling_conv = (
            QASRPoolingConv(int(config.audio_config.hidden_size), pool_stride)
            if pool_stride > 1
            else None
        )
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
            encoder_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        else:
            if output_mask.shape != hidden_states.shape[:2]:
                raise ValueError(
                    "Cohere encoder output mask and hidden states disagree: "
                    f"{tuple(output_mask.shape)} vs {tuple(hidden_states.shape[:2])}"
                )
            encoder_mask = output_mask.to(device=hidden_states.device, dtype=torch.bool)
        encoder_lengths = encoder_mask.sum(-1)

        # Keep the full (b, t, d) encoder output for the CTC auxiliary head;
        # the forward override reads this together with ``encoder_lengths``.
        audio_output.encoder_states = hidden_states
        audio_output.encoder_lengths = encoder_lengths

        pool_conv = getattr(self, "pooling_conv", None)
        if pool_conv is not None:
            hidden_states, encoder_mask, encoder_lengths = pool_conv(
                hidden_states, encoder_mask, encoder_lengths
            )
            audio_output.attention_mask = encoder_mask.to(dtype=output_mask.dtype if output_mask is not None else torch.long)
            audio_output.pool_lengths = encoder_lengths

        # With device_map="auto" the encoder mask can land on a different
        # GPU than the hidden states; align devices before boolean indexing.
        valid_hidden_states = hidden_states[
            encoder_mask.to(device=hidden_states.device, dtype=torch.bool)
        ]
        audio_output.pooler_output = self.multi_modal_projector(valid_hidden_states)
        # Cache for QASRForConditionalGeneration.forward (CTC auxiliary loss).
        self._last_audio_output = audio_output
        return audio_output


class QASRForConditionalGeneration(Qwen3ASRForConditionalGeneration):
    """Causal ASR model combining a Cohere Conformer encoder and Qwen3 language model."""

    config_class = QASRConfig
    # The encoder block class is registered under its transformers library name.
    _no_split_modules: ClassVar[list[str]] = ["ParakeetEncoderBlock", "Qwen3DecoderLayer"]

    def __init__(self, config: QASRConfig) -> None:
        Qwen3ASRPreTrainedModel.__init__(self, config)
        self.model = QASRModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.ctc_loss_weight = 0.0
        self.post_init()

    def tie_weights(self, **kwargs: Any) -> None:
        super().tie_weights(**kwargs)
        # ``lm_head`` is tied to ``embed_tokens``; the CTC auxiliary head is a
        # separate module and must never be retied onto it.
        head = getattr(self.model, "ctc_head", None)
        if head is not None:
            embed = self.model.language_model.get_input_embeddings()
            if head.weight is embed.weight:
                fresh = nn.Linear(head.in_features, head.out_features, bias=False)
                fresh.weight.data.copy_(head.weight.data)
                fresh = fresh.to(device=head.weight.device, dtype=head.weight.dtype)
                self.model.ctc_head = fresh

    def add_ctc_head(self, vocab_size: int | None = None) -> nn.Linear:
        """Attach the CTC auxiliary head (training-time scaffold only).

        Targets are shifted Qwen token ids (+1, blank=0). The head is created
        after checkpoint loading so it starts from fresh initialization and old
        checkpoints remain loadable.
        """
        if vocab_size is None:
            vocab_size = int(self.config.text_config.vocab_size)
        head = nn.Linear(int(self.config.audio_config.hidden_size), vocab_size + 1, bias=False)
        self.model.ctc_head = head
        return head

    @property
    def ctc_enabled(self) -> bool:
        return self.ctc_loss_weight > 0 and getattr(self.model, "ctc_head", None) is not None

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ):
        # ``generate()`` validates model kwargs against this signature, so it
        # must expose the parent's named parameters rather than *args/**kwargs.
        outputs = super().forward(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
        if (
            not self.ctc_enabled
            or labels is None
            or not getattr(self.model, "training", False)
        ):
            return outputs

        audio_output = getattr(self.model, "_last_audio_output", None)
        if audio_output is None or getattr(audio_output, "encoder_states", None) is None:
            return outputs

        encoder_states = audio_output.encoder_states
        encoder_lengths = audio_output.encoder_lengths.to(encoder_states.device)
        # Match the head's dtype (bf16 under mixed-precision training —
        # a .float() input against a bf16 head is a hard matmul dtype error),
        # then do the loss arithmetic in float32 for numerical stability.
        ctc_logits = self.model.ctc_head(
            encoder_states.to(self.model.ctc_head.weight.dtype)
        )

        # CTC targets: contiguous unmasked labels = the shifted target tokens
        # (prompt positions are -100). Shift ids by +1 so blank can live at 0.
        # The sequence always ends with EOS, which is masked in ``labels``;
        # re-append it (shifted) so every utterance target terminates with EOS.
        eos_token_id = self.config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        eos_shifted = torch.as_tensor([int(eos_token_id) + 1])
        targets = []
        target_lengths = []
        kept_rows = []
        for row_index, (row_labels, length) in enumerate(
            zip(labels, encoder_lengths, strict=True)
        ):
            ids = row_labels[row_labels != -100]
            if len(ids) == 0:
                # Non-speech / silence samples carry empty targets; CTC cannot
                # consume them (zero-length targets are undefined). The
                # autoregressive loss still teaches "emit nothing" for these.
                continue
            if int(length.item()) < len(ids) + 1:
                # A transcript cannot fit in this audio's CTC frame budget;
                # skip the row instead of silently truncating the target.
                continue
            ids = torch.cat(
                [
                    ids + 1,
                    eos_shifted.to(dtype=ids.dtype, device=ids.device),
                ]
            )
            targets.append(ids)
            target_lengths.append(len(ids))
            kept_rows.append(row_index)
        if not targets:
            return outputs

        # Restrict logits/lengths to the kept rows so batch dimensions match
        # after the skips above.
        kept = torch.as_tensor(kept_rows, device=encoder_states.device)
        ctc_logits = ctc_logits[kept]
        encoder_lengths = encoder_lengths[kept]

        ctc_targets = torch.cat(targets)
        ctc_target_lengths = torch.as_tensor(
            target_lengths, dtype=torch.long, device=encoder_states.device
        )
        ctc_loss = nn.functional.ctc_loss(
            ctc_logits.float().log_softmax(-1).transpose(0, 1),
            ctc_targets,
            encoder_lengths,
            ctc_target_lengths,
            blank=0,
            reduction="mean",
            zero_infinity=True,
        )
        outputs.loss = outputs.loss + self.ctc_loss_weight * ctc_loss
        return outputs

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
    "QASRPoolingConv",
]
