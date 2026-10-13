"""Experimental model classes. Import explicitly; no global model registration."""

from __future__ import annotations

from typing import Any

import torch
from qasr.configuration import QASRConfig
from qasr.modeling import (
    QASRForConditionalGeneration,
    QASRMultiModalProjector,
    QASRPoolingConv,
)
from torch import nn


class LabQASRConfig(QASRConfig):
    """Distinct saved type: automatic production loaders cannot ignore extensions."""

    model_type = "qasr_lab"

    def __init__(self, **kwargs):
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)


class MaskedPoolingConv(QASRPoolingConv):
    """Keep padded encoder values out of valid convolution boundary outputs."""

    def forward(self, hidden_states, attention_mask, lengths):
        valid = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        if valid.shape != hidden_states.shape[:2]:
            raise ValueError("Pooling mask and hidden states disagree")
        expected = (
            torch.arange(valid.shape[1], device=valid.device)[None]
            < lengths.to(valid.device)[:, None]
        )
        if not torch.equal(valid, expected):
            raise ValueError("Pooling requires contiguous, right-padded masks matching lengths")
        hidden_states = hidden_states.masked_fill(~valid.unsqueeze(-1), 0)
        return super().forward(hidden_states, attention_mask, lengths)


class ResidualBridge(QASRMultiModalProjector):
    """Preserve the original bridge and add zero-initialized output residuals."""

    def __init__(self, config):
        super().__init__(config)
        width = int(config.text_config.hidden_size)
        bottleneck = int(config.lab_bridge_dim)
        self.lab_residual = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, width),
        )
        self.reset_residual()

    def reset_residual(self):
        nn.init.zeros_(self.lab_residual[-1].weight)
        nn.init.zeros_(self.lab_residual[-1].bias)

    def forward(self, audio_features):
        baseline = super().forward(audio_features)
        return baseline + self.lab_residual(baseline)


def transcript_vocabulary(tokenizer, samples, *, maximum=16384):
    special = set(tokenizer.all_special_ids)
    ids = set()
    for sample in samples:
        encoded = tokenizer.encode(sample["text"], add_special_tokens=False)
        if special.intersection(encoded):
            raise ValueError(f"Transcript contains a reserved token: {sample['id']}")
        ids.update(encoded)
    if not ids or len(ids) > maximum:
        raise ValueError(f"Compact CTC vocabulary has {len(ids)} tokens; expected 1..{maximum}")
    return sorted(ids)


def compact_targets(tokenizer, texts, vocabulary):
    mapping = {token: index + 1 for index, token in enumerate(vocabulary)}
    rows = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        try:
            rows.append([mapping[token] for token in ids])
        except KeyError as exc:
            raise ValueError("Transcript token is absent from the frozen CTC vocabulary") from exc
    padded = torch.full((len(rows), max(map(len, rows), default=0)), -100, dtype=torch.long)
    for index, row in enumerate(rows):
        padded[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return padded


def transcript_ctc_loss(logits, lengths, targets):
    """Transcript-only CTC, including blank-only silence and repeat feasibility."""
    if targets.ndim != 2 or targets.shape[0] != logits.shape[0]:
        raise ValueError("CTC targets must have shape (batch, padded target length)")
    rows, keep, target_lengths = [], [], []
    for index, row in enumerate(targets):
        ids = row[row != -100]
        if ids.numel() and (int(ids.min()) < 1 or int(ids.max()) >= logits.shape[-1]):
            raise ValueError("CTC targets must be compact vocabulary IDs; blank=0 is reserved")
        repeats = int((ids[1:] == ids[:-1]).sum())
        frames = int(lengths[index])
        if frames < 1 or frames > logits.shape[1]:
            raise ValueError("Invalid encoder length for CTC")
        if frames < len(ids) + repeats:
            continue
        rows.append(ids)
        target_lengths.append(len(ids))
        keep.append(index)
    stats = {
        "ctc_rows": len(targets),
        "ctc_feasible_rows": len(keep),
        "ctc_infeasible_rows": len(targets) - len(keep),
        "ctc_empty_rows": sum(length == 0 for length in target_lengths),
    }
    if not keep:
        raise ValueError("No CTC-feasible rows in batch; inspect target/frame lengths")
    selected = logits[keep].float().log_softmax(-1).transpose(0, 1)
    loss = nn.functional.ctc_loss(
        selected,
        torch.cat(rows).to(logits.device),
        lengths[keep].to(dtype=torch.long, device="cpu"),
        torch.tensor(target_lengths, dtype=torch.long),
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite CTC loss; this experiment does not hide invalid paths")
    return loss, stats


class LabQASRForConditionalGeneration(QASRForConditionalGeneration):
    """Baseline-compatible checkpoint format with explicit lab-only extensions."""

    config_class = LabQASRConfig

    def __init__(self, config):
        if not isinstance(config, LabQASRConfig):
            original = config
            config = LabQASRConfig(**original.to_dict())
            config._attn_implementation = original._attn_implementation
        super().__init__(config)
        if getattr(config, "lab_mask_pool_padding", False) and self.model.pooling_conv is not None:
            old = self.model.pooling_conv
            new = MaskedPoolingConv(int(config.audio_config.hidden_size), old.stride)
            new.load_state_dict(old.state_dict())
            self.model.pooling_conv = new
        if int(getattr(config, "lab_bridge_dim", 0)) > 0:
            old = self.model.multi_modal_projector
            new = ResidualBridge(config)
            new.load_state_dict(old.state_dict(), strict=False)
            self.model.multi_modal_projector = new
        vocabulary = getattr(config, "lab_ctc_vocabulary", [])
        if vocabulary != sorted(set(vocabulary)):
            raise ValueError("lab_ctc_vocabulary must be sorted and unique")
        if vocabulary and (min(vocabulary) < 0 or max(vocabulary) >= config.text_config.vocab_size):
            raise ValueError("CTC vocabulary contains an invalid decoder token ID")
        self.lab_ctc_head = (
            nn.Linear(int(config.audio_config.hidden_size), len(vocabulary) + 1, bias=False)
            if vocabulary
            else None
        )
        self.lab_ctc_weight = float(getattr(config, "lab_ctc_weight", 0.0))
        if self.lab_ctc_weight < 0 or (self.lab_ctc_weight and self.lab_ctc_head is None):
            raise ValueError("Positive lab_ctc_weight requires a compact vocabulary")
        self.lab_ctc_stats = {}
        # The production CTC path is deliberately disabled on this instance.
        self.ctc_loss_weight = 0.0

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        lab_ctc_targets: torch.LongTensor | None = None,
        **kwargs,
    ):
        needs_ctc = bool(self.training and labels is not None and self.lab_ctc_weight)
        if needs_ctc:
            if lab_ctc_targets is None or input_features is None:
                raise ValueError("CTC training requires audio and transcript-only lab_ctc_targets")
            self.model._last_audio_output = None
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
        if needs_ctc:
            audio = self.model._last_audio_output
            if audio is None:
                raise RuntimeError("This forward did not produce encoder states")
            logits = self.lab_ctc_head(audio.encoder_states.to(self.lab_ctc_head.weight.dtype))
            loss, stats = transcript_ctc_loss(logits, audio.encoder_lengths, lab_ctc_targets)
            outputs.loss = outputs.loss + self.lab_ctc_weight * loss
            self.lab_ctc_stats = {**stats, "ctc_loss": float(loss.detach())}
        return outputs


def configure_trainable(model, mode, *, upper_layers=4):
    if mode not in ("projector", "upper_encoder", "full"):
        raise ValueError("trainable must be projector, upper_encoder, or full")
    model.requires_grad_(mode == "full")
    model.model.multi_modal_projector.requires_grad_(True)
    if model.model.pooling_conv is not None:
        model.model.pooling_conv.requires_grad_(True)
    if model.lab_ctc_head is not None:
        model.lab_ctc_head.requires_grad_(True)
    if mode == "upper_encoder":
        layers = model.model.audio_tower.layers
        if not 1 <= upper_layers <= len(layers):
            raise ValueError("upper_layers must fit the encoder depth")
        for layer in layers[-upper_layers:]:
            layer.requires_grad_(True)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
