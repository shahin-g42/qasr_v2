"""Frame-preserving EmbeddingGemma audio tower with a compact CTC reader."""

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .layers import ContextBlock, sinusoidal


def ctc_loss(logits, mask, targets, target_mask):
    lengths, target_lengths = mask.sum(-1), target_mask.sum(-1)
    repeats = ((targets[:, 1:] == targets[:, :-1]) & target_mask[:, 1:]).sum(-1)
    if torch.any(lengths < target_lengths + repeats):
        raise ValueError(
            "CTC alignment impossible: frames < tokens + adjacent repeats. "
            "Use a larger subword vocabulary or asr_upsample=2; do not truncate text."
        )
    if torch.any(targets[target_mask] <= 0) or torch.any(targets[target_mask] >= logits.shape[-1]):
        raise ValueError("CTC targets must exclude blank and stay inside the vocabulary")
    # FP32 log_softmax + loss is required even when the encoder uses BF16 autocast.
    return F.ctc_loss(
        logits.float().log_softmax(-1).transpose(0, 1),
        targets[target_mask],
        lengths,
        target_lengths,
        blank=0,
        reduction="mean",
        zero_infinity=False,
    )


def greedy_tokens(logits, mask):
    result = []
    for row, valid in zip(logits.argmax(-1), mask, strict=True):
        collapsed = torch.unique_consecutive(row[valid]).tolist()
        result.append([token for token in collapsed if token != 0])
    return result


class ASRModel(nn.Module):
    def __init__(self, component, config, vocab_size):
        super().__init__()
        self.config = config
        self.backbone = Backbone(component, config.train_backbone_layers)
        self.projection = nn.Linear(
            component.config.output_proj_dims, config.width * config.asr_upsample
        )
        self.blocks = nn.ModuleList(
            [ContextBlock(config.width, config.heads, config.dropout) for _ in range(config.layers)]
        )
        self.norm = nn.LayerNorm(config.width)
        self.classifier = nn.Linear(config.width, vocab_size)
        self.midpoint = config.layers // 2 - 1 if config.layers >= 2 else -1
        self.self_condition = (
            nn.Linear(vocab_size, config.width, bias=False)
            if self.midpoint >= 0 and config.intermediate_ctc_weight
            else None
        )

    def forward(self, input_features, input_features_mask, targets=None, target_mask=None):
        features = self.backbone(input_features=input_features, attention_mask=input_features_mask)
        mask = features.attention_mask.bool()
        if torch.any(mask.sum(-1) == 0):
            raise ValueError("audio produced no valid frames")
        # The native output is 1536-dimensional audio frames, not the pooled 768d embedding.
        x = self.projection(features.last_hidden_state)
        x = x.reshape(x.shape[0], -1, self.config.width)
        mask = mask.repeat_interleave(self.config.asr_upsample, dim=1)
        x = x + sinusoidal(torch.arange(x.shape[1], device=x.device), x.shape[-1]).to(x.dtype)
        x = x.masked_fill(~mask[..., None], 0)
        intermediate = None
        for i, block in enumerate(self.blocks):
            x = block(x, mask)
            if i == self.midpoint and self.self_condition is not None:
                intermediate = self.classifier(self.norm(x))
                x = x + self.self_condition(intermediate.softmax(-1))
                x = x.masked_fill(~mask[..., None], 0)
        logits = self.classifier(self.norm(x))
        result = {"logits": logits, "mask": mask}
        if targets is not None:
            loss = ctc_loss(logits, mask, targets, target_mask)
            if intermediate is not None:
                weight = self.config.intermediate_ctc_weight
                loss = (1 - weight) * loss + weight * ctc_loss(
                    intermediate, mask, targets, target_mask
                )
            result["loss"] = loss
        return result
