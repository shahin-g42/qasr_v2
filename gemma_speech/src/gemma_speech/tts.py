"""Experimental multilingual, known-speaker conditional flow-matching TTS.

The pretrained text tower supplies context. Lexical tokens retain pronunciation detail;
the new acoustic model learns timing and sound. This is not a released TTS checkpoint.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Backbone
from .layers import ContextBlock, average_upsample, length_mask, masked_mean, sinusoidal


class TTSModel(nn.Module):
    def __init__(self, component, config, vocab_size, speakers, languages):
        super().__init__()
        self.config = config
        width = config.width
        self.backbone = (
            Backbone(component, config.train_backbone_layers) if component is not None else None
        )
        if config.use_semantic_text != (self.backbone is not None):
            raise ValueError("semantic text configuration does not match the supplied backbone")
        self.tokens = nn.Embedding(vocab_size, width, padding_idx=0)
        self.text_blocks = nn.ModuleList(
            [ContextBlock(width, config.heads, config.dropout) for _ in range(2)]
        )
        if self.backbone is not None:
            self.semantic_projection = nn.Linear(component.config.embedding_dim, width)
            self.semantic_attention = nn.MultiheadAttention(width, config.heads, batch_first=True)
            self.semantic_gate = nn.Parameter(torch.tensor(-2.0))
        self.speakers = nn.Embedding(speakers, width)
        self.languages = nn.Embedding(languages, width)
        self.duration = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.input = nn.Linear(config.mel_bins * config.mel_pack, width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(
            [ContextBlock(width, config.heads, config.dropout) for _ in range(config.layers)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, config.mel_bins * config.mel_pack)
        )

    def encode_condition(
        self, tokens, token_mask, speaker_ids, language_ids, input_ids=None, attention_mask=None
    ):
        if torch.any(token_mask.sum(-1) == 0):
            raise ValueError("TTS text must be nonempty")
        text = self.tokens(tokens)
        text = text + sinusoidal(
            torch.arange(text.shape[1], device=text.device), text.shape[-1]
        ).to(text.dtype)
        for block in self.text_blocks:
            text = block(text, token_mask)
        if self.backbone is not None:
            if (
                input_ids is None
                or attention_mask is None
                or torch.any(attention_mask.sum(-1) == 0)
            ):
                raise ValueError("semantic conditioning needs nonempty, masked Gemma text tokens")
            semantic = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            semantic = self.semantic_projection(semantic)
            attended, _ = self.semantic_attention(
                text,
                semantic,
                semantic,
                key_padding_mask=~attention_mask.bool(),
                need_weights=False,
            )
            text = text + self.semantic_gate.sigmoid() * attended
        style = self.speakers(speaker_ids) + self.languages(language_ids)
        duration = self.duration(masked_mean(text, token_mask) + style).squeeze(-1)
        return text + style[:, None], duration

    def velocity(self, x, times, condition, mask):
        h = self.input(x) + condition
        h = h + self.time(sinusoidal(times * 1000, self.config.width).to(h.dtype))[:, None]
        h = h + sinusoidal(torch.arange(h.shape[1], device=h.device), h.shape[-1]).to(h.dtype)
        for block in self.blocks:
            h = block(h, mask)
        return self.output(h).masked_fill(~mask[..., None], 0)

    def forward(
        self,
        mel,
        mel_lengths,
        tokens,
        token_mask,
        speaker_ids,
        language_ids,
        input_ids=None,
        attention_mask=None,
    ):
        condition, log_ratio = self.encode_condition(
            tokens, token_mask, speaker_ids, language_ids, input_ids, attention_mask
        )
        pack, bins = self.config.mel_pack, self.config.mel_bins
        padded_frames = ((mel.shape[1] + pack - 1) // pack) * pack
        target = F.pad(
            (mel - self.config.mel_mean) / self.config.mel_std,
            (0, 0, 0, padded_frames - mel.shape[1]),
        )
        frame_mask = length_mask(mel_lengths, padded_frames)
        target = target.masked_fill(~frame_mask[..., None], 0)
        target = target.reshape(target.shape[0], -1, bins * pack)
        element_mask = frame_mask[..., None].expand(-1, -1, bins).reshape_as(target)
        packed_lengths = (mel_lengths + pack - 1) // pack
        mask = length_mask(packed_lengths, target.shape[1])
        condition = average_upsample(condition, token_mask.sum(-1), packed_lengths, target.shape[1])
        if self.training:
            dropped = (
                torch.rand(target.shape[0], device=target.device) < self.config.condition_dropout
            )
            condition = condition.masked_fill(dropped[:, None, None], 0)
        noise = torch.randn_like(target).masked_fill(~element_mask, 0)
        times = torch.rand(target.shape[0], device=target.device)
        x = (1 - times[:, None, None]) * noise + times[:, None, None] * target
        predicted = self.velocity(x, times, condition, mask)
        errors = (predicted.float() - (target - noise).float()).square() * element_mask
        # Per-example reduction keeps the DDP objective independent of rank-local durations.
        flow = (errors.sum((1, 2)) / element_mask.sum((1, 2)).clamp_min(1)).mean()
        expected_ratio = (mel_lengths.float() / token_mask.sum(-1)).log()
        duration = F.mse_loss(log_ratio.float(), expected_ratio)
        return {
            "loss": flow + self.config.duration_weight * duration,
            "flow_loss": flow.detach(),
            "duration_loss": duration.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        tokens,
        token_mask,
        speaker_ids,
        language_ids,
        input_ids=None,
        attention_mask=None,
        mel_lengths=None,
        steps=32,
        guidance=1.0,
        max_frames=2812,
        generator=None,
    ):
        if self.training:
            raise ValueError("call eval() before TTS sampling")
        if steps < 1 or guidance < 0 or max_frames < 1:
            raise ValueError("invalid sampling settings")
        if guidance != 1 and self.config.condition_dropout == 0:
            raise ValueError("guidance requires a model trained with condition dropout")
        condition, log_ratio = self.encode_condition(
            tokens, token_mask, speaker_ids, language_ids, input_ids, attention_mask
        )
        if mel_lengths is None:
            estimate = log_ratio.float().clamp(-5, 10).exp() * token_mask.sum(-1)
            mel_lengths = estimate.round().long().clamp_min(1)
        if torch.any(mel_lengths < 1) or torch.any(mel_lengths > max_frames):
            raise ValueError(
                "predicted/requested TTS duration is outside the configured limit; segment text"
            )
        pack, bins = self.config.mel_pack, self.config.mel_bins
        lengths = (mel_lengths + pack - 1) // pack
        mask = length_mask(lengths)
        condition = average_upsample(condition, token_mask.sum(-1), lengths, mask.shape[1])
        frame_mask = length_mask(mel_lengths, mask.shape[1] * pack)
        element_mask = (
            frame_mask[..., None].expand(-1, -1, bins).reshape(tokens.shape[0], -1, pack * bins)
        )
        x = torch.randn(element_mask.shape, device=tokens.device, generator=generator).masked_fill(
            ~element_mask, 0
        )
        for step in range(steps):
            times = torch.full((tokens.shape[0],), step / steps, device=tokens.device)
            velocity = self.velocity(x, times, condition, mask)
            if guidance != 1:
                unconditional = self.velocity(x, times, torch.zeros_like(condition), mask)
                velocity = unconditional + guidance * (velocity - unconditional)
            x = (x + velocity / steps).masked_fill(~element_mask, 0)
        mel = x.reshape(tokens.shape[0], -1, bins) * self.config.mel_std + self.config.mel_mean
        return mel, mel_lengths
