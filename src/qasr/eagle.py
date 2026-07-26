"""EAGLE-2 speculative decoding for QASR.

Implements a lightweight autoregressive draft head that predicts next-token
logits from the Qwen3 decoder's hidden states. During inference the head
drafts K candidate tokens which the full model verifies in a single parallel
forward pass, yielding 2-3x decode speedup on ASR transcripts.

Architecture (EAGLE lookahead head):
    concat(hidden_state h_t, embedding(token x_{t+1}))  (B, T, 2*2048)
        -> FC(4096, 2048) + SiLU + LayerNorm
        -> residual add with h_t   -> predicted next feature h_{t+1}
        -> FC(2048, vocab_size)    -> logits for token x_{t+2}
    Conditioning on the token that was actually chosen is what gives the head
    genuine lookahead; without it the head can only relearn the target's own
    lm_head (which yields zero speculative acceptance).

Training:
    Freeze the entire QASR model. Train only the fusion layer (fc1 + norm)
    with KL-divergence against the target's distribution one position ahead:
    head(h_t, embed(x_{t+1})) is matched to target_logits[t+1].

Inference:
    1. Run QASR prefill to get the KV cache, the first target token, and the
       true hidden state that produced it.
    2. EAGLE head drafts K tokens: the first draft rides on the target's true
       hidden state; each further draft chains the head's own predicted
       feature plus the drafted token's embedding.
    3. QASR verifies the K drafts in one forward pass (with output_hidden_states
       so the next round starts from a true hidden state again).
    4. Accept the longest matching prefix, emit the target's own next token
       for free, and crop the rejected positions out of the KV cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .modeling import QASRForConditionalGeneration


LOGGER = logging.getLogger("qasr.eagle")


@dataclass(frozen=True)
class EagleConfig:
    """Configuration for the EAGLE-2 draft head."""

    hidden_size: int = 2048
    vocab_size: int = 151936
    num_draft_tokens: int = 5
    temperature: float = 0.0
    top_k: int = 0
    acceptance_threshold: float = 0.0
    max_tree_width: int = 1


@dataclass
class EagleGenerationStats:
    """Telemetry collected during one speculative ``generate`` call."""

    rounds: int = 0
    drafted_tokens: int = 0
    accepted_tokens: int = 0
    emitted_tokens: int = 0
    target_forwards: int = 0
    accepted_by_position: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        """Fraction of drafted tokens the target model accepted."""
        return self.accepted_tokens / self.drafted_tokens if self.drafted_tokens else 0.0

    @property
    def tokens_per_forward(self) -> float:
        """Mean tokens emitted per target forward pass (1.0 = no speedup)."""
        return self.emitted_tokens / self.target_forwards if self.target_forwards else 0.0


class EagleHead(nn.Module):
    """EAGLE lookahead head for speculative token drafting.

    Consumes the target's hidden state ``h_t`` together with the embedding of
    the token chosen at ``t+1`` and predicts the feature ``h_{t+1}`` (hence the
    distribution over token ``x_{t+2}``). Conditioning on the next token is what
    gives the head genuine lookahead; without it the head can only relearn the
    target's own lm_head. Only fc1 + norm are trained (~8.4M params for
    hidden=2048); the lm_head is copied from the target and frozen.
    """

    def __init__(self, config: EagleConfig) -> None:
        super().__init__()
        self.config = config
        # fc1 takes concat(hidden_state, next-token embedding) -> 2 * hidden.
        self.fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.act = nn.SiLU()
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def features(self, hidden_states: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """Predict the next-position feature h_{t+1} from (h_t, embed(x_{t+1})).

        Args:
            hidden_states: (..., hidden_size) target hidden state ``h_t``.
            token_embeds: (..., hidden_size) embedding of the next token ``x_{t+1}``.

        Returns:
            Predicted feature (..., hidden_size); feed it to ``lm_head`` for
            logits or back into ``features`` to chain the next draft.
        """
        combined = torch.cat([hidden_states, token_embeds], dim=-1)
        x = self.fc1(combined)
        x = self.act(x)
        x = self.norm(x)
        return x + hidden_states

    def forward(self, hidden_states: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """Predict logits for token x_{t+2} from (h_t, embed(x_{t+1})).

        Args:
            hidden_states: (..., hidden_size) target hidden state ``h_t``.
            token_embeds: (..., hidden_size) embedding of the next token ``x_{t+1}``.

        Returns:
            Logits of shape (..., vocab_size).
        """
        return self.lm_head(self.features(hidden_states, token_embeds))

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.state_dict(), "config": self.config.__dict__},
            path / "eagle_head.pt",
        )
        LOGGER.info("Saved EAGLE head (%d params) to %s", self.num_parameters, path)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cpu") -> "EagleHead":
        path = Path(path)
        checkpoint = torch.load(path / "eagle_head.pt", map_location=device, weights_only=True)
        config = EagleConfig(**checkpoint["config"])
        head = cls(config)
        head.load_state_dict(checkpoint["state_dict"])
        LOGGER.info(
            "Loaded EAGLE head (%d params) from %s", head.num_parameters, path
        )
        return head


class EagleSpeculativeDecoder:
    """Wraps a QASR model + EAGLE head for accelerated generation.

    Implements the draft-verify loop:
    1. Target model produces hidden states for the current sequence.
    2. EAGLE head drafts K tokens autoregressively.
    3. Target model verifies all K positions in one forward pass.
    4. Accept the longest matching prefix; discard the rest.
    """

    def __init__(
        self,
        model: QASRForConditionalGeneration,
        eagle_head: EagleHead,
        config: EagleConfig | None = None,
    ) -> None:
        self.model = model
        self.eagle_head = eagle_head
        self.config = config or eagle_head.config
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.eagle_head.to(device=self.device, dtype=self.dtype)
        self.eagle_head.eval()
        self.last_stats = EagleGenerationStats()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        eagle_path: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "auto",
    ) -> "EagleSpeculativeDecoder":
        from transformers import AutoConfig

        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "attn_implementation": "sdpa",
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        model = QASRForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
        model.eval()
        eagle_head = EagleHead.from_pretrained(eagle_path)
        return cls(model=model, eagle_head=eagle_head)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 256,
        num_draft_tokens: int | None = None,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Generate with speculative decoding (greedy decoding is lossless).

        With ``temperature <= 0`` every emitted token is either verified equal
        to the target's argmax or is the target's argmax itself, so the output
        matches plain greedy generation token for token.

        Args:
            input_ids: Prompt token IDs (1, prompt_len). Batch size must be 1.
            input_features: Mel features (1, frames, 128).
            input_features_mask: Feature mask (1, frames).
            attention_mask: Attention mask (1, total_len).
            max_new_tokens: Maximum tokens to generate.
            num_draft_tokens: Override draft length (default from config).
            temperature: Sampling temperature (0 = greedy).

        Returns:
            Full output token IDs (1, prompt_len + generated_len). Telemetry
            for the call is stored in ``self.last_stats``.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("EagleSpeculativeDecoder.generate only supports batch size 1")
        k = num_draft_tokens or self.config.num_draft_tokens
        temp = temperature if temperature is not None else self.config.temperature
        eos_token_id = self.model.config.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]
        embed_tokens = self.model.get_input_embeddings()
        stats = EagleGenerationStats(accepted_by_position=[0] * k)
        self.last_stats = stats

        # Prefill: build the KV cache, take the first token from the target, and
        # keep the true hidden state that produced it for the first draft.
        outputs = self.model(
            input_ids=input_ids,
            input_features=input_features,
            input_features_mask=input_features_mask,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = outputs.past_key_values
        cache_length = past_key_values.get_seq_length()
        source_hidden = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, H) h_t
        next_token = self._select_token(outputs.logits[:, -1, :], temp)  # (1, 1) x_{t+1}
        stats.target_forwards += 1
        stats.emitted_tokens += 1

        generated: list[torch.Tensor] = [next_token]
        finished = next_token.item() == eos_token_id

        while len(generated) < max_new_tokens and not finished:
            # --- Draft phase: the first draft rides on the target's true hidden
            # state; each further draft chains the head's predicted feature. ---
            draft_tokens: list[torch.Tensor] = []
            draft_hidden = source_hidden
            cur_token = next_token
            for _ in range(k):
                token_embed = embed_tokens(cur_token)
                draft_hidden = self.eagle_head.features(draft_hidden, token_embed)
                draft_logits = self.eagle_head.lm_head(draft_hidden)[:, -1, :]
                cur_token = self._select_token(draft_logits, temp)
                draft_tokens.append(cur_token)
            draft_ids = torch.cat(draft_tokens, dim=-1)  # (1, K)

            # --- Verify phase: one target forward over [next_token, drafts],
            # with hidden states so the next round can start from a true one. ---
            verify_input = torch.cat([next_token, draft_ids], dim=-1)  # (1, K+1)
            verify_outputs = self.model(
                input_ids=verify_input,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = verify_outputs.past_key_values
            verify_hidden = verify_outputs.hidden_states[-1]  # (1, K+1, H)
            stats.target_forwards += 1
            # target_tokens[i] is the target's choice after consuming
            # verify_input[:, i]; position i therefore judges draft i.
            target_tokens = self._select_token(verify_outputs.logits[0], temp)  # (K+1, 1)

            accepted = 0
            while accepted < k and target_tokens[accepted, 0] == draft_ids[0, accepted]:
                stats.accepted_by_position[accepted] += 1
                accepted += 1

            # Emit accepted drafts plus the target's own next token for free
            # (the correction on rejection, the bonus token on full acceptance).
            round_tokens = [draft_ids[:, i : i + 1] for i in range(accepted)]
            round_tokens.append(target_tokens[accepted].view(1, 1))
            stats.rounds += 1
            stats.drafted_tokens += k
            stats.accepted_tokens += accepted
            for token in round_tokens:
                generated.append(token)
                stats.emitted_tokens += 1
                if token.item() == eos_token_id:
                    finished = True
                    break

            # Drop the rejected draft positions from the KV cache; only
            # next_token and the accepted drafts were truly consumed. The next
            # round starts from the true hidden state that produced the bonus
            # token (verify_hidden at the last consumed position).
            cache_length += accepted + 1
            past_key_values.crop(cache_length)
            next_token = generated[-1]
            source_hidden = verify_hidden[:, accepted : accepted + 1, :]

        generated = generated[:max_new_tokens]
        return torch.cat([input_ids] + generated, dim=-1)

    @staticmethod
    def _select_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Pick one token per row of (rows, vocab) logits, returned as (rows, 1)."""
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        probs = F.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)


def compute_eagle_loss(
    eagle_head: EagleHead,
    hidden_states: torch.Tensor,
    token_embeds: torch.Tensor,
    target_logits: torch.Tensor,
    label_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """KL-divergence training loss for the EAGLE lookahead head.

    Aligns one position ahead: from ``hidden[t]`` and the embedding of the
    actually-chosen token ``x_{t+1}`` the head must predict the target's
    distribution at ``t+1`` (the distribution over ``x_{t+2}``). This lookahead
    objective is what makes the draft head useful; matching the target at the
    same position would only relearn its lm_head.

    Args:
        eagle_head: The draft head module.
        hidden_states: (B, S, H) last-layer hidden states from the frozen target.
        token_embeds: (B, S, H) input embeddings of ``input_ids``.
        target_logits: (B, S, V) logits from the frozen target model.
        label_mask: (B, S) boolean mask; True where ``input_ids[p]`` is a
            transcript token. Position ``t`` is trained when ``label_mask[t+1]``
            is True (the conditioned-on token is a real transcript token).
        temperature: Softmax temperature for softening the target distribution.

    Returns:
        Scalar KL-divergence loss.
    """
    # Shift by one: head(h_t, embed(x_{t+1})) -> target_logits[t+1].
    src_hidden = hidden_states[:, :-1, :]
    nxt_embed = token_embeds[:, 1:, :]
    tgt_logits = target_logits[:, 1:, :]

    if label_mask is not None:
        valid = label_mask[:, 1:]
        src_hidden = src_hidden[valid]
        nxt_embed = nxt_embed[valid]
        tgt_logits = tgt_logits[valid]
    else:
        src_hidden = src_hidden.reshape(-1, src_hidden.shape[-1])
        nxt_embed = nxt_embed.reshape(-1, nxt_embed.shape[-1])
        tgt_logits = tgt_logits.reshape(-1, tgt_logits.shape[-1])

    # Project only the retained positions so the (N, V) draft logits stay small
    # even at large per-device batch sizes.
    draft_logits = eagle_head(src_hidden, nxt_embed)

    if draft_logits.numel() == 0:
        # Connected zero so every DDP rank still produces head gradients and the
        # gradient all-reduce does not stall.
        return draft_logits.sum() * 0.0

    # KL(target || draft) with log-space target to avoid a large exp() tensor.
    target_log_probs = F.log_softmax(tgt_logits / temperature, dim=-1)
    draft_log_probs = F.log_softmax(draft_logits / temperature, dim=-1)
    kl = F.kl_div(draft_log_probs, target_log_probs, log_target=True, reduction="batchmean")
    # Scale by T^2 as per the knowledge-distillation convention.
    return kl * (temperature ** 2)


__all__ = [
    "EagleConfig",
    "EagleGenerationStats",
    "EagleHead",
    "EagleSpeculativeDecoder",
    "compute_eagle_loss",
]
