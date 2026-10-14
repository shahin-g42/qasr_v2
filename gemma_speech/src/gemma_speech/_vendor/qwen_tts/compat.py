"""Local Transformers 5 compatibility; no global monkey patches.

The selected Qwen checkpoint and codec use default RoPE. Preserve the upstream
4.57 formula explicitly: Transformers 5 removed 'default' from its RoPE registry.
"""
import torch


def default_rope(config, device=None, seq_len=None):
    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    dim = int(dim * getattr(config, "partial_rotary_factor", 1.0))
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS = {"default": default_rope}


def create_causal_mask(*, config, input_embeds, attention_mask, past_key_values,
                       cache_position=None, position_ids=None, **kwargs):
    from transformers.masking_utils import create_causal_mask as current
    return current(config=config, inputs_embeds=input_embeds, attention_mask=attention_mask,
                   past_key_values=past_key_values, position_ids=position_ids, **kwargs)


def create_sliding_window_causal_mask(*, config, input_embeds, attention_mask, past_key_values,
                                      cache_position=None, position_ids=None, **kwargs):
    from transformers.masking_utils import create_sliding_window_causal_mask as current
    return current(config=config, inputs_embeds=input_embeds, attention_mask=attention_mask,
                   past_key_values=past_key_values, position_ids=position_ids, **kwargs)
