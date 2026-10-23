from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import Gemma4AudioConfig, Gemma4AudioModel

from audm.collator import gemma_specaugment
from audm.modeling import name_aud_layers


class PrefixMask:
    def augment_features(self, features, mask):
        result = features.clone()
        result[0, :20] = 0
        return result


def frontend():
    return SimpleNamespace(mel_floor=1e-3, per_bin_mean=None, per_bin_stddev=None)


def test_fill_preserves_unselected_bins_padding_and_clean_rows():
    x = torch.randn(2, 32, 8)
    x[1, 0] = 0  # legitimate zeros must not be mistaken for selected bins
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[0, 15:] = False
    result = gemma_specaugment(x, mask, PrefixMask(), frontend())
    torch.testing.assert_close(result[0, :15], torch.full_like(result[0, :15], np.log(1e-3)))
    torch.testing.assert_close(result[0, 15:], x[0, 15:])
    torch.testing.assert_close(result[1], x[1])
    torch.testing.assert_close(
        gemma_specaugment(x, torch.zeros_like(mask), PrefixMask(), frontend()), x
    )


def test_fill_uses_frontend_normalization():
    x = torch.ones(1, 24, 8)
    f = frontend()
    f.per_bin_mean = np.arange(8).reshape(1, 1, 8)
    f.per_bin_stddev = np.ones((1, 1, 8)) * 2
    result = gemma_specaugment(x, torch.ones(1, 24, dtype=torch.bool), PrefixMask(), f)
    expected = torch.tensor((np.log(1e-3) - f.per_bin_mean) / 2, dtype=x.dtype)
    torch.testing.assert_close(result[:, :20], expected.expand(1, 20, 8))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_zero_prefix_regression(dtype, checkpointed):
    torch.manual_seed(42)
    model = Gemma4AudioModel(Gemma4AudioConfig(
        hidden_size=32, num_attention_heads=4, output_proj_dims=32,
        num_hidden_layers=12, subsampling_conv_channels=[128, 4],
        attention_chunk_size=12, attention_context_left=13, use_clipped_linears=True,
    )).to(dtype).train()
    name_aud_layers(model)
    if checkpointed:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )
    x = torch.randn(2, 128, 128, dtype=dtype) - 2
    mask = torch.ones(2, 128, dtype=torch.bool)
    unsafe = PrefixMask().augment_features(x, mask)
    output = model(input_features=unsafe, attention_mask=mask)
    assert torch.isfinite(output.last_hidden_state).all()
    output.last_hidden_state.square().mean().backward()
    assert any(p.grad is not None and not p.grad.isfinite().all() for p in model.parameters())

    model.zero_grad(set_to_none=True)
    safe = gemma_specaugment(x, mask, PrefixMask(), frontend())
    output = model(input_features=safe, attention_mask=mask)
    output.last_hidden_state.square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
