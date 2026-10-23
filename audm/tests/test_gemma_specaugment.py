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


def test_fill_preserves_unselected_bins_padding_and_clean_rows():
    x = torch.randn(2, 32, 8)
    x[1, 0] = 0  # legitimate zeros must not be mistaken for selected bins
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[0, 15:] = False
    result = gemma_specaugment(x, mask, PrefixMask())
    torch.testing.assert_close(result[0, :15], x[0, :15].mean(0).expand_as(result[0, :15]))
    torch.testing.assert_close(result[0, 15:], x[0, 15:])
    torch.testing.assert_close(result[1], x[1])
    torch.testing.assert_close(
        gemma_specaugment(x, torch.zeros_like(mask), PrefixMask()), x
    )


def test_fill_uses_original_per_frequency_mean_for_both_masks():
    class BothMasks:
        def augment_features(self, features, mask):
            out = features.clone()
            out[:, :3] = 0
            out[:, :, 2:5] = 0
            return out

    x = torch.arange(1, 193, dtype=torch.float32).reshape(1, 24, 8)
    mask = torch.ones(1, 24, dtype=torch.bool)
    mask[:, 20:] = False
    result = gemma_specaugment(x, mask, BothMasks())
    mean = x[:, :20].mean(1, keepdim=True)
    torch.testing.assert_close(result[:, :3], mean.expand(1, 3, 8))
    torch.testing.assert_close(result[:, :20, 2:5], mean[:, :, 2:5].expand(1, 20, 3))
    torch.testing.assert_close(result[:, 20:], x[:, 20:])


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
    safe = gemma_specaugment(x, mask, PrefixMask())
    output = model(input_features=safe, attention_mask=mask)
    output.last_hidden_state.square().mean().backward()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
