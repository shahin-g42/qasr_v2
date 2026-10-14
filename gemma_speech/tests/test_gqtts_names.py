"""Renaming must preserve each source class's implementation and tensor values."""

import inspect

import pytest
import torch
from gqtts.base import name_modules, public_class_name
from gqtts.speaker import SpeakerAdapter
from torch import nn
from transformers.models.gemma4 import modeling_gemma4

from gemma_speech._vendor.qwen_tts.core.models import modeling_qwen3_tts
from gemma_speech._vendor.qwen_tts.core.tokenizer_12hz import modeling_qwen3_tts_tokenizer_v2


@pytest.mark.parametrize("reverse", [False, True])
def test_rmsnorm_renaming_preserves_both_implementations(reverse):
    originals = [modeling_gemma4.Gemma4RMSNorm, modeling_qwen3_tts.Qwen3TTSRMSNorm]
    if reverse:
        originals.reverse()
    modules = nn.ModuleList([cls(8, eps=1e-5) for cls in originals])
    x = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
    expected = [m(x).detach().clone() for m in modules]
    weights = {k: v.clone() for k, v in modules.state_dict().items()}
    name_modules(modules)
    assert len({type(m) for m in modules}) == 2
    for module, original, output in zip(modules, originals, expected, strict=True):
        assert type(module).__bases__ == (original,)
        assert isinstance(module, original)
        torch.testing.assert_close(module(x), output, rtol=0, atol=0)
    for key, tensor in weights.items():
        torch.testing.assert_close(modules.state_dict()[key], tensor, rtol=0, atol=0)
    # Repeated naming is a no-op.
    classes = [type(m) for m in modules]
    name_modules(modules)
    assert [type(m) for m in modules] == classes


def test_source_class_names_are_unique_across_all_components():
    aliases = {}
    classes = {SpeakerAdapter}
    for module in (modeling_gemma4, modeling_qwen3_tts, modeling_qwen3_tts_tokenizer_v2):
        classes.update(
            cls
            for _, cls in inspect.getmembers(module, inspect.isclass)
            if issubclass(cls, nn.Module)
        )
    for original in classes:
        name = public_class_name(original)
        if name is not None:
            assert name not in aliases or aliases[name] is original, (
                name,
                original,
                aliases.get(name),
            )
            aliases[name] = original
    assert len(aliases) > 20
