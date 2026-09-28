import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from qasr.convert_weights import (
    _convert_encoder_state,
    _load_exact,
    _load_prefixed_state,
    _resolve_safetensors,
)


class ExactTransferTest(unittest.TestCase):
    def test_encoder_keys_are_converted_to_cohere_encoder_names(self) -> None:
        tensor = torch.ones(1)
        converted = _convert_encoder_state(
            {
                "pre_encode.conv.0.weight": tensor,
                "pre_encode.out.bias": tensor,
                "layers.3.self_attn.linear_q.weight": tensor,
                "layers.3.conv.batch_norm.running_mean": tensor,
                "layers.3.feed_forward1.linear1.weight": tensor,
            }
        )

        self.assertEqual(
            set(converted),
            {
                "subsampling.layers.0.weight",
                "subsampling.linear.bias",
                "layers.3.self_attn.q_proj.weight",
                "layers.3.conv.norm.running_mean",
                "layers.3.feed_forward1.linear1.weight",
            },
        )
        self.assertTrue(all(value is tensor for value in converted.values()))

    def test_encoder_key_collisions_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mapped multiple tensors"):
            _convert_encoder_state(
                {
                    "layers.0.self_attn.linear_q.weight": torch.ones(1),
                    "layers.0.self_attn.q_proj.weight": torch.zeros(1),
                }
            )

    def test_selective_safetensors_loading_strips_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            weights_path = Path(directory) / "model.safetensors"
            save_file(
                {
                    "model.encoder.weight": torch.arange(4),
                    "model.decoder.weight": torch.arange(8),
                },
                weights_path,
            )

            state = _load_prefixed_state(_resolve_safetensors(directory), "model.encoder.")

        self.assertEqual(list(state), ["weight"])
        self.assertTrue(torch.equal(state["weight"], torch.arange(4)))

    def test_every_tensor_is_equal_after_transfer(self) -> None:
        source = torch.nn.Linear(4, 3)
        target = torch.nn.Linear(4, 3)

        _load_exact(target, source.state_dict(), "linear")

        for key, tensor in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], tensor))

    def test_missing_or_wrong_shape_tensor_fails(self) -> None:
        target = torch.nn.Linear(4, 3)

        with self.assertRaises(RuntimeError):
            _load_exact(target, {"weight": torch.zeros(2, 4)}, "linear")


class EncoderFeatureSizeTest(unittest.TestCase):
    """Regression: conversion crashed with AttributeError because
    ParakeetEncoderConfig defines no ``feature_size`` — the attribute only
    survives when the source repo's JSON carries it as an extra key. The
    helper must fall back to the always-present ``num_mel_bins``."""

    def test_falls_back_to_num_mel_bins(self) -> None:
        from transformers import ParakeetEncoderConfig

        from qasr.convert_weights import _encoder_feature_size

        bare = ParakeetEncoderConfig(num_mel_bins=128)
        self.assertFalse(hasattr(bare, "feature_size"))
        self.assertEqual(_encoder_feature_size(bare), 128)

    def test_explicit_feature_size_wins(self) -> None:
        from transformers import ParakeetEncoderConfig

        from qasr.convert_weights import _encoder_feature_size

        with_key = ParakeetEncoderConfig(num_mel_bins=128, feature_size=128)
        self.assertEqual(_encoder_feature_size(with_key), 128)


if __name__ == "__main__":
    unittest.main()
