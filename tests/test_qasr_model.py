import tempfile
import unittest

import torch
import torch.nn.functional as F
from transformers import AutoModelForMultimodalLM, Qwen3Config

from qasr import QASRConfig, QASRForConditionalGeneration, QASRMultiModalProjector
from qasr.configuration import CohereEncoderConfig
from qasr.utils import get_subsampled_attention_mask, get_subsampling_output_lengths


AUDIO_TOKEN_ID = 120


def tiny_config() -> QASRConfig:
    audio_config = CohereEncoderConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_mel_bins=128,
        subsampling_factor=8,
        subsampling_conv_channels=4,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        conv_kernel_size=3,
        max_position_embeddings=64,
    )
    text_config = Qwen3Config(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    return QASRConfig(
        audio_config=audio_config,
        text_config=text_config,
        audio_token_id=AUDIO_TOKEN_ID,
        timestamp_token_id=121,
        pad_token_id=0,
        eos_token_id=2,
    )


class SubsamplingTest(unittest.TestCase):
    def test_three_stride_two_convolutions_use_ceil_division(self) -> None:
        lengths = torch.tensor([1, 2, 7, 8, 9, 100, 101, 3000])

        output = get_subsampling_output_lengths(lengths)

        self.assertEqual(output.tolist(), [1, 1, 1, 1, 2, 13, 13, 375])

    def test_padded_mask_propagates_valid_lengths(self) -> None:
        mask = torch.zeros(2, 17, dtype=torch.long)
        mask[0, :17] = 1
        mask[1, :9] = 1

        output = get_subsampled_attention_mask(mask, target_length=3)

        self.assertEqual(output.tolist(), [[True, True, True], [True, True, False]])


class HybridModelTest(unittest.TestCase):
    def test_projector_has_planned_shapes(self) -> None:
        projector = QASRMultiModalProjector(tiny_config())

        self.assertEqual(tuple(projector.linear_1.weight.shape), (16, 16))
        self.assertEqual(tuple(projector.linear_2.weight.shape), (16, 16))
        self.assertEqual(tuple(projector(torch.randn(5, 16)).shape), (5, 16))

    def test_audio_features_flatten_only_valid_positions(self) -> None:
        model = QASRForConditionalGeneration(tiny_config()).eval()
        mask = torch.zeros(2, 17, dtype=torch.long)
        mask[0, :17] = 1
        mask[1, :9] = 1

        with torch.no_grad():
            output = model.get_audio_features(
                torch.randn(2, 17, 128),
                mask,
                return_dict=True,
            )

        self.assertEqual(tuple(output.attention_mask.shape), (2, 3))
        self.assertEqual(output.attention_mask.sum().item(), 5)
        self.assertEqual(tuple(output.pooler_output.shape), (5, 16))

    def test_placeholder_mismatch_raises(self) -> None:
        model = QASRForConditionalGeneration(tiny_config()).eval()
        input_ids = torch.tensor([[1, AUDIO_TOKEN_ID, 3]])

        with self.assertRaisesRegex(ValueError, "Audio features and audio tokens do not match"):
            model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                input_features=torch.randn(1, 16, 128),
                input_features_mask=torch.ones(1, 16, dtype=torch.long),
            )

    def test_forward_backward_uses_causal_loss_and_tied_embeddings(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        input_ids = torch.tensor(
            [
                [1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 5, 6],
                [1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 5, 6, 0],
            ]
        )
        attention_mask = input_ids.ne(0).long()
        feature_mask = torch.zeros(2, 17, dtype=torch.long)
        feature_mask[0, :17] = 1
        feature_mask[1, :9] = 1
        labels = torch.full_like(input_ids, -100)
        labels[0, -2:] = input_ids[0, -2:]
        labels[1, 3:5] = input_ids[1, 3:5]

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=torch.randn(2, 17, 128),
            input_features_mask=feature_mask,
            labels=labels,
        )
        output.loss.backward()

        self.assertTrue(torch.isfinite(output.loss))
        self.assertIsNotNone(model.model.multi_modal_projector.linear_1.weight.grad)
        self.assertIs(model.lm_head.weight, model.get_input_embeddings().weight)

    def test_projector_backward_with_frozen_towers_and_reentrant_checkpointing(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        model.model.audio_tower.requires_grad_(False)
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)
        model.model.multi_modal_projector.requires_grad_(True)
        model.gradient_checkpointing_enable({"use_reentrant": True})
        self.assertFalse(model.model.audio_tower.is_gradient_checkpointing)
        self.assertTrue(model.model.language_model.is_gradient_checkpointing)
        input_ids = torch.tensor([[1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 5, 6]])
        labels = torch.full_like(input_ids, -100)
        labels[0, -2:] = input_ids[0, -2:]

        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=torch.randn(1, 9, 128),
            input_features_mask=torch.ones(1, 9, dtype=torch.long),
            labels=labels,
            use_cache=False,
        )
        output.loss.backward()

        self.assertIsNotNone(model.model.multi_modal_projector.linear_1.weight.grad)
        self.assertIsNotNone(model.model.multi_modal_projector.linear_2.weight.grad)
        self.assertIsNone(model.model.audio_tower.layers[0].feed_forward1.linear1.weight.grad)

    def test_loss_predicts_the_next_label(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        labels = torch.tensor([[7, 8, 9, -100]])
        logits = torch.full((1, 4, 128), -8.0)
        logits[0, 0, 8] = 8.0
        logits[0, 1, 9] = 8.0

        loss = model.loss_function(logits=logits, labels=labels, vocab_size=128)
        same_position_loss = F.cross_entropy(logits[:, :3].reshape(-1, 128), labels[:, :3].reshape(-1))

        self.assertLess(loss.item(), 0.01)
        self.assertGreater(same_position_loss.item(), 5.0)

    def test_config_model_reload_and_cached_generation(self) -> None:
        model = QASRForConditionalGeneration(tiny_config()).eval()
        input_ids = torch.tensor([[1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 3]])
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "input_features": torch.randn(1, 16, 128),
            "input_features_mask": torch.ones(1, 16, dtype=torch.long),
        }
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = AutoModelForMultimodalLM.from_pretrained(directory).eval()
            with torch.no_grad():
                generated = reloaded.generate(
                    **kwargs,
                    max_new_tokens=2,
                    do_sample=False,
                    use_cache=True,
                )

        self.assertEqual(reloaded.config.model_type, "qasr")
        self.assertIsInstance(reloaded, QASRForConditionalGeneration)
        # transformers registers the Cohere Conformer encoder as "parakeet_encoder"
        self.assertEqual(reloaded.config.audio_config.model_type, "parakeet_encoder")
        self.assertIs(reloaded.lm_head.weight, reloaded.get_input_embeddings().weight)
        self.assertEqual(tuple(generated.shape), (1, 6))


if __name__ == "__main__":
    unittest.main()
