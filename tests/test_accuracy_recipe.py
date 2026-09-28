"""Tests for the Phase A accuracy recipe (loss, projector, groups, data)."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config, TrainingArguments
from transformers.loss.loss_utils import ForCausalLMLoss

from qasr.config import TrainConfig
from qasr.configuration import CohereEncoderConfig, QASRConfig
from qasr.data import JsonlSpeechDataset, parse_record
from qasr.losses import make_smoothed_causal_lm_loss
from qasr.modeling import QASRForConditionalGeneration, QASRMultiModalProjector
from qasr.train import PredictionLoggingTrainer
from qasr.utils import get_projector_pool_output_lengths

AUDIO_TOKEN_ID = 120


def tiny_config(**overrides) -> QASRConfig:
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
        **overrides,
    )


class SmoothedLossTest(unittest.TestCase):
    def _fixtures(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 7, 64)
        labels = torch.randint(0, 64, (2, 7))
        labels[0, :3] = -100  # prompt masking
        labels[1, 5:] = -100
        return logits, labels

    def test_epsilon_zero_matches_stock_loss_bitwise(self) -> None:
        logits, labels = self._fixtures()
        custom = make_smoothed_causal_lm_loss(0.0)

        for num_items in (None, torch.tensor(6)):
            expected = ForCausalLMLoss(logits, labels, 64, num_items_in_batch=num_items)
            actual = custom(logits, labels, 64, num_items_in_batch=num_items)
            self.assertTrue(torch.equal(expected, actual))

    def test_epsilon_matches_hand_computed_smoothed_ce(self) -> None:
        logits, labels = self._fixtures()
        custom = make_smoothed_causal_lm_loss(0.1)

        actual = custom(logits.clone(), labels.clone(), 64)

        # Mirror the stock shift/pad, then hand-compute smoothed mean CE.
        padded = F.pad(labels, (0, 1), value=-100)
        shifted = padded[..., 1:].reshape(-1)
        expected = F.cross_entropy(
            logits.float().reshape(-1, 64),
            shifted,
            ignore_index=-100,
            reduction="mean",
            label_smoothing=0.1,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_sum_reduction_divides_by_num_items_in_batch(self) -> None:
        logits, labels = self._fixtures()
        custom = make_smoothed_causal_lm_loss(0.1)
        num_items = torch.tensor(5)

        actual = custom(logits, labels, 64, num_items_in_batch=num_items)

        padded = F.pad(labels, (0, 1), value=-100)
        shifted = padded[..., 1:].reshape(-1)
        expected = (
            F.cross_entropy(
                logits.float().reshape(-1, 64),
                shifted,
                ignore_index=-100,
                reduction="sum",
                label_smoothing=0.1,
            )
            / 5
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_fully_masked_labels_yield_zero_not_nan(self) -> None:
        logits = torch.randn(1, 4, 64)
        labels = torch.full((1, 4), -100, dtype=torch.long)
        custom = make_smoothed_causal_lm_loss(0.1)

        loss = custom(logits, labels, 64)

        self.assertEqual(loss.item(), 0.0)

    def test_invalid_epsilon_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_smoothed_causal_lm_loss(-0.1)
        with self.assertRaises(ValueError):
            make_smoothed_causal_lm_loss(0.2)


class DeepProjectorTest(unittest.TestCase):
    def test_depth_one_reproduces_legacy_module_exactly(self) -> None:
        config = tiny_config()
        projector = QASRMultiModalProjector(config)

        legacy = torch.nn.Sequential(projector.linear_1, projector.act, projector.linear_2)
        inputs = torch.randn(5, 16)

        self.assertIsNone(projector.input_layernorm)
        self.assertIsNone(projector.hidden_layers)
        self.assertTrue(torch.equal(projector(inputs), legacy(inputs)))

    def test_depth_two_adds_layernorm_and_residual_blocks(self) -> None:
        config = tiny_config(projector_depth=2)
        projector = QASRMultiModalProjector(config)

        self.assertIsNotNone(projector.input_layernorm)
        self.assertEqual(len(projector.hidden_layers), 1)
        output = projector(torch.randn(5, 16))
        self.assertEqual(tuple(output.shape), (5, 16))

    def test_legacy_state_dict_loads_into_deep_projector(self) -> None:
        legacy = QASRMultiModalProjector(tiny_config())
        deep = QASRMultiModalProjector(tiny_config(projector_depth=3))

        missing, unexpected = deep.load_state_dict(legacy.state_dict(), strict=False)

        self.assertEqual(unexpected, [])
        # Fresh-init blocks are exactly the missing keys.
        self.assertTrue(any("hidden_layers" in key for key in missing))
        self.assertTrue(any("input_layernorm" in key for key in missing))
        # Shared weights carry over unchanged.
        self.assertTrue(torch.equal(deep.linear_1.weight, legacy.linear_1.weight))
        self.assertTrue(torch.equal(deep.linear_2.weight, legacy.linear_2.weight))


class PoolStrideLengthsTest(unittest.TestCase):
    def test_stride_two_halves_lengths_with_ceil(self) -> None:
        lengths = torch.tensor([1, 2, 3, 4, 7, 100])
        output = get_projector_pool_output_lengths(lengths, pool_stride=2)
        self.assertEqual(output.tolist(), [1, 1, 2, 2, 4, 50])

    def test_stride_one_is_identity(self) -> None:
        lengths = torch.tensor([1, 5, 375])
        output = get_projector_pool_output_lengths(lengths, pool_stride=1)
        self.assertEqual(output.tolist(), [1, 5, 375])


class CTCLossTest(unittest.TestCase):
    def _batch(self) -> dict:
        input_ids = torch.tensor(
            [[1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 5, 6]]
        )
        labels = torch.full_like(input_ids, -100)
        labels[0, -2:] = input_ids[0, -2:]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "input_features": torch.randn(1, 17, 128),
            "input_features_mask": torch.ones(1, 17, dtype=torch.long),
            "labels": labels,
        }

    def test_ctc_adds_finite_term_and_head_receives_gradient(self) -> None:
        torch.manual_seed(3)
        model = QASRForConditionalGeneration(tiny_config())
        head = model.add_ctc_head()
        model.train()
        batch = self._batch()

        model.ctc_loss_weight = 0.0
        loss_without = model(**batch).loss.detach().clone()
        model.ctc_loss_weight = 0.3
        output = model(**batch)

        self.assertTrue(torch.isfinite(output.loss))
        self.assertFalse(torch.equal(output.loss, loss_without))
        output.loss.backward()
        self.assertIsNotNone(head.weight.grad)

    def test_ctc_works_under_bf16_mixed_precision(self) -> None:
        """Regression: the head used to be cast to bf16 while being fed
        .float() encoder states — a hard matmul dtype error on the very first
        training step of any bf16 run with ctc_loss_weight > 0."""
        torch.manual_seed(3)
        model = QASRForConditionalGeneration(tiny_config()).to(torch.bfloat16)
        head = model.add_ctc_head()
        head.to(dtype=torch.bfloat16)  # mirrors train.py's compute-dtype cast
        model.ctc_loss_weight = 0.3
        model.train()
        batch = self._batch()
        batch["input_features"] = batch["input_features"].to(torch.bfloat16)

        output = model(**batch)

        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(head.weight.grad)

    def test_no_labels_skips_ctc(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        model.add_ctc_head()
        model.ctc_loss_weight = 0.3
        model.train()
        batch = self._batch()
        batch.pop("labels")

        output = model(**batch)

        self.assertIsNone(output.loss)

    def test_empty_target_rows_are_skipped_from_ctc(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        model.add_ctc_head()
        # Production wiring: train.py swaps in the smoothed loss, whose
        # fully-masked-batch guard turns the 0/0 mean into a zero loss.
        model.loss_function = make_smoothed_causal_lm_loss(0.1)
        model.ctc_loss_weight = 0.3
        model.train()
        batch = self._batch()
        # Fully masked labels = non-speech sample => zero-length CTC target.
        batch["labels"] = torch.full_like(batch["input_ids"], -100)

        output = model(**batch)

        # The AR loss teaches "emit nothing"; CTC skips the row entirely.
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))


class OptimizerGroupsTest(unittest.TestCase):
    def _trainer(self, tmp_dir: str, model) -> PredictionLoggingTrainer:
        args = TrainingArguments(
            output_dir=tmp_dir,
            report_to=[],
            learning_rate=4.0e-5,
            weight_decay=0.01,
            optim="adamw_torch",
            per_device_train_batch_size=1,
        )

        class DummyProcessor:
            class feature_extractor:
                sampling_rate = 16000

        # Minimal trainer without datasets; create_optimizer only needs model+args.
        return PredictionLoggingTrainer(
            model=model,
            args=args,
            train_dataset=None,
            data_collator=lambda x: x,
            prediction_processor=DummyProcessor(),
            prediction_collator=None,
            eval_log_samples=0,
            eval_generation_max_new_tokens=1,
            llm_lr_factor=0.5,
            embed_lr_factor=0.25,
        )

    def test_three_lr_groups_and_no_duplicate_params(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self._trainer(tmp_dir, model)
            optimizer = trainer.create_optimizer()

        lrs = sorted(round(group["lr"], 10) for group in optimizer.param_groups)
        # Groups with zero matching parameters are omitted; the tiny model has
        # no embed bias/norm params, so only five groups exist here. The LLM
        # and encoder sides always split into decay + no-decay pairs.
        self.assertEqual(lrs, sorted([4.0e-5, 4.0e-5, 2.0e-5, 2.0e-5, 1.0e-5]))
        self.assertEqual(
            sorted(group["lr"] for group in optimizer.param_groups if group["lr"] != 1.0e-5),
            sorted([4.0e-5, 4.0e-5, 2.0e-5, 2.0e-5]),
        )

        # Every trainable parameter appears exactly once; the tied lm_head
        # weight shows up under embed_tokens, never as its own group entry.
        seen = []
        for group in optimizer.param_groups:
            seen.extend(param.data_ptr() for param in group["params"])
        self.assertEqual(len(seen), len(set(seen)))

        trainable_ptrs = {
            param.data_ptr() for param in model.parameters() if param.requires_grad
        }
        self.assertEqual(set(seen), trainable_ptrs)
        embed_ptr = model.model.language_model.get_input_embeddings().weight.data_ptr()
        self.assertIn(embed_ptr, seen)

    def test_weight_decay_split_per_group(self) -> None:
        model = QASRForConditionalGeneration(tiny_config())
        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = self._trainer(tmp_dir, model)
            optimizer = trainer.create_optimizer()

        decay_values = {group["weight_decay"] for group in optimizer.param_groups}
        self.assertEqual(decay_values, {0.0, 0.01})


class EmptyTargetDataTest(unittest.TestCase):
    def test_parse_record_empty_target_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "silence.jsonl"
            parsed = parse_record(
                {"wav_path": "noise.wav", "text": "", "duration": 2.0},
                manifest_path=manifest,
                line_number=1,
                audio_root=None,
                language="ar",
                empty_target_ok=True,
            )
            self.assertEqual(parsed["text"], "")

            with self.assertRaises(Exception):
                parse_record(
                    {"wav_path": "noise.wav", "text": "", "duration": 2.0},
                    manifest_path=manifest,
                    line_number=1,
                    audio_root=None,
                    language="ar",
                    empty_target_ok=False,
                )

    def test_dataset_keeps_blank_records_with_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest = Path(tmp_dir) / "silence.jsonl"
            lines = [
                {"wav_path": "a.wav", "text": "", "duration": 1.0},
                {"wav_path": "b.wav", "text": "   ", "duration": 1.5},
                {"wav_path": "c.wav", "text": "مرحبا", "duration": 2.0},
            ]
            manifest.write_text(
                "\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8"
            )

            dataset = JsonlSpeechDataset(
                manifest,
                min_duration_seconds=0.1,
                max_duration_seconds=30.0,
                empty_target_ok=True,
            )

            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0]["text"], "")
            self.assertEqual(dataset[1]["text"], "")
            self.assertEqual(dataset[2]["text"], "مرحبا")

            strict = JsonlSpeechDataset(
                manifest, min_duration_seconds=0.1, max_duration_seconds=30.0
            )
            self.assertEqual(len(strict), 1)


class TrainConfigKeysTest(unittest.TestCase):
    def test_new_phase_a_keys_parse_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text(
                """
train_manifest:
  ar: [manifests/train_ar.json]
balanced_sampling: true
sampling_temperature: 0.5
llm_lr_factor: 0.5
embed_lr_factor: 0.25
label_smoothing: 0.1
lr_scheduler_type: warmup_stable_decay
lr_scheduler_kwargs:
  num_decay_steps: 20000
  min_lr_ratio: 0.1
non_speech_manifest: null
ctc_loss_weight: 0.0
eval_wer_samples_per_language: 100
""",
                encoding="utf-8",
            )
            config = TrainConfig.from_yaml(path)
            config.validate()
            self.assertEqual(config.llm_lr_factor, 0.5)
            self.assertEqual(config.lr_scheduler_kwargs["num_decay_steps"], 20000)
            self.assertEqual(config.non_speech_manifest_specs, [])

    def test_unknown_keys_still_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.yaml"
            path.write_text("train_manifest: a.json\nbogus_key: 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                TrainConfig.from_yaml(path)

    def test_validation_bounds(self) -> None:
        config = TrainConfig(train_manifest={"ar": ["a.json"]})
        config.label_smoothing = 0.5
        with self.assertRaises(ValueError):
            config.validate()
        config.label_smoothing = 0.1
        config.llm_lr_factor = 2.0
        with self.assertRaises(ValueError):
            config.validate()
        config.llm_lr_factor = 0.5
        config.sampling_temperature = 0.0
        with self.assertRaises(ValueError):
            config.validate()


if __name__ == "__main__":
    unittest.main()
