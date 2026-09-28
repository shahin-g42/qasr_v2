"""Tests for the EAGLE draft head and its distillation wrapper.

eagle.py previously had zero coverage; these pin the two production-critical
behaviors: the head trains at all, and — the fix this file was written for —
the EAGLE-3 layer-fusion weights actually receive gradients (they used to sit
inside a ``torch.no_grad()`` block and stayed frozen at their zero init for
the whole run, silently degrading fusion to a uniform average).
"""

import os
import tempfile
import unittest
from unittest import mock

import torch
from test_qasr_model import AUDIO_TOKEN_ID, tiny_config

from qasr import QASRForConditionalGeneration
from qasr.eagle import EagleConfig, EagleHead, compute_eagle_loss
from qasr.train_eagle import (
    EagleDistillModel,
    EagleTrainConfig,
    _configure_experiment_tracking,
    _load_yaml_config,
)


def _toy_batch() -> dict:
    input_ids = torch.tensor([[1, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, AUDIO_TOKEN_ID, 5, 6]])
    labels = torch.full_like(input_ids, -100)
    labels[0, -2:] = input_ids[0, -2:]
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "input_features": torch.randn(1, 17, 128),
        "input_features_mask": torch.ones(1, 17, dtype=torch.long),
        "labels": labels,
    }


def _head(fusion: tuple[int, ...]) -> EagleHead:
    return EagleHead(
        EagleConfig(hidden_size=16, vocab_size=128, fusion_layer_indices=fusion)
    )


class EagleDistillGradientTest(unittest.TestCase):
    def _wrapper(self, fusion: tuple[int, ...]) -> tuple[EagleDistillModel, EagleHead]:
        torch.manual_seed(11)
        teacher = QASRForConditionalGeneration(tiny_config())
        teacher.eval()
        teacher.requires_grad_(False)
        head = _head(fusion)
        with torch.no_grad():
            head.lm_head.weight.copy_(teacher.lm_head.weight)
        head.lm_head.weight.requires_grad_(False)
        return EagleDistillModel(teacher, head, kl_temperature=1.0), head

    def test_fusion_layer_weights_receive_gradients(self) -> None:
        """The EAGLE-3 mix weights must train — the historical bug left them
        at zero init because fusion ran inside the teacher's no_grad block."""
        wrapper, head = self._wrapper(fusion=(-2, -1))
        self.assertIsNotNone(head.layer_weights)

        loss = wrapper(**_toy_batch())["loss"]
        loss.backward()

        self.assertIsNotNone(head.layer_weights.grad, "fusion weights got no gradient")
        self.assertGreater(float(head.layer_weights.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.fc1.weight.grad)

    def test_teacher_stays_frozen_and_lm_head_untrained(self) -> None:
        wrapper, head = self._wrapper(fusion=(-2, -1))
        loss = wrapper(**_toy_batch())["loss"]
        loss.backward()

        self.assertIsNone(head.lm_head.weight.grad)
        for parameter in wrapper.qasr.parameters():
            self.assertIsNone(parameter.grad)

    def test_single_layer_head_still_trains(self) -> None:
        wrapper, head = self._wrapper(fusion=(-1,))
        self.assertIsNone(head.layer_weights)

        loss = wrapper(**_toy_batch())["loss"]
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(head.fc1.weight.grad)


class EagleLossTest(unittest.TestCase):
    def test_fully_masked_batch_yields_zero_loss_with_graph(self) -> None:
        head = _head(fusion=(-1,))
        hidden = torch.randn(1, 4, 16)
        embeds = torch.randn(1, 4, 16)
        logits = torch.randn(1, 4, 128)

        loss = compute_eagle_loss(
            eagle_head=head,
            hidden_states=hidden,
            token_embeds=embeds,
            target_logits=logits,
            label_mask=torch.zeros(1, 4, dtype=torch.bool),
            temperature=1.0,
        )

        self.assertEqual(float(loss), 0.0)
        loss.backward()  # DDP keep-alive guard must produce a real graph


class ExperimentTrackingTest(unittest.TestCase):
    """Pin the EAGLE observability wiring.

    ``EagleTrainConfig.report_to`` used to default to ``"none"`` and the config
    had no ``wandb_project``/``run_name`` fields at all, so a multi-million-step
    distillation run produced no graph anywhere and there was no way to name it.
    These tests fail loudly if that regresses, because the failure mode is
    silent: training looks healthy right up until you go looking for the curve.
    """

    _TRACKING_KEYS = ("WANDB_PROJECT", "WANDB_LOG_MODEL", "WANDB_WATCH")

    def setUp(self) -> None:
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for key in self._TRACKING_KEYS:
            os.environ.pop(key, None)

    def test_report_to_defaults_to_wandb(self) -> None:
        config = EagleTrainConfig()
        self.assertEqual(config.report_to, "wandb")
        self.assertEqual(config.wandb_project, "qasr")
        self.assertIsNone(config.run_name)

    def test_tracking_exports_project_environment(self) -> None:
        config = EagleTrainConfig()
        config.wandb_project = "qasr-eagle-v76"
        config.run_name = "adhoc-1node"

        reporters = _configure_experiment_tracking(config)

        self.assertEqual(reporters, ["wandb"])
        self.assertEqual(os.environ["WANDB_PROJECT"], "qasr-eagle-v76")
        # The frozen 3.62B teacher must never be uploaded or autograd-hooked.
        self.assertEqual(os.environ["WANDB_LOG_MODEL"], "false")
        self.assertEqual(os.environ["WANDB_WATCH"], "false")

    def test_report_to_none_leaves_environment_untouched(self) -> None:
        config = EagleTrainConfig()
        config.report_to = "none"
        config.wandb_project = ""  # would raise if tracking were attempted

        reporters = _configure_experiment_tracking(config)

        self.assertEqual(reporters, ["none"])
        for key in self._TRACKING_KEYS:
            self.assertNotIn(key, os.environ)

    def test_list_report_to_is_normalized(self) -> None:
        config = EagleTrainConfig()
        config.report_to = ["tensorboard", "wandb"]
        config.wandb_project = "p"

        self.assertEqual(_configure_experiment_tracking(config), ["tensorboard", "wandb"])
        self.assertEqual(os.environ["WANDB_PROJECT"], "p")

    def test_blank_project_raises_instead_of_tracking_unnamed(self) -> None:
        for blank in ("", "   ", None):
            with self.subTest(wandb_project=blank):
                config = EagleTrainConfig()
                config.wandb_project = blank
                with self.assertRaises(ValueError):
                    _configure_experiment_tracking(config)

    def test_yaml_loads_tracking_fields_without_dropping_them(self) -> None:
        # _load_yaml_config silently ignores unknown keys, so a field that is
        # missing from the dataclass disappears with only a warning.
        yaml_text = (
            "report_to: wandb\n"
            "wandb_project: qasr-eagle-v76\n"
            "run_name: eagle_v2_1node_ckpt150k_fullpool\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write(yaml_text)
            path = handle.name
        self.addCleanup(os.unlink, path)

        config = _load_yaml_config(path)

        self.assertEqual(config.report_to, "wandb")
        self.assertEqual(config.wandb_project, "qasr-eagle-v76")
        self.assertEqual(config.run_name, "eagle_v2_1node_ckpt150k_fullpool")
        self.assertEqual(_configure_experiment_tracking(config), ["wandb"])
        self.assertEqual(os.environ["WANDB_PROJECT"], "qasr-eagle-v76")

    def test_explicit_yaml_none_overrides_the_wandb_default(self) -> None:
        # The 8-node EAGLE config still opts out explicitly; that must keep
        # winning over the new default so its behaviour does not change.
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
            handle.write("report_to: none\n")
            path = handle.name
        self.addCleanup(os.unlink, path)

        config = _load_yaml_config(path)

        self.assertEqual(config.report_to, "none")
        self.assertEqual(_configure_experiment_tracking(config), ["none"])
        self.assertNotIn("WANDB_PROJECT", os.environ)


if __name__ == "__main__":
    unittest.main()
