import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from qasr.config import TrainConfig
from qasr.data import CombinedSpeechDataset
from qasr.train import (
    _batch_norm_modules,
    _disable_training_cache,
    _gradient_checkpointing_kwargs,
    _parameter_counts,
    _sample_subset,
    _StatsSubset,
)


class _Part:
    def __init__(self, values):
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]


class _Combined:
    pass


class SmokeSamplingTest(unittest.TestCase):
    def test_staged_configs_retain_latest_manifest_set(self) -> None:
        root = Path(__file__).resolve().parents[1]
        projector = TrainConfig.from_yaml(root / "configs/train_full_4node_filtered.yaml")
        full = TrainConfig.from_yaml(root / "configs/train_full_8node_filtered.yaml")

        self.assertTrue(projector.train_manifest_specs)
        self.assertEqual(projector.train_manifest_specs, full.train_manifest_specs)
        self.assertEqual(projector.eval_manifest_specs, full.eval_manifest_specs)

    def test_qasr_smoke_config_accepts_combined_manifests(self) -> None:
        config = TrainConfig(
            train_manifest=["train-a.jsonl", "train-b.jsonl"],
            smoke_test=True,
        )

        config.validate()

        self.assertEqual(config.train_manifest_paths, ["train-a.jsonl", "train-b.jsonl"])

    def test_regular_subset_is_deterministic(self) -> None:
        first = _sample_subset(list(range(100)), 10, 7)
        second = _sample_subset(list(range(100)), 10, 7)

        self.assertEqual(first.indices, second.indices)
        self.assertEqual(len(first), 10)

    def test_combined_subset_includes_every_manifest(self) -> None:
        combined = object.__new__(CombinedSpeechDataset)
        combined.datasets = (["a0", "a1"], ["b0", "b1", "b2"], ["c0", "c1"])
        combined.cumulative_sizes = [2, 5, 7]

        subset = _sample_subset(combined, 3, 11)
        selected = set(subset.indices)

        self.assertTrue(selected & {0, 1})
        self.assertTrue(selected & {2, 3, 4})
        self.assertTrue(selected & {5, 6})


class StatsSubsetTest(unittest.TestCase):
    def test_smoke_subsets_stay_sampler_compatible(self) -> None:
        # LanguageBalancedSampler weights parts by stats.total_kept_hours; the
        # smoke-mode shim must forward those stats through the Subset wrapper.
        class StatsPart:
            stats = SimpleNamespace(total_kept_hours=3.0, kept_records=4)

            def __len__(self):
                return 4

            def __getitem__(self, index):
                return index

        subset = _StatsSubset(StatsPart(), [0, 2])

        self.assertEqual(subset.stats.total_kept_hours, 3.0)
        self.assertEqual(len(subset), 2)
        self.assertEqual(subset[1], 2)


class BatchNormTest(unittest.TestCase):
    def test_only_batch_norm_statistics_are_frozen(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.BatchNorm1d(4),
            torch.nn.Sequential(torch.nn.BatchNorm2d(2)),
        )

        modules = _batch_norm_modules(model)
        for module in modules:
            module.eval()

        self.assertEqual(modules, (model[1], model[2][0]))
        self.assertTrue(model[1].weight.requires_grad)
        self.assertFalse(model[1].training)


class ParameterCountTest(unittest.TestCase):
    def test_regular_parameters_use_local_tensor_size(self) -> None:
        model = torch.nn.Linear(3, 2)

        trainable, total = _parameter_counts(model)

        self.assertEqual((trainable, total), (8, 8))

    def test_zero3_placeholders_use_logical_parameter_size(self) -> None:
        model = torch.nn.Module()
        model.trainable = torch.nn.Parameter(torch.empty(0))
        model.trainable.ds_numel = 24
        model.frozen = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        model.frozen.ds_numel = 10

        trainable, total = _parameter_counts(model)

        self.assertEqual((trainable, total), (24, 34))


class GradientCheckpointingTest(unittest.TestCase):
    def test_deepspeed_uses_reentrant_checkpointing(self) -> None:
        config = SimpleNamespace(gradient_checkpointing=True, deepspeed="zero3.json")

        self.assertEqual(_gradient_checkpointing_kwargs(config), {"use_reentrant": True})

    def test_training_without_deepspeed_uses_non_reentrant_checkpointing(self) -> None:
        config = SimpleNamespace(gradient_checkpointing=True, deepspeed=None)

        self.assertEqual(_gradient_checkpointing_kwargs(config), {"use_reentrant": False})

    def test_disabled_gradient_checkpointing_has_no_kwargs(self) -> None:
        config = SimpleNamespace(gradient_checkpointing=False, deepspeed="zero3.json")

        self.assertIsNone(_gradient_checkpointing_kwargs(config))


class TrainingCacheTest(unittest.TestCase):
    def test_decoder_cache_is_disabled_at_every_config_level(self) -> None:
        text_config = SimpleNamespace(use_cache=True)
        decoder_config = SimpleNamespace(use_cache=True)
        model = SimpleNamespace(
            config=SimpleNamespace(use_cache=True, text_config=text_config),
            model=SimpleNamespace(language_model=SimpleNamespace(config=decoder_config)),
        )

        _disable_training_cache(model)

        self.assertFalse(model.config.use_cache)
        self.assertFalse(text_config.use_cache)
        self.assertFalse(decoder_config.use_cache)


class EvalCollatorGateTest(unittest.TestCase):
    """Evaluation dataloaders must use the clean (un-augmented) collator."""

    def _build_trainer(self, eval_collator):
        from unittest.mock import patch

        from transformers import Trainer

        import qasr.train as train_module
        from qasr.train import PredictionLoggingTrainer

        def stub_init(self, *a, **k):
            # Minimal stand-in: the subclass body only needs self.model.
            self.model = k.get("model")

        with patch.object(Trainer, "__init__", stub_init), \
                patch.object(train_module, "_batch_norm_modules", lambda model: []):
            trainer = PredictionLoggingTrainer(
                model=torch.nn.Linear(2, 2),
                data_collator="train-collator",
                prediction_processor=object(),
                prediction_collator=object(),
                eval_log_samples=1,
                eval_generation_max_new_tokens=4,
                eval_collator=eval_collator,
            )
        # The patched parent __init__ skips attribute setup the gate needs.
        trainer.data_collator = "train-collator"
        return trainer

    def test_eval_dataloader_swaps_in_clean_collator_and_restores(self) -> None:
        from unittest.mock import patch

        from transformers import Trainer

        trainer = self._build_trainer(eval_collator="clean-collator")
        seen = {}

        def capture(self, eval_dataset=None):
            seen["collator"] = self.data_collator
            return "eval-dataloader"

        with patch.object(Trainer, "get_eval_dataloader", capture):
            dataloader = trainer.get_eval_dataloader()

        self.assertEqual(dataloader, "eval-dataloader")
        self.assertEqual(seen["collator"], "clean-collator")
        # The training dataloader keeps the augmented collator afterwards.
        self.assertEqual(trainer.data_collator, "train-collator")

    def test_without_eval_collator_parent_behavior_is_untouched(self) -> None:
        from unittest.mock import patch

        from transformers import Trainer

        trainer = self._build_trainer(eval_collator=None)
        seen = {}

        def capture(self, eval_dataset=None):
            seen["collator"] = self.data_collator
            return "eval-dataloader"

        with patch.object(Trainer, "get_eval_dataloader", capture):
            trainer.get_eval_dataloader()

        self.assertEqual(seen["collator"], "train-collator")


if __name__ == "__main__":
    unittest.main()
