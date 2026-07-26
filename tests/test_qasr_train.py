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
        projector = TrainConfig.from_yaml(root / "configs/train_v2.yaml")
        full = TrainConfig.from_yaml(root / "configs/train_v3.yaml")

        self.assertEqual(len(projector.train_manifest_specs), 22)
        self.assertEqual(len(projector.eval_manifest_specs), 6)
        self.assertEqual(projector.train_manifest_specs, full.train_manifest_specs)
        self.assertEqual(projector.eval_manifest_specs, full.eval_manifest_specs)
        self.assertEqual(projector.per_device_train_batch_size, 8)
        self.assertEqual(projector.gradient_accumulation_steps, 2)
        self.assertEqual(full.per_device_train_batch_size, 16)
        self.assertEqual(full.gradient_accumulation_steps, 1)

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


if __name__ == "__main__":
    unittest.main()
