"""Tests for the augmentation builder and phase-config coverage."""

import unittest
from typing import ClassVar

from qasr.augmentation import build_augmenter

# The full Phase A augmentation recipe, shared verbatim by all four training
# YAMLs (projector / full / HQ / EAGLE).
FULL_RECIPE = {
    "spec_augment": {
        "enabled": True,
        "num_time_masks": 2,
        "max_time_mask_ratio": 0.05,
        "num_freq_masks": 2,
        "max_freq_mask_ratio": 0.15,
        "p": 0.37,
    },
    "speed_perturb": {
        "enabled": True,
        "rate_range": [0.85, 1.15],
        "p": 0.37,
    },
    "noise_injection": {
        "enabled": True,
        "noise_dir": "/tmp/nonexistent-musan",
        "snr_range": [5.0, 20.0],
        "p": 0.37,
    },
    "codec_augment": {
        "enabled": True,
        "p": 0.37,
    },
}


class BuildAugmenterTest(unittest.TestCase):
    def test_missing_or_empty_config_returns_none(self) -> None:
        self.assertIsNone(build_augmenter(None, sampling_rate=16000))
        self.assertIsNone(build_augmenter({}, sampling_rate=16000))

    def test_full_recipe_enables_all_four_augmentations(self) -> None:
        augmenter = build_augmenter(FULL_RECIPE, sampling_rate=16000)
        self.assertIsNotNone(augmenter)
        self.assertTrue(augmenter.has_waveform_augmentation)
        self.assertTrue(augmenter.has_spec_augmentation)
        self.assertIsNotNone(augmenter.speed_perturb)
        self.assertIsNotNone(augmenter.noise_injection)
        self.assertIsNotNone(augmenter.codec_augment)
        self.assertIsNotNone(augmenter.spec_augment)

    def test_partial_config_merges_over_defaults(self) -> None:
        augmenter = build_augmenter(
            {"codec_augment": {"enabled": True, "p": 0.5}},
            sampling_rate=16000,
        )
        self.assertIsNotNone(augmenter.codec_augment)
        self.assertIsNone(augmenter.speed_perturb)
        self.assertIsNone(augmenter.noise_injection)
        self.assertIsNone(augmenter.spec_augment)
        # Untouched sections keep their disabled defaults.
        self.assertFalse(augmenter.config.spec_augment["enabled"])

    def test_unknown_section_is_ignored(self) -> None:
        augmenter = build_augmenter(
            {"not_a_real_augmentation": {"enabled": True}},
            sampling_rate=16000,
        )
        self.assertFalse(augmenter.has_waveform_augmentation)
        self.assertFalse(augmenter.has_spec_augmentation)


class TrainingYamlAugmentationCoverageTest(unittest.TestCase):
    """Every training phase must ship the full augmentation recipe."""

    FULL_PHASE_YAMLS: ClassVar[list[str]] = [
        "configs/train_projector_8node.yaml",
        "configs/train_projector_4node.yaml",
        "configs/train_projector_4node_filtered.yaml",
        "configs/train_full_8node_filtered.yaml",
        "configs/train_full_4node.yaml",
        "configs/train_full_4node_filtered.yaml",
        "configs/train_hq_8node.yaml",
    ]

    def test_full_phase_configs_enable_all_augmentations(self) -> None:
        from qasr.config import TrainConfig

        for path in self.FULL_PHASE_YAMLS:
            with self.subTest(config=path):
                config = TrainConfig.from_yaml(path)
                config.validate()
                self.assertIsNotNone(config.augmentation, f"{path} has no augmentation block")
                augmenter = build_augmenter(config.augmentation, sampling_rate=16000)
                self.assertTrue(
                    augmenter.has_spec_augmentation, f"{path} missing SpecAugment"
                )
                self.assertIsNotNone(augmenter.speed_perturb, f"{path} missing speed perturb")
                self.assertIsNotNone(augmenter.noise_injection, f"{path} missing noise injection")
                self.assertIsNotNone(augmenter.codec_augment, f"{path} missing codec augment")

    def test_eagle_config_enables_all_augmentations(self) -> None:
        from qasr.train_eagle import _load_yaml_config

        config = _load_yaml_config("configs/train_eagle_8node_filtered.yaml")
        self.assertIsNotNone(config.augmentation)
        augmenter = build_augmenter(config.augmentation, sampling_rate=16000)
        self.assertTrue(augmenter.has_spec_augmentation)
        self.assertTrue(augmenter.has_waveform_augmentation)


if __name__ == "__main__":
    unittest.main()
