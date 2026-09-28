import tempfile
import unittest
from pathlib import Path

from qasr.config import TrainConfig, parse_config


class TrainConfigTest(unittest.TestCase):
    def test_wandb_logging_is_enabled_by_default(self) -> None:
        config = TrainConfig(train_manifest="train.jsonl")
        config.validate()

        self.assertEqual(config.report_to, "wandb")
        self.assertEqual(config.wandb_project, "qasr")

    def test_train_and_eval_manifest_lists(self) -> None:
        config = TrainConfig(
            train_manifest=["train-a.jsonl", "train-b.jsonl"],
            eval_manifest=["eval-a.jsonl", "eval-b.jsonl"],
            bf16=False,
            tf32=False,
        )
        config.validate()

        self.assertEqual(config.train_manifest_paths, ["train-a.jsonl", "train-b.jsonl"])
        self.assertEqual(config.eval_manifest_paths, ["eval-a.jsonl", "eval-b.jsonl"])

    def test_manifests_can_be_grouped_by_language(self) -> None:
        config = TrainConfig(
            train_manifest={
                "ar": ["ar-a.jsonl", "ar-b.jsonl"],
                "en": ["en.jsonl"],
            },
            eval_manifest={"ar": ["ar-eval.jsonl"], "en": "en-eval.jsonl"},
        )
        config.validate()

        self.assertEqual(
            config.train_manifest_specs,
            [
                ("ar-a.jsonl", "ar"),
                ("ar-b.jsonl", "ar"),
                ("en.jsonl", "en"),
            ],
        )
        self.assertEqual(
            config.eval_manifest_specs,
            [("ar-eval.jsonl", "ar"), ("en-eval.jsonl", "en")],
        )

    def test_prediction_logging_configuration_validation(self) -> None:
        config = TrainConfig(train_manifest="train.jsonl", eval_log_samples=-1)
        with self.assertRaisesRegex(ValueError, "eval_log_samples"):
            config.validate()

    def test_eval_split_and_explicit_eval_are_mutually_exclusive(self) -> None:
        config = TrainConfig(
            train_manifest="train.jsonl",
            eval_manifest="eval.jsonl",
            eval_split_ratio=0.1,
        )
        with self.assertRaisesRegex(ValueError, "either eval_manifest or eval_split_ratio"):
            config.validate()

    def test_smoke_test_accepts_multiple_manifests(self) -> None:
        config = TrainConfig(
            train_manifest=["train-a.jsonl", "train-b.jsonl"],
            smoke_test=True,
        )
        config.validate()

        self.assertEqual(config.train_manifest_paths, ["train-a.jsonl", "train-b.jsonl"])

    def test_cli_enables_smoke_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(
                "train_manifest: train.jsonl\nbf16: false\ntf32: false\n",
                encoding="utf-8",
            )

            config = parse_config(["--config", str(config_path), "--smoke-test"])

            self.assertTrue(config.smoke_test)

    def test_cli_accepts_multiple_train_and_eval_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "train.yaml"
            config_path.write_text(
                "train_manifest: original.jsonl\nbf16: false\ntf32: false\n",
                encoding="utf-8",
            )

            config = parse_config(
                [
                    "--config",
                    str(config_path),
                    "--train-manifest",
                    "train-a.jsonl",
                    "train-b.jsonl",
                    "--eval-manifest",
                    "eval-a.jsonl",
                    "eval-b.jsonl",
                ]
            )

            self.assertEqual(config.train_manifest_paths, ["train-a.jsonl", "train-b.jsonl"])
            self.assertEqual(config.eval_manifest_paths, ["eval-a.jsonl", "eval-b.jsonl"])


class SamplingStrategyValidationTest(unittest.TestCase):
    """Fail-fast guards for the v7.6 stratified full-coverage sampling."""

    def _config(self, **overrides):
        from qasr.config import TrainConfig

        values = {
            "train_manifest": {
                "ar": ["/tmp/a.jsonl"],
                "en": ["/tmp/b.jsonl"],
                "ml": ["/tmp/c.jsonl"],
            },
            "eval_manifest": {"ar": ["/tmp/e.jsonl"]},
            "per_device_train_batch_size": 6,
        }
        values.update(overrides)
        return TrainConfig(**values)

    def test_unknown_strategy_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self._config(sampling_strategy="round_robin").validate()

    def test_stratified_batch_must_divide_by_language_count(self) -> None:
        config = self._config(
            sampling_strategy="stratified", per_device_train_batch_size=4
        )
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("multiple of the number of languages", str(ctx.exception))
        # 6 % 3 languages == 0 -> accepted.
        self._config(sampling_strategy="stratified").validate()

    def test_balanced_sampling_incompatible_with_eval_split_ratio(self) -> None:
        """Both samplers index the unsplit dataset; a split train Subset under
        them mis-maps records and IndexErrors mid-epoch. Must fail at load."""
        config = self._config(eval_manifest=None, eval_split_ratio=0.1)
        with self.assertRaises(ValueError) as ctx:
            config.validate()
        self.assertIn("eval_split_ratio", str(ctx.exception))
        # Turning balanced sampling off restores the legacy split behavior.
        self._config(
            eval_manifest=None, eval_split_ratio=0.1, balanced_sampling=False
        ).validate()


class LocalCheckpointValidationTest(unittest.TestCase):
    """Regression: a missing checkpoint dir surfaced as transformers'
    'Repo id must be in the form ...' HFValidationError, and an INCOMPLETE
    one (interrupted conversion) as a per-file OSError deep inside processor
    loading — both multiplied across 64 ranks. validate_local_checkpoint
    must name the real problem up front."""

    def test_missing_directory_names_the_path(self) -> None:
        from qasr.config import validate_local_checkpoint

        with self.assertRaises(FileNotFoundError) as ctx:
            validate_local_checkpoint("/definitely/not/a/real/checkpoint")
        self.assertIn("does not exist on this node", str(ctx.exception))

    def test_incomplete_checkpoint_lists_missing_artifacts(self) -> None:
        import tempfile
        from pathlib import Path

        from qasr.config import validate_local_checkpoint

        with tempfile.TemporaryDirectory() as tmp:
            # Simulate an interrupted conversion: weights landed, processor
            # artifacts did not.
            Path(tmp, "model.safetensors").touch()
            Path(tmp, "config.json").write_text("{}")
            with self.assertRaises(FileNotFoundError) as ctx:
                validate_local_checkpoint(tmp)
            self.assertIn("INCOMPLETE", str(ctx.exception))
            self.assertIn("preprocessor_config.json", str(ctx.exception))

    def test_complete_checkpoint_passes_in_both_processor_layouts(self) -> None:
        """transformers <5 wrote preprocessor_config.json; 5.x folds the
        feature extractor into processor_config.json. A 5.x-saved checkpoint
        (verified to reload) was wrongly rejected by the first guard version —
        both layouts must pass."""
        import tempfile
        from pathlib import Path

        from qasr.config import validate_local_checkpoint

        for processor_file in ("preprocessor_config.json", "processor_config.json"):
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp, "model.safetensors").touch()
                Path(tmp, "config.json").write_text("{}")
                Path(tmp, processor_file).write_text("{}")
                validate_local_checkpoint(tmp)  # must not raise

    def test_hub_repo_ids_are_left_alone(self) -> None:
        from qasr.config import validate_local_checkpoint

        validate_local_checkpoint("audarai/Audar-ASR-V1.2-Turbo")  # no raise


if __name__ == "__main__":
    unittest.main()
