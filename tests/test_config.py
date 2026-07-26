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


if __name__ == "__main__":
    unittest.main()
