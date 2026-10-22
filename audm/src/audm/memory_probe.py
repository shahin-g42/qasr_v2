"""Prepare a synthetic longest-shape rehearsal for the normal AUD distributed trainer.

This prepares files only. Run the printed recipe with train_distributed.sh; never
use the resulting probe checkpoint as a trained ASR model.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from .prepare_non_speech import new_directory
from .processing import AUDProcessor
from .training_config import load_recipe


def prepare(config, model, output, steps=2):
    config = Path(config).resolve()
    recipe = yaml.safe_load(config.read_text())
    source = yaml.safe_load((config.parent / recipe["data_recipe"]).read_text())
    training = copy.deepcopy(recipe["training"])
    processor = AUDProcessor.from_pretrained(model)
    seconds = min(
        training.get("max_duration_seconds", source.get("max_duration_seconds", 30)),
        processor.config.max_audio_seconds,
    )
    limit = training.get("max_target_length", source.get("max_target_length", 512))
    micro = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    world = training["expected_world_size"]
    if type(steps) is not int or steps < 2:
        raise ValueError("Use at least two updates to exercise initialized optimizer state")
    ids = processor.tokenizer.encode("hello " * limit, add_special_tokens=False)[:limit]
    text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
    while len(processor.tokenizer.encode(text, add_special_tokens=False)) > limit:
        text = text[:-1]
    token_count = len(processor.tokenizer.encode(text, add_special_tokens=False))
    if token_count < max(1, limit - 8):
        raise ValueError("Could not construct a near-limit transcript with this tokenizer")
    disabled_augmentation = {
        name: {"enabled": False}
        for name in ("speed_perturb", "noise_injection", "codec_augment", "spec_augment")
    }
    with new_directory(output) as (stage, destination):
        wav = stage / "synthetic.wav"
        sf.write(
            wav,
            np.random.default_rng(42).normal(0, 0.01, math.floor(seconds * 16000)),
            16000,
            subtype="PCM_16",
        )
        manifest = stage / "synthetic.jsonl"
        row = dict(audio_filepath=str(destination / wav.name), duration=seconds, text=text)
        manifest.write_text((json.dumps(row) + "\n") * (world * micro * accumulation * steps))
        data = dict(
            train_manifest={"en": [str(destination / manifest.name)]},
            eval_manifest=None,
            language="en",
            min_duration_seconds=0.1,
            max_duration_seconds=seconds,
            max_target_length=limit,
            truncate_long_transcripts=False,
            balanced_sampling=True,
            sampling_strategy="weighted_stratified",
            language_batch_counts={"en": micro * accumulation},
            augmentation=disabled_augmentation,
            code_switch_index=None,
            code_switch_probability=0.0,
        )
        (stage / "data.yaml").write_text(yaml.safe_dump(data))
        if training.get("deepspeed"):
            training["deepspeed"] = str((config.parent / training["deepspeed"]).resolve())
        training.update(
            model_name_or_path=str(Path(model).resolve()),
            output_dir=str(destination / "run"),
            max_duration_seconds=seconds,
            max_target_length=limit,
            max_steps=steps,
            num_train_epochs=1,
            warmup_steps=0,
            eval_steps=steps + 1,
            save_steps=steps + 1,
            logging_steps=1,
            eval_log_samples=0,
            eval_wer_samples_per_language=0,
            report_to="none",
            run_name="aud-memory-probe",
            augmentation=disabled_augmentation,
            code_switch_index=None,
            code_switch_probability=0.0,
        )
        for key in (
            "train_manifest",
            "eval_manifest",
            "non_speech_manifest",
            "language_batch_counts",
            "resume_from_checkpoint",
            "smoke_test",
            "audio_root",
        ):
            training.pop(key, None)
        (stage / "recipe.yaml").write_text(
            yaml.safe_dump(
                dict(
                    data_recipe="data.yaml",
                    stage=recipe["stage"],
                    training=training,
                )
            )
        )
        # Validate without touching the production manifests or running a model.
        load_recipe(stage / "recipe.yaml")
        report = dict(
            config=str(destination / "recipe.yaml"),
            output=str(destination / "run"),
            stage=recipe["stage"],
            audio_seconds=seconds,
            transcript_tokens=token_count,
            microbatch=micro,
            accumulation=accumulation,
            ranks=world,
            updates=steps,
            warning="Synthetic memory test only. Discard its trained weights.",
        )
        (stage / "probe.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
