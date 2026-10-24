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
import torch
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
    ctc_text = text
    ctc_count = None
    if processor.config.ctc_version:
        # Include a feasible CTC path as well as near-limit AR targets in every batch.
        frames = processor.audio_token_lengths(torch.ones(1, int(seconds * 100), dtype=torch.bool))[
            0
        ].item()
        while True:
            ctc_ids = processor.ctc_tokenizer.encode(ctc_text, add_special_tokens=False).ids
            required = len(ctc_ids) + sum(a == b for a, b in zip(ctc_ids, ctc_ids[1:]))
            if ctc_ids and required <= frames:
                ctc_count = len(ctc_ids)
                break
            ctc_text = ctc_text[: len(ctc_text) // 2].strip()
            if not ctc_text:
                raise ValueError("Could not construct a feasible CTC memory probe target")
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
        updates = world * accumulation * steps
        manifests = {"en": [str(destination / manifest.name)]}
        quotas = {"en": micro * accumulation}
        if ctc_count is None:
            manifest.write_text((json.dumps(row) + "\n") * (updates * micro))
        else:
            # Separate sampler quota guarantees one feasible CTC example in every
            # microbatch despite shuffling. Language prefixes here are synthetic.
            eligible_manifest = stage / "synthetic-ctc.jsonl"
            eligible_manifest.write_text((json.dumps(dict(row, text=ctc_text)) + "\n") * updates)
            manifests = {"ar": [str(destination / eligible_manifest.name)]}
            quotas = {"ar": 1}
            if micro > 1:
                manifest.write_text((json.dumps(row) + "\n") * (updates * (micro - 1)))
                manifests["en"] = [str(destination / manifest.name)]
                quotas["en"] = micro - 1
        data = dict(
            train_manifest=manifests,
            eval_manifest=None,
            language="en",
            min_duration_seconds=0.1,
            max_duration_seconds=seconds,
            max_target_length=limit,
            truncate_long_transcripts=False,
            balanced_sampling=True,
            sampling_strategy="weighted_stratified",
            language_batch_counts=quotas,
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
                    efficiency=copy.deepcopy(recipe.get("efficiency", {})),
                    ctc={"warmup_steps": 0},
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
            transcript_tokens=(
                len(processor.tokenizer.encode(ctc_text, add_special_tokens=False))
                if ctc_count is not None and micro == 1
                else token_count
            ),
            feasible_ctc_tokens=ctc_count,
            guaranteed_ctc_examples_per_microbatch=1 if ctc_count is not None else 0,
            ctc_vocab_size=processor.config.ctc_vocab_size,
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
