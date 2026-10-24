"""Prepare a fresh stride-one adaptation/full run from trained AUD weights."""

import argparse
import copy
import json
from pathlib import Path

import yaml

from .configuration import AUDConfig
from .efficiency import resolve_efficiency


def prepare(source, output, adaptation_steps=2000, loss_backend="chunked"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"New run must use a new directory: {output}")
    if type(adaptation_steps) is not int or adaptation_steps < 1:
        raise ValueError("adaptation_steps must be positive")
    config = AUDConfig.from_pretrained(source, local_files_only=True)
    if (config.variant, config.flash_encoder, config.flash_audio_stride) != (
        "audflash",
        "gemma4",
        1,
    ):
        raise ValueError("This recipe preserves a stride-one Gemma AudFlash source")
    if not (source / "model.safetensors").is_file():
        raise ValueError("Source must contain complete AUD model weights")
    receipt_path = source / "training_recipe.json"
    if not receipt_path.is_file():
        receipt_path = source.parent / "training_recipe.json"
    receipt = json.loads(receipt_path.read_text())
    training = copy.deepcopy(receipt["resolved"])
    evaluation = training.get("eval_manifest")
    if not isinstance(evaluation, dict):
        raise ValueError("Expected language-keyed evaluation manifests")
    training["eval_manifest"] = {k: v for k, v in evaluation.items() if k != "zh"}
    if not training["eval_manifest"]:
        raise ValueError("No evaluation languages remain")
    training.update(
        per_device_train_batch_size=40,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=40,
        expected_world_size=8,
        dataloader_num_workers=4,
        dataloader_persistent_workers=True,
        resume_from_checkpoint=None,
        smoke_test=False,
        gradient_checkpointing=True,
        max_target_length=512,
        ctc_loss_weight=0.1,
    )
    efficiency = resolve_efficiency(
        dict(
            loss_backend=loss_backend,
            loss_chunk_size=512,
            duration_bucketing=True,
            index_cache=str(output.parent / "aud-manifest-index-v1"),
            dataloader_timeout=300,
            prefetch_factor=2,
            # Conservative first run. Benchmark less recomputation separately at batch 40.
            encoder_checkpoint_every=1,
            decoder_checkpoint_every=1,
            category_metrics=True,
            distributed_wer=True,
        ),
        output,
    )
    common = dict(data_recipe=receipt["data_recipe"], efficiency=efficiency)
    if receipt.get("robustness"):
        common["robustness"] = copy.deepcopy(receipt["robustness"])
        effects = common["robustness"].get("augmentation", {}).get("effects")
        if effects is not None:
            effects.setdefault("silence", {"p": 0.2, "start_ms": [0, 100], "end_ms": [0, 100]})
    recipes = {}
    for name in ("adaptation", "full"):
        resolved = copy.deepcopy(training)
        resolved.update(
            model_name_or_path=str(
                output / "export" if name == "adaptation" else output / "adaptation"
            ),
            output_dir=str(output / name),
            run_name=f"audflash-stride1-ctc-{name}",
            max_steps=adaptation_steps if name == "adaptation" else -1,
            num_train_epochs=1.0 if name == "adaptation" else 3.0,
            warmup_steps=min(100, max(1, adaptation_steps // 10)) if name == "adaptation" else 500,
            eval_steps=500 if name == "adaptation" else 3700,
            save_steps=500 if name == "adaptation" else 3700,
            label_smoothing=0.0 if name == "adaptation" else 0.1,
        )
        recipes[name] = dict(
            common,
            stage="decoder" if name == "adaptation" else "full",
            training=resolved,
            ctc={"warmup_steps": 100 if name == "adaptation" else 500},
        )
    # Publish only after all validation/serialization succeeds.
    rendered = {
        name: yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
        for name, value in recipes.items()
    }
    report = dict(
        source=str(source),
        output=str(output),
        source_step=None,
        training_batch="40 x 2 x 8 = 640",
        evaluation_batch=40,
        stride=1,
        architecture_changed="Auxiliary CTC head only; encoder/stride-one bridge/decoder unchanged",
        ctc_weight=0.1,
        optimizer_resume=False,
        adaptation_steps=adaptation_steps,
        full_epochs=3,
        augmentation="Inherited source settings plus silence padding (0-100 ms at each boundary) if absent",
        efficiency=efficiency,
    )
    state = source / "trainer_state.json"
    if state.exists():
        report["source_step"] = json.loads(state.read_text())["global_step"]
    from .prepare_non_speech import new_directory

    with new_directory(output) as (staging, _):
        (staging / "logs").mkdir()
        for name, value in rendered.items():
            (staging / f"{name}.yaml").write_text(value)
        (staging / "plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adaptation-steps", type=int, default=2000)
    parser.add_argument("--loss-backend", choices=("chunked", "liger"), default="chunked")
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
