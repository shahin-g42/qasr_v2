from __future__ import annotations

import json
import logging
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from .augmentation import AudioAugmenter, AugmentationConfig
from .collator import QASRDataCollator
from .config import TrainConfig, parse_config
from .data import CombinedSpeechDataset, JsonlSpeechDataset, ResilientAudioDataset
from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor


LOGGER = logging.getLogger("qasr")


def _sample_subset(dataset: Any, count: int, seed: int) -> Subset | None:
    if count == 0:
        return None
    if count > len(dataset):
        raise ValueError(f"Smoke test requests {count} samples from a dataset of {len(dataset)}")
    rng = random.Random(seed)
    selected: list[int] = []
    if isinstance(dataset, CombinedSpeechDataset) and count >= len(dataset.datasets):
        previous_size = 0
        for part, cumulative_size in zip(dataset.datasets, dataset.cumulative_sizes):
            selected.append(previous_size + rng.randrange(len(part)))
            previous_size = cumulative_size
    remaining = count - len(selected)
    if remaining:
        selected_set = set(selected)
        candidates = [index for index in range(len(dataset)) if index not in selected_set]
        selected.extend(rng.sample(candidates, remaining))
    rng.shuffle(selected)
    return Subset(dataset, selected)


def _split_dataset(dataset: Any, *, eval_ratio: float, seed: int) -> tuple[Any, Any | None]:
    if eval_ratio == 0:
        return dataset, None
    eval_count = max(1, round(len(dataset) * eval_ratio))
    if eval_count >= len(dataset):
        raise ValueError("The evaluation split must leave at least one training sample")
    eval_indices = set(random.Random(seed).sample(range(len(dataset)), eval_count))
    train_indices = [index for index in range(len(dataset)) if index not in eval_indices]
    return Subset(dataset, train_indices), Subset(dataset, sorted(eval_indices))


def _batch_norm_modules(model: Any) -> tuple[torch.nn.Module, ...]:
    return tuple(
        module
        for module in model.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )


def _parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Return logical trainable and total sizes, including ZeRO-3 parameters."""
    trainable = 0
    total = 0
    for parameter in model.parameters():
        # A parameter initialized under DeepSpeed ZeRO-3 is represented by an
        # empty local tensor on every rank. DeepSpeed records its unpartitioned
        # size in ``ds_numel``.
        count = int(getattr(parameter, "ds_numel", parameter.numel()))
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def _gradient_checkpointing_kwargs(config: TrainConfig) -> dict[str, bool] | None:
    if not config.gradient_checkpointing:
        return None
    # ZeRO-3 releases a module's parameters after its forward hook. PyTorch's
    # non-reentrant checkpoint implementation keeps references to tensors saved
    # during recomputation, which therefore become empty ZeRO placeholders before
    # their metadata is checked. Reentrant checkpointing recomputes the complete
    # module and lets DeepSpeed's backward hooks gather the parameters as intended.
    return {"use_reentrant": config.deepspeed is not None}


def _disable_training_cache(model: QASRForConditionalGeneration) -> None:
    # QASR's outer config does not define the Qwen decoder's forward defaults.
    # Set the nested text config that Qwen3Model actually consults as well.
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.model.language_model.config.use_cache = False


def _log_dataset(name: str, dataset: Any) -> None:
    stats = dataset.stats
    LOGGER.info(
        "%s: %d/%d records kept (%.2f known hours); %d duration-filtered; "
        "%d text-filtered; %d deferred-duration",
        name,
        stats.kept_records,
        stats.total_records,
        stats.total_kept_hours,
        stats.skipped_by_duration,
        stats.skipped_by_text,
        stats.missing_duration,
    )


class PredictionLoggingTrainer(Trainer):
    """Trainer that preserves Conformer statistics and logs prompt-only generations."""

    def __init__(
        self,
        *args: Any,
        prediction_processor: QASRProcessor,
        prediction_collator: QASRDataCollator,
        eval_log_samples: int,
        eval_generation_max_new_tokens: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_batch_norm_modules = _batch_norm_modules(self.model)
        self.prediction_processor = prediction_processor
        self.prediction_collator = prediction_collator
        self.eval_log_samples = eval_log_samples
        self.eval_generation_max_new_tokens = eval_generation_max_new_tokens

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ):
        if model.training:
            for module in self._frozen_batch_norm_modules:
                module.eval()
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _log_prediction_examples(self, eval_dataset: Any) -> None:
        count = min(self.eval_log_samples, len(eval_dataset))
        if count == 0:
            return
        # Draw a fresh random subset each evaluation, seeded by the global step
        # so every rank selects the same indices deterministically.
        rng = random.Random(self.args.seed + self.state.global_step)
        sample_indices = rng.sample(range(len(eval_dataset)), count)
        model_dtype = next(self.model.parameters()).dtype
        progress = tqdm(
            list(enumerate(sample_indices, start=1)),
            desc="Generating evaluation examples",
            unit="sample",
            disable=not self.is_world_process_zero(),
        )
        for position, sample_index in progress:
            sample = eval_dataset[sample_index]
            batch = self.prediction_collator.generation_batch([sample])
            generation_inputs = self._prepare_inputs(batch)
            for name, value in generation_inputs.items():
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    generation_inputs[name] = value.to(dtype=model_dtype)
            prompt_length = generation_inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = self.model.generate(
                    **generation_inputs,
                    max_new_tokens=self.eval_generation_max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            generated_ids = output_ids[:, prompt_length:]
            prediction = self.prediction_processor.decode(
                generated_ids[0],
                return_format="transcription_only",
            ).strip()
            if self.is_world_process_zero():
                LOGGER.info(
                    "Evaluation example %d/%d (dataset index %d)\nGround truth: %s\nPrediction: %s",
                    position,
                    count,
                    sample_index,
                    sample["text"],
                    prediction,
                )

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        was_training = self.model.training
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        try:
            if dataset is not None:
                self._log_prediction_examples(dataset)
        finally:
            if was_training:
                self.model.train()
        return metrics


def _model_kwargs(config: TrainConfig) -> dict[str, Any]:
    dtype = torch.bfloat16 if config.bf16 else torch.float16 if config.fp16 else torch.float32
    kwargs: dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": True}
    if config.attn_implementation:
        kwargs["attn_implementation"] = config.attn_implementation
    return kwargs


def _training_arguments(config: TrainConfig, has_eval: bool) -> TrainingArguments:
    worker_count = min(config.dataloader_num_workers, 2) if config.smoke_test else config.dataloader_num_workers
    output_dir = str(Path(config.output_dir) / "smoke-test") if config.smoke_test else config.output_dir
    return TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=has_eval,
        prediction_loss_only=True,
        eval_on_start=config.smoke_test and has_eval,
        eval_strategy="no" if config.smoke_test else ("steps" if has_eval else "no"),
        eval_steps=None if config.smoke_test or not has_eval else config.eval_steps,
        save_strategy="no" if config.smoke_test else "steps",
        save_steps=config.save_steps,
        save_total_limit=1 if config.smoke_test else config.save_total_limit,
        logging_strategy="steps",
        logging_steps=1 if config.smoke_test else config.logging_steps,
        logging_first_step=True,
        num_train_epochs=1.0 if config.smoke_test else config.num_train_epochs,
        max_steps=config.smoke_test_max_steps if config.smoke_test else config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=_gradient_checkpointing_kwargs(config),
        bf16=config.bf16,
        fp16=config.fp16,
        tf32=config.tf32,
        torch_compile=config.torch_compile,
        dataloader_num_workers=worker_count,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_persistent_workers=config.dataloader_persistent_workers and worker_count > 0,
        remove_unused_columns=False,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        deepspeed=config.deepspeed,
        report_to="none" if config.smoke_test else config.report_to,
        run_name=f"{config.run_name}-smoke" if config.smoke_test and config.run_name else config.run_name,
        seed=config.seed,
        data_seed=config.data_seed,
    )


def _apply_freezing(model: QASRForConditionalGeneration, config: TrainConfig) -> None:
    model.model.audio_tower.requires_grad_(not config.freeze_audio_tower)
    model.model.language_model.requires_grad_(not config.freeze_language_model)
    model.lm_head.requires_grad_(not config.freeze_language_model)
    model.model.multi_modal_projector.requires_grad_(True)
    model.tie_weights()
    trainable, total = _parameter_counts(model)
    if total == 0:
        raise RuntimeError("The loaded model has no parameters")
    LOGGER.info(
        "Trainable parameters: %s/%s (%.2f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )


def run(config: TrainConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    set_seed(config.seed)

    has_eval = bool(config.eval_manifest_paths) or config.eval_split_ratio > 0 or (
        config.smoke_test and config.smoke_test_eval_samples > 0
    )
    training_args = _training_arguments(config, has_eval)
    processor = QASRProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = QASRForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        **_model_kwargs(config),
    )
    _disable_training_cache(model)
    _apply_freezing(model, config)

    feature_limit = float(processor.feature_extractor.max_audio_clip_s)
    if config.max_duration_seconds > feature_limit:
        raise ValueError(
            f"max_duration_seconds={config.max_duration_seconds} exceeds the frontend limit "
            f"of {feature_limit}s"
        )

    dataset_kwargs = {
        "min_duration_seconds": config.min_duration_seconds,
        "max_duration_seconds": config.max_duration_seconds,
        "audio_root": config.audio_root,
        "validate_audio_paths": config.validate_audio_paths,
    }
    train_parts = [
        JsonlSpeechDataset(path, language=language, **dataset_kwargs)
        for path, language in config.train_manifest_specs
    ]
    full_train = train_parts[0] if len(train_parts) == 1 else CombinedSpeechDataset(train_parts)
    eval_parts = [
        JsonlSpeechDataset(path, language=language, **dataset_kwargs)
        for path, language in config.eval_manifest_specs
    ]
    full_eval = None
    if len(eval_parts) == 1:
        full_eval = eval_parts[0]
    elif eval_parts:
        full_eval = CombinedSpeechDataset(eval_parts)

    if config.smoke_test:
        train_dataset = _sample_subset(full_train, config.smoke_test_train_samples, config.data_seed)
        eval_source = full_eval if full_eval is not None else full_train
        eval_dataset = _sample_subset(eval_source, config.smoke_test_eval_samples, config.data_seed + 1)
    elif full_eval is not None:
        train_dataset, eval_dataset = full_train, full_eval
    else:
        train_dataset, eval_dataset = _split_dataset(
            full_train,
            eval_ratio=config.eval_split_ratio,
            seed=config.data_seed,
        )

    _log_dataset("train manifests", full_train)
    LOGGER.info("Training manifests by language: %s", dict(Counter(language for _, language in config.train_manifest_specs)))
    if full_eval is not None:
        _log_dataset("eval manifests", full_eval)
        LOGGER.info("Evaluation manifests by language: %s", dict(Counter(language for _, language in config.eval_manifest_specs)))

    global_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.world_size
        * training_args.gradient_accumulation_steps
    )
    LOGGER.info(
        "Global batch size: %d (%d devices x %d samples x %d accumulation)",
        global_batch_size,
        training_args.world_size,
        training_args.per_device_train_batch_size,
        training_args.gradient_accumulation_steps,
    )
    LOGGER.info(
        "Approximately %d updates/epoch",
        math.ceil(len(train_dataset) / global_batch_size),
    )

    resilient_kwargs = {
        "sampling_rate": int(processor.feature_extractor.sampling_rate),
        "min_audio_seconds": config.min_duration_seconds,
        "max_audio_seconds": config.max_duration_seconds,
        "tokenizer": processor.tokenizer,
        "max_target_length": config.max_target_length,
        "truncate_long_transcripts": config.truncate_long_transcripts,
    }
    train_dataset = ResilientAudioDataset(train_dataset, **resilient_kwargs)
    if eval_dataset is not None:
        eval_dataset = ResilientAudioDataset(eval_dataset, **resilient_kwargs)
    # Build the audio augmenter from config (if provided)
    augmenter: AudioAugmenter | None = None
    if config.augmentation:
        aug_config = AugmentationConfig()
        for aug_name, aug_params in config.augmentation.items():
            if hasattr(aug_config, aug_name) and isinstance(aug_params, dict):
                merged = getattr(aug_config, aug_name).copy()
                merged.update(aug_params)
                setattr(aug_config, aug_name, merged)
        augmenter = AudioAugmenter(aug_config, sampling_rate=int(processor.feature_extractor.sampling_rate))
        if training_args.should_save:
            LOGGER.info("Audio augmentation enabled: %s", [
                name for name in ["spec_augment", "speed_perturb", "noise_injection", "codec_augment"]
                if getattr(aug_config, name).get("enabled", False)
            ])

    collator = QASRDataCollator(
        processor=processor,
        language=config.language,
        min_audio_seconds=config.min_duration_seconds,
        max_audio_seconds=config.max_duration_seconds,
        max_target_length=config.max_target_length,
        augmenter=augmenter,
    )

    if training_args.should_save:
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(output_dir)
        with (output_dir / "training_config.json").open("w", encoding="utf-8") as handle:
            json.dump(config.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    trainer = PredictionLoggingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
        prediction_processor=processor,
        prediction_collator=collator,
        eval_log_samples=min(config.eval_log_samples, 2) if config.smoke_test else config.eval_log_samples,
        eval_generation_max_new_tokens=min(config.eval_generation_max_new_tokens, 64)
        if config.smoke_test
        else config.eval_generation_max_new_tokens,
    )
    LOGGER.info(
        "Keeping pretrained running statistics fixed for %d BatchNorm modules",
        len(trainer._frozen_batch_norm_modules),
    )
    resume = None if config.smoke_test else config.resume_from_checkpoint
    result = trainer.train(resume_from_checkpoint=resume)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    if eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_model()
    trainer.save_state()


def main(argv: list[str] | None = None) -> None:
    run(parse_config(argv))


if __name__ == "__main__":
    main()
