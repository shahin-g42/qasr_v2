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
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from .augmentation import build_augmenter
from .collator import QASRDataCollator
from .config import TrainConfig, parse_config
from .data import CombinedSpeechDataset, JsonlSpeechDataset, ResilientAudioDataset
from .eval_callback import WEREvalCallback
from .losses import make_smoothed_causal_lm_loss
from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor
from .sampling import LanguageBalancedSampler

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
        for part, cumulative_size in zip(dataset.datasets, dataset.cumulative_sizes, strict=True):
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


class _SamplerEpochDataLoader(DataLoader):
    """DataLoader exposing ``set_epoch`` so the Trainer advances the sampler."""

    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


class _StatsSubset(Subset):
    """Subset that exposes the wrapped dataset's ``stats`` (language hours).

    ``LanguageBalancedSampler`` weights parts by ``stats.total_kept_hours``;
    torch's plain ``Subset`` does not forward attribute lookups, so smoke-mode
    per-part subsets need this shim to stay sampler-compatible.
    """

    @property
    def stats(self):
        return self.dataset.stats


class PredictionLoggingTrainer(Trainer):
    """Trainer that preserves Conformer statistics and logs prompt-only generations."""

    def __init__(
        self,
        *args: Any,
        prediction_processor: QASRProcessor,
        prediction_collator: QASRDataCollator,
        eval_log_samples: int,
        eval_generation_max_new_tokens: int,
        balanced_parts: list[JsonlSpeechDataset] | None = None,
        balanced_sampling: bool = False,
        sampling_temperature: float = 0.5,
        llm_lr_factor: float = 1.0,
        embed_lr_factor: float = 1.0,
        eval_collator: QASRDataCollator | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frozen_batch_norm_modules = _batch_norm_modules(self.model)
        self.prediction_processor = prediction_processor
        self.prediction_collator = prediction_collator
        self.eval_log_samples = eval_log_samples
        self.eval_generation_max_new_tokens = eval_generation_max_new_tokens
        self._balanced_parts = balanced_parts
        self._balanced_sampling = balanced_sampling and bool(balanced_parts)
        self._sampling_temperature = sampling_temperature
        self._llm_lr_factor = llm_lr_factor
        self._embed_lr_factor = embed_lr_factor
        # Evaluation batches must stay augmentation-free (no masks, noise, or
        # speed shifts on WER-gating data); swap in the clean collator while
        # the parent builds the eval dataloader.
        self._eval_collator = eval_collator

    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        original = self.data_collator
        self.data_collator = self._eval_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = original

    def get_train_dataloader(self):
        if not self._balanced_sampling or self.train_dataset is None:
            return super().get_train_dataloader()
        sampler = LanguageBalancedSampler(
            self._balanced_parts,
            temperature=self._sampling_temperature,
            seed=self.args.data_seed,
            epoch_size=len(self.train_dataset),
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
        )
        return _SamplerEpochDataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=self.args.dataloader_persistent_workers
            and self.args.dataloader_num_workers > 0,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer(self, model: Any = None) -> torch.optim.Optimizer:
        """Per-group LR split: encoder+projector, LLM body, token embeddings.

        Six AdamW groups ({encoder/projector, LLM, embeddings} x {decay,
        no-decay}). The tied ``lm_head`` weight never appears under its own
        name (``named_parameters`` dedupes by first registration, i.e.
        ``embed_tokens``), so it cannot be double-counted.
        """
        if self.optimizer is not None:
            return self.optimizer
        opt_model = self.model if model is None else model
        decay_names = set(self.get_decay_parameter_names(opt_model))

        embed_prefix = "model.language_model.embed_tokens."
        llm_prefix = "model.language_model."
        encoder_prefixes = (
            "model.audio_tower.",
            "model.multi_modal_projector.",
            "model.pooling_conv.",
            "model.ctc_head.",
        )
        groups: dict[tuple[str, bool], list[torch.Tensor]] = {}
        for name, parameter in opt_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith(embed_prefix):
                key = ("embeddings", name in decay_names)
            elif name.startswith(llm_prefix):
                key = ("llm", name in decay_names)
            elif name.startswith(encoder_prefixes):
                key = ("encoder", name in decay_names)
            else:
                # lm_head (tied) or anything unexpected: route with the LLM
                # body but flag it so nothing silently drops out.
                LOGGER.warning("Optimizer group fallback for parameter %s", name)
                key = ("llm", name in decay_names)
            groups.setdefault(key, []).append(parameter)

        lr_map = {
            "encoder": self.args.learning_rate,
            "llm": self.args.learning_rate * self._llm_lr_factor,
            "embeddings": self.args.learning_rate * self._embed_lr_factor,
        }
        optimizer_grouped_parameters = []
        for (group_name, use_decay), parameters in sorted(groups.items()):
            count = sum(int(getattr(p, "ds_numel", p.numel())) for p in parameters)
            LOGGER.info(
                "Optimizer group %-12s decay=%-5s params=%-12s lr=%.3e",
                group_name,
                use_decay,
                f"{count:,}",
                lr_map[group_name],
            )
            optimizer_grouped_parameters.append(
                {
                    "params": parameters,
                    "weight_decay": self.args.weight_decay if use_decay else 0.0,
                    "lr": lr_map[group_name],
                }
            )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("model", None)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

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
        lr_scheduler_kwargs=config.lr_scheduler_kwargs or None,
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
    # Non-speech / silence manifests join the training mix; their records may
    # have blank transcripts (empty_target_ok) so the model learns to output
    # nothing for them.
    non_speech_parts = [
        JsonlSpeechDataset(path, language=language, empty_target_ok=True, **dataset_kwargs)
        for path, language in config.non_speech_manifest_specs
    ]
    train_parts.extend(non_speech_parts)
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
        # Per-part subsets preserve the CombinedSpeechDataset index layout, so
        # the balanced sampler's smoke run exercises the exact production
        # dataloader path (spec smoke gate: dataloader override active).
        budget_per_part = max(2, -(-config.smoke_test_train_samples // len(train_parts)))
        smoke_parts = []
        for index, part in enumerate(train_parts):
            subset = _sample_subset(part, min(budget_per_part, len(part)), config.data_seed + index)
            if isinstance(subset, Subset) and not isinstance(subset, _StatsSubset):
                subset = _StatsSubset(subset.dataset, subset.indices)
            smoke_parts.append(subset)
        train_dataset = smoke_parts[0] if len(smoke_parts) == 1 else CombinedSpeechDataset(smoke_parts)
        balanced_parts = smoke_parts
        eval_source = full_eval if full_eval is not None else full_train
        eval_dataset = _sample_subset(eval_source, config.smoke_test_eval_samples, config.data_seed + 1)
    elif full_eval is not None:
        train_dataset, eval_dataset = full_train, full_eval
        balanced_parts = train_parts
    else:
        train_dataset, eval_dataset = _split_dataset(
            full_train,
            eval_ratio=config.eval_split_ratio,
            seed=config.data_seed,
        )
        balanced_parts = train_parts

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
    augmenter = build_augmenter(
        config.augmentation,
        sampling_rate=int(processor.feature_extractor.sampling_rate),
    )
    if augmenter is not None and training_args.should_save:
        LOGGER.info("Audio augmentation enabled: %s", [
            name for name in ["spec_augment", "speed_perturb", "noise_injection", "codec_augment"]
            if getattr(augmenter.config, name).get("enabled", False)
        ])

    collator = QASRDataCollator(
        processor=processor,
        language=config.language,
        min_audio_seconds=config.min_duration_seconds,
        max_audio_seconds=config.max_duration_seconds,
        max_target_length=config.max_target_length,
        augmenter=augmenter,
    )
    # Clean collator for evaluation: augmentation is training-only.
    eval_collator = (
        QASRDataCollator(
            processor=processor,
            language=config.language,
            min_audio_seconds=config.min_duration_seconds,
            max_audio_seconds=config.max_duration_seconds,
            max_target_length=config.max_target_length,
            augmenter=None,
        )
        if augmenter is not None
        else None
    )

    # --- Phase A accuracy recipe model-side wiring ---
    # Label smoothing must come through a loss_function swap: transformers
    # 5.14 does not smooth ForCausalLMLoss via label_smoothing_factor.
    if config.label_smoothing > 0:
        model.loss_function = make_smoothed_causal_lm_loss(config.label_smoothing)
        LOGGER.info("Label smoothing enabled: epsilon=%.3f", config.label_smoothing)
    # CTC auxiliary head: created after checkpoint load so old checkpoints stay
    # loadable; the head starts from fresh initialization.
    if config.ctc_loss_weight > 0:
        ctc_head = model.add_ctc_head()
        compute_dtype = next(model.model.multi_modal_projector.parameters()).dtype
        ctc_head.to(dtype=compute_dtype)
        model.ctc_loss_weight = config.ctc_loss_weight
        LOGGER.info(
            "CTC auxiliary head attached (weight=%.2f, vocab=%d+blank)",
            config.ctc_loss_weight,
            ctc_head.out_features - 1,
        )
    # Pool-stride sanity: processor placeholder counts must match the model's
    # projector downsampling or the collator contract check fails anyway.
    model_pool_stride = int(getattr(model.config, "projector_pool_stride", 1))
    processor_pool_stride = int(getattr(processor, "projector_pool_stride", 1))
    if model_pool_stride != processor_pool_stride:
        raise ValueError(
            "projector_pool_stride mismatch: model config has "
            f"{model_pool_stride} but the processor has {processor_pool_stride}; "
            "re-save the processor with the matching stride"
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
        balanced_parts=balanced_parts,
        balanced_sampling=config.balanced_sampling,
        sampling_temperature=config.sampling_temperature,
        llm_lr_factor=config.llm_lr_factor,
        embed_lr_factor=config.embed_lr_factor,
        eval_collator=eval_collator,
    )
    # The smoke gate must show the callback firing at the first evaluation
    # (eval_on_start); cap the decode budget so it stays a quick check.
    wer_samples = config.eval_wer_samples_per_language
    if config.smoke_test:
        wer_samples = min(wer_samples, 2)
    if config.eval_manifest_specs and wer_samples > 0:
        trainer.add_callback(
            WEREvalCallback(
                processor=processor,
                eval_manifest_specs=config.eval_manifest_specs,
                samples_per_language=wer_samples,
                seed=config.seed,
                dataset_kwargs=dataset_kwargs,
                max_new_tokens=config.eval_generation_max_new_tokens,
            )
        )
        LOGGER.info(
            "WER eval callback registered: %d samples/language across %d manifests",
            config.eval_wer_samples_per_language,
            len(config.eval_manifest_specs),
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
