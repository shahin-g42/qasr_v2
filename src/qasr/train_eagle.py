"""Train the EAGLE-2 draft head on a frozen QASR checkpoint.

The EAGLE head learns to predict the target model's distribution one token
into the future from its last-layer hidden state plus the embedding of the
next token. Only the ~8.4M-parameter fusion layer (fc1 + norm) is trained;
the draft lm_head is copied from the target model and frozen, as is the full
QASR model.

Training runs on the same stack as Phase 2: HF Trainer + Accelerate +
DeepSpeed ZeRO-2, launched per-node with torchrun. The frozen teacher runs
under ``torch.no_grad`` inside the wrapper's forward, so DeepSpeed only
manages gradients/optimizer state for the tiny trainable head. Checkpoints
contain just the EAGLE head (``eagle_head.pt``), never the 8 GB teacher.

Usage:
    PYTHONPATH=src python -m qasr.train_eagle --config configs/train_eagle_8node_filtered.yaml
    torchrun --nproc_per_node=8 train_eagle.py --config configs/train_eagle_8node_filtered.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from .augmentation import build_augmenter
from .collator import QASRDataCollator
from .config import validate_local_checkpoint
from .data import CombinedSpeechDataset, JsonlSpeechDataset, ResilientAudioDataset
from .eagle import EagleConfig, EagleHead, compute_eagle_loss
from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor
from .sampling import StratifiedLanguageSampler

LOGGER = logging.getLogger("qasr.eagle.train")


@dataclass
class EagleTrainConfig:
    """Configuration specific to EAGLE head training."""

    # Model paths
    model_name_or_path: str = "/lustrefs/shared/shahin.konadath/workspace/expmt/qasr/full"
    eagle_output_dir: str = "/lustrefs/shared/shahin.konadath/workspace/expmt/qasr/eagle"

    # EAGLE architecture
    hidden_size: int = 2048
    vocab_size: int = 151936
    num_draft_tokens: int = 5
    # EAGLE-3 multi-layer fusion: indices into outputs.hidden_states
    # (0 = embedding output, -1 = last layer). Keep the legacy single-layer
    # default unless retraining the head with fusion enabled.
    fusion_layer_indices: tuple[int, ...] = (-1,)

    # Training hyperparameters
    max_steps: int = 2000
    per_device_train_batch_size: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    optim: str = "adamw_torch"
    kl_temperature: float = 1.0

    # Data (reuses the same manifest format as QASR training)
    train_manifest: Any = ""
    language: str = "ar"
    min_duration_seconds: float = 0.1
    max_duration_seconds: float = 30.0
    max_target_length: int = 256
    # "random"     -> HF Trainer's default RandomSampler (one shuffled pass,
    #                 no per-batch language guarantee).
    # "stratified" -> StratifiedLanguageSampler: equal languages per mini-batch
    #                 and every record of every language seen at least once per
    #                 epoch (small languages repeat). Requires
    #                 per_device_train_batch_size % num_languages == 0.
    sampling_strategy: str = "random"
    # Permit max_steps < one full stratified epoch (quick experiments only;
    # the default refuses so a broken multi-node launch cannot silently
    # truncate coverage).
    allow_partial_epoch: bool = False
    # Data augmentation (same schema as Phase 2 training YAMLs). Applied to
    # the teacher's input audio so the head learns to draft on augmented
    # spectrograms, matching what the deployed model sees during fine-tune.
    augmentation: dict[str, Any] | None = None

    # Infrastructure (uniform with Phase 2: Trainer + DeepSpeed ZeRO-2)
    deepspeed: str | None = None
    bf16: bool = True
    tf32: bool = True
    # Kept for config compatibility; the teacher runs under no_grad so
    # activation checkpointing has nothing to save.
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 8
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int | None = None
    # Experiment tracking. Defaults mirror TrainConfig so an EAGLE run lands in
    # wandb next to the runs it distills from. The previous "none" default meant
    # a multi-million-step run produced no graph at all, and with no
    # ``wandb_project``/``run_name`` fields there was no way to make the run
    # identifiable even after flipping it in the YAML.
    report_to: str | list[str] = "wandb"
    wandb_project: str = "qasr"
    run_name: str | None = None
    seed: int = 42

    # Model loading
    attn_implementation: str = "sdpa"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> EagleTrainConfig:
        config = cls()
        for key in vars(args):
            if hasattr(config, key) and getattr(args, key) is not None:
                setattr(config, key, getattr(args, key))
        return config


class EagleDistillModel(nn.Module):
    """Frozen QASR teacher + trainable EAGLE head, packaged for the HF Trainer.

    The forward pass runs the teacher under ``torch.no_grad`` and returns the
    KL distillation loss, so the standard Trainer/Accelerate/DeepSpeed loop
    only backprops through the ~8.4M-parameter fusion layer of the head.
    """

    def __init__(
        self,
        qasr: QASRForConditionalGeneration,
        eagle_head: EagleHead,
        kl_temperature: float,
    ) -> None:
        super().__init__()
        self.qasr = qasr
        self.eagle_head = eagle_head
        self.kl_temperature = kl_temperature
        # Trainer/DeepSpeed introspect model.config (e.g. for "auto" values).
        self.config = qasr.config

    def train(self, mode: bool = True) -> EagleDistillModel:
        # Keep the teacher in eval mode (no dropout) so distillation targets
        # stay deterministic; only the EAGLE head follows train/eval.
        super().train(mode)
        self.qasr.eval()
        return self

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Frozen teacher forward: hidden states (optionally multi-layer fused),
        # target logits, and the input token embeddings the lookahead head
        # conditions on.
        with torch.no_grad():
            outputs = self.qasr(
                input_ids=input_ids,
                input_features=input_features,
                input_features_mask=input_features_mask,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            target_logits = outputs.logits.detach()
            token_embeds = self.qasr.get_input_embeddings()(input_ids).detach()
        if self.eagle_head.fuses_multiple_layers:
            # OUTSIDE the no_grad block: the learnable layer-mix weights must
            # receive gradients. Inside no_grad the softmax mix is recorded
            # with no graph and layer_weights stays frozen at its zero init.
            # The teacher activations themselves are detached — only the mix
            # coefficients train.
            hidden_states = self.eagle_head.fuse_hidden_states(
                tuple(h.detach() for h in outputs.hidden_states)
            )
        else:
            hidden_states = outputs.hidden_states[-1].detach()

        # Distill one step ahead on positions where the conditioned-on token is
        # a real transcript token (labels != -100).
        label_mask = labels != -100 if labels is not None else None
        loss = compute_eagle_loss(
            eagle_head=self.eagle_head,
            hidden_states=hidden_states,
            token_embeds=token_embeds,
            target_logits=target_logits,
            label_mask=label_mask,
            temperature=self.kl_temperature,
        )
        return {"loss": loss}


class _EpochAwareDataLoader(DataLoader):
    """DataLoader exposing ``set_epoch`` so the Trainer advances the sampler."""

    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)


class EagleTrainer(Trainer):
    """Trainer that checkpoints only the EAGLE head, never the 8 GB teacher."""

    def __init__(
        self,
        *args,
        sampler_parts: list[JsonlSpeechDataset] | None = None,
        sampling_strategy: str = "random",
        allow_partial_epoch: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._sampler_parts = sampler_parts
        self._sampling_strategy = sampling_strategy
        self._allow_partial_epoch = allow_partial_epoch

    def get_train_dataloader(self) -> DataLoader:
        if self._sampling_strategy != "stratified" or not self._sampler_parts:
            return super().get_train_dataloader()
        sampler = StratifiedLanguageSampler(
            self._sampler_parts,
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.seed,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
        )
        if self.args.process_index == 0:
            repeats = ", ".join(
                f"{language} x{factor:.1f}"
                for language, factor in sorted(sampler.repeat_factors.items())
            )
            LOGGER.info(
                "Stratified sampling: %d languages, %d per language per batch, "
                "%d draws/language/epoch (%s)",
                len(sampler.languages),
                sampler.per_language,
                sampler.epoch_draws,
                repeats,
            )
        # Full-coverage guard, mirroring PredictionLoggingTrainer: refuse a
        # max_steps that cannot fit one stratified epoch at the LIVE world
        # size, so a broken multi-node launch fails loudly instead of
        # distilling on a fraction of the data.
        if (
            self.args.max_steps
            and self.args.max_steps > 0
            and not self._allow_partial_epoch
            and self.args.max_steps < sampler.num_batches
        ):
            covered = self.args.max_steps / sampler.num_batches
            raise RuntimeError(
                f"max_steps ({self.args.max_steps:,}) covers only "
                f"{covered:.1%} of one stratified epoch at world_size "
                f"{self.args.world_size} ({sampler.num_batches:,} steps "
                "needed). Fix the launch fan-out or recompute max_steps; "
                "set allow_partial_epoch: true for quick experiments."
            )
        # Rank-distinct worker base seeds (see PredictionLoggingTrainer).
        generator = torch.Generator()
        generator.manual_seed(self.args.seed * 100_003 + self.args.process_index)
        return _EpochAwareDataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
            generator=generator,
        )

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False) -> None:
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.args.should_save:
            wrapper = self.accelerator.unwrap_model(self.model)
            wrapper.eagle_head.save_pretrained(Path(output_dir))
            LOGGER.info("Saved EAGLE head to %s", output_dir)


def _build_dataset(
    config: EagleTrainConfig, processor: QASRProcessor
) -> tuple[ResilientAudioDataset, list[JsonlSpeechDataset]]:
    dataset_kwargs = {
        "min_duration_seconds": config.min_duration_seconds,
        "max_duration_seconds": config.max_duration_seconds,
    }
    if isinstance(config.train_manifest, dict):
        parts = []
        for language, paths in config.train_manifest.items():
            if isinstance(paths, str):
                paths = [paths]
            for path in paths:
                parts.append(JsonlSpeechDataset(path, language=language, **dataset_kwargs))
    elif isinstance(config.train_manifest, list):
        parts = [
            JsonlSpeechDataset(p, language=config.language, **dataset_kwargs)
            for p in config.train_manifest
        ]
    else:
        parts = [JsonlSpeechDataset(config.train_manifest, language=config.language, **dataset_kwargs)]

    dataset = parts[0] if len(parts) == 1 else CombinedSpeechDataset(parts)
    resilient = ResilientAudioDataset(
        dataset,
        sampling_rate=int(processor.feature_extractor.sampling_rate),
        min_audio_seconds=config.min_duration_seconds,
        max_audio_seconds=config.max_duration_seconds,
        tokenizer=processor.tokenizer,
        max_target_length=config.max_target_length,
    )
    return resilient, parts


def _configure_experiment_tracking(config: EagleTrainConfig) -> list[str]:
    """Export the ``WANDB_*`` environment the Trainer's callback relies on.

    Mirrors ``train.run()`` (Phases 1-3). HF's WandbCallback reads the project
    from ``WANDB_PROJECT`` rather than from ``TrainingArguments``, so without
    this an EAGLE run lands in wandb's default "uncategorized" project even with
    ``report_to: wandb``. ``LOG_MODEL``/``WATCH`` stay off for the same reason as
    the other phases: the 3.62B frozen teacher must never be uploaded or hooked.

    Returns the normalized reporter list so callers and tests can assert on what
    will actually be attached. Raises rather than silently tracking to an
    unnamed project, since that failure is invisible until you look for a graph
    that was never there.
    """
    reporters = [
        str(item)
        for item in (
            config.report_to
            if isinstance(config.report_to, (list, tuple))
            else [config.report_to]
        )
    ]
    if any(item.lower() != "none" for item in reporters):
        if not isinstance(config.wandb_project, str) or not config.wandb_project.strip():
            raise ValueError(
                "wandb_project must be a non-empty string when report_to is enabled"
            )
        os.environ["WANDB_PROJECT"] = config.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        LOGGER.info(
            "Experiment tracking: report_to=%s project=%s run_name=%s",
            reporters, config.wandb_project, config.run_name or "(auto-generated)",
        )
    return reporters


def train_eagle(config: EagleTrainConfig) -> Path:
    """Run EAGLE head training on a frozen QASR model via HF Trainer."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    _configure_experiment_tracking(config)

    dtype = torch.bfloat16 if config.bf16 else torch.float32
    output_dir = Path(config.eagle_output_dir)

    # Fail before the multi-GB teacher load, not at dataloader construction.
    if config.sampling_strategy == "stratified" and isinstance(config.train_manifest, dict):
        num_languages = len(config.train_manifest)
        if num_languages and config.per_device_train_batch_size % num_languages:
            raise ValueError(
                f"per_device_train_batch_size ({config.per_device_train_batch_size}) "
                f"must be a multiple of the number of languages ({num_languages}) "
                "under sampling_strategy: stratified"
            )

    # --- Load frozen QASR teacher (Trainer/Accelerate handle device placement) ---
    validate_local_checkpoint(config.model_name_or_path)
    LOGGER.info("Loading frozen QASR model from %s", config.model_name_or_path)
    model = QASRForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=config.attn_implementation,
    )
    model.eval()
    model.requires_grad_(False)

    processor = QASRProcessor.from_pretrained(config.model_name_or_path)

    # --- Initialize EAGLE head ---
    eagle_config = EagleConfig(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_draft_tokens=config.num_draft_tokens,
        fusion_layer_indices=tuple(config.fusion_layer_indices),
    )
    eagle_head = EagleHead(eagle_config).to(dtype=dtype)
    if eagle_head.fuses_multiple_layers:
        LOGGER.info(
            "EAGLE-3 multi-layer fusion enabled over layers %s",
            eagle_head.fusion_layer_indices,
        )

    # Reuse the target model's output projection instead of learning a ~311M
    # parameter lm_head from scratch. Only the ~8.4M-parameter fusion layer
    # (fc1 + norm) is trained; the copied lm_head stays frozen. This is what
    # makes a few thousand steps of KL distillation sufficient.
    with torch.no_grad():
        eagle_head.lm_head.weight.copy_(model.lm_head.weight.to(dtype=dtype))
    eagle_head.lm_head.weight.requires_grad_(False)

    trainable = sum(p.numel() for p in eagle_head.parameters() if p.requires_grad)
    LOGGER.info(
        "EAGLE head: %d trainable / %d total params (lm_head frozen from target)",
        trainable,
        eagle_head.num_parameters,
    )

    wrapper = EagleDistillModel(model, eagle_head, config.kl_temperature)

    # --- Dataset + collator (identical pipeline to Phase 2) ---
    train_dataset, sampler_parts = _build_dataset(config, processor)
    augmenter = build_augmenter(
        config.augmentation,
        sampling_rate=int(processor.feature_extractor.sampling_rate),
    )
    if augmenter is not None:
        LOGGER.info("EAGLE training augmentation enabled: %s", [
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

    # --- TrainingArguments mirroring the Phase 2 setup ---
    args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        bf16=config.bf16,
        tf32=config.tf32,
        logging_strategy="steps",
        logging_steps=config.logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        # Checkpoints hold just the head; skip optimizer/DeepSpeed engine
        # state so checkpoint-* dirs stay a few hundred MB instead of ~8 GB.
        save_only_model=True,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        deepspeed=config.deepspeed,
        report_to=config.report_to,
        run_name=config.run_name,
        seed=config.seed,
    )

    trainer = EagleTrainer(
        model=wrapper,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
        sampler_parts=sampler_parts,
        sampling_strategy=config.sampling_strategy,
        allow_partial_epoch=config.allow_partial_epoch,
    )
    trainer.train()

    # Final head lands at the output_dir root, matching the previous layout.
    trainer.save_model()
    return output_dir


def _load_yaml_config(path: str) -> EagleTrainConfig:
    """Build an EagleTrainConfig from a YAML file, ignoring unknown keys."""
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    config = EagleTrainConfig()
    known = {f.name for f in fields(config)}
    unknown = sorted(set(data) - known)
    if unknown:
        LOGGER.warning("Ignoring unknown EAGLE config keys: %s", unknown)
    for key, value in data.items():
        if key in known:
            setattr(config, key, value)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train EAGLE-2 draft head for QASR.")
    parser.add_argument("--config", default=None, help="Path to EAGLE YAML config")
    parser.add_argument("--model", default=None, help="QASR checkpoint path")
    parser.add_argument("--output-dir", default=None, help="EAGLE head output directory")
    parser.add_argument("--train-manifest", nargs="+", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-draft-tokens", type=int, default=None)
    parser.add_argument("--kl-temperature", type=float, default=None)
    args = parser.parse_args(argv)

    config = _load_yaml_config(args.config) if args.config else EagleTrainConfig()
    if args.model:
        config.model_name_or_path = args.model
    if args.output_dir:
        config.eagle_output_dir = args.output_dir
    if args.train_manifest:
        config.train_manifest = args.train_manifest
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.num_draft_tokens:
        config.num_draft_tokens = args.num_draft_tokens
    if args.kl_temperature:
        config.kl_temperature = args.kl_temperature

    train_eagle(config)


if __name__ == "__main__":
    main()
