"""QASR Trainer behavior with AUD optimizer groups and accumulation-window quotas."""

import logging
import math
import os
import time

import torch
from qasr.sampling import WeightedStratifiedLanguageSampler
from qasr.train import PredictionLoggingTrainer, _SamplerEpochDataLoader

from .numerics import check_gradients, check_loss, install_zero2_guard


class AUDTrainer(PredictionLoggingTrainer):
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        started = time.monotonic()
        result = super().get_batch_samples(epoch_iterator, num_batches, device)
        self.model.aud_data_wait_seconds = time.monotonic() - started
        batches, _ = result
        if batches and "ctc_targets" in batches[0]:
            from .ctc import eligibility

            count = sum(
                eligibility(b["ctc_targets"], b["ctc_input_lengths"]).sum() for b in batches
            ).to(device)
            if self.args.average_tokens_across_devices:
                count = self.accelerator.gather(count).sum()
            for batch in batches:
                batch["ctc_num_items"] = count
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        if self.is_deepspeed_enabled:
            install_zero2_guard(model.optimizer)
        result = self._aud_training_step(model, inputs, num_items_in_batch)
        if not self.is_deepspeed_enabled and self.accelerator.sync_gradients:
            check_gradients(model)
        return result

    def _aud_training_step(self, model, inputs, num_items_in_batch=None):
        if os.environ.get("AUD_DEBUG_NUMERICS") == "1":
            # Enclose forward AND backward so the traceback identifies the
            # forward operation whose backward first generated NaNs.
            try:
                with torch.autograd.detect_anomaly(check_nan=True):
                    return super().training_step(model, inputs, num_items_in_batch)
            except Exception as error:
                from .numerics import save_failure

                try:
                    report = save_failure(
                        self.args.output_dir,
                        inputs,
                        rank=self.args.process_index,
                        step=self.state.global_step,
                        error=error,
                    )
                    logging.getLogger("audm").error("Saved failing batch: %s", report["batch"])
                except Exception:
                    logging.getLogger("audm").exception("Could not save failing batch")
                raise
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Training and loss-only evaluation do not need vocabulary logits for
        # the audio prompt, padding, or the entire supervised sequence at once.
        # Public model forward/generate keep the native Transformers behavior.
        if "labels" in inputs and (model.training or self.args.prediction_loss_only):
            from .efficiency import DEFAULTS

            efficiency = getattr(self, "aud_efficiency", DEFAULTS)
            inputs = dict(
                inputs,
                loss_only=True,
                use_cache=False,
                loss_smoothing=getattr(self, "aud_label_smoothing", 0.0),
                loss_chunk_size=efficiency["loss_chunk_size"],
                loss_backend=efficiency["loss_backend"],
                category_metrics=model.training and efficiency["category_metrics"],
            )
            if "ctc_targets" in inputs:
                warmup = getattr(self, "aud_ctc_warmup_steps", 100)
                ramp = (
                    min(1.0, self.state.global_step / max(1, warmup))
                    if warmup and model.training
                    else 1.0
                )
                inputs["ctc_weight"] = getattr(self, "aud_ctc_weight", 0.0) * ramp
                self.accelerator.unwrap_model(model).aud_current_ctc_weight = inputs["ctc_weight"]
                if (
                    "ctc_num_items" not in inputs
                    and self.args.average_tokens_across_devices
                    and num_items_in_batch is not None
                ):
                    from .ctc import eligibility

                    inputs["ctc_num_items"] = self.accelerator.gather(
                        eligibility(inputs["ctc_targets"], inputs["ctc_input_lengths"]).sum()
                    ).sum()
            if model.training and {"input_features_mask", "attention_mask"} <= inputs.keys():
                unwrapped = self.accelerator.unwrap_model(model)
                valid = inputs["input_features_mask"]
                sequence = inputs["attention_mask"]
                counters = torch.stack(
                    (
                        valid.sum(),
                        torch.tensor(valid.numel(), device=valid.device),
                        sequence.sum(),
                        torch.tensor(sequence.numel(), device=sequence.device),
                    )
                )
                previous = getattr(unwrapped, "aud_padding_totals", None)
                unwrapped.aud_padding_totals = counters if previous is None else previous + counters
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        if model.training and torch.is_grad_enabled():
            check_loss(result[0] if return_outputs else result)
        return result

    def get_train_dataloader(self):
        if self._sampling_strategy != "weighted_stratified" or not self._balanced_sampling:
            return super().get_train_dataloader()
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        virtual_batch = sum(self._language_batch_counts.values())
        sampler = WeightedStratifiedLanguageSampler(
            self._balanced_parts,
            batch_size=virtual_batch,
            seed=seed,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            language_batch_counts=self._language_batch_counts,
        )
        from .efficiency import DEFAULTS

        efficiency = getattr(self, "aud_efficiency", DEFAULTS)
        if efficiency["duration_bucketing"]:
            from .sampling import UpdateDurationSampler

            sampler = UpdateDurationSampler(
                sampler,
                self._balanced_parts,
                self.args.per_device_train_batch_size,
                self.args.gradient_accumulation_steps,
            )
        updates = math.ceil(
            len(sampler)
            / self.args.per_device_train_batch_size
            / self.args.gradient_accumulation_steps
        )
        if 0 < self.args.max_steps < updates and not self._allow_partial_epoch:
            raise ValueError(
                f"max_steps={self.args.max_steps} covers less than an epoch ({updates} updates)"
            )
        generator = torch.Generator().manual_seed(seed * 100003 + self.args.process_index)
        return _SamplerEpochDataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=False,
            persistent_workers=self.args.dataloader_persistent_workers
            and self.args.dataloader_num_workers > 0,
            generator=generator,
            timeout=efficiency["dataloader_timeout"] if self.args.dataloader_num_workers else 0,
            prefetch_factor=efficiency["prefetch_factor"]
            if self.args.dataloader_num_workers
            else None,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        from .efficiency import DEFAULTS

        efficiency = getattr(self, "aud_efficiency", DEFAULTS)
        self.args.dataloader_prefetch_factor = (
            efficiency["prefetch_factor"] if self.args.dataloader_num_workers else None
        )
        loader = super().get_eval_dataloader(eval_dataset)
        base = getattr(loader, "base_dataloader", loader)
        if base.num_workers:
            base.timeout = efficiency["dataloader_timeout"]
        return loader

    def create_optimizer(self, model=None):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model if model is None else model
        decay_names = set(self.get_decay_parameter_names(model))
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("aud_decoder.model.embed_tokens."):
                rate = self._embed_lr_factor
            elif name.startswith("aud_decoder."):
                rate = self._llm_lr_factor
            elif name.startswith(("aud_encoder.", "aud_projector.", "aud_pool.", "aud_ctc_head.")):
                rate = 1.0
            else:
                raise ValueError(f"Unclassified trainable AUD parameter: {name}")
            groups.setdefault((rate, name in decay_names), []).append(parameter)
        cls, kwargs = self.get_optimizer_cls_and_kwargs(self.args, model)
        kwargs.pop("params", None)
        kwargs.pop("model", None)
        self.optimizer = cls(
            [
                {
                    "params": parameters,
                    "lr": self.args.learning_rate * rate,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                }
                for (rate, decay), parameters in sorted(groups.items())
            ],
            **kwargs,
        )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        directory = output_dir or self.args.output_dir
        self.prediction_processor.save_pretrained(directory)
        if hasattr(self, "aud_receipt"):
            from pathlib import Path

            from .export import write_json

            write_json(Path(directory) / "training_recipe.json", self.aud_receipt)
            write_json(
                Path(directory) / "provenance.json",
                {
                    "kind": "trained_aud_checkpoint",
                    "source_checkpoint": self.aud_source,
                    "global_step": self.state.global_step,
                    "stage": self.aud_receipt["stage"],
                    "quality_validated": False,
                },
            )
