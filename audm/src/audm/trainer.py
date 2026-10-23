"""QASR Trainer behavior with AUD optimizer groups and accumulation-window quotas."""

import math
import os

import torch
from qasr.sampling import WeightedStratifiedLanguageSampler
from qasr.train import PredictionLoggingTrainer, _SamplerEpochDataLoader


class AUDTrainer(PredictionLoggingTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        if os.environ.get("AUD_DEBUG_NUMERICS") == "1":
            # Enclose forward AND backward so the traceback identifies the
            # forward operation whose backward first generated NaNs.
            with torch.autograd.detect_anomaly(check_nan=True):
                return super().training_step(model, inputs, num_items_in_batch)
        return super().training_step(model, inputs, num_items_in_batch)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Training and loss-only evaluation do not need vocabulary logits for
        # the audio prompt, padding, or the entire supervised sequence at once.
        # Public model forward/generate keep the native Transformers behavior.
        if "labels" in inputs and (model.training or self.args.prediction_loss_only):
            inputs = dict(
                inputs,
                loss_only=True,
                use_cache=False,
                loss_smoothing=getattr(self, "aud_label_smoothing", 0.0),
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

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
        )

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
            elif name.startswith(("aud_encoder.", "aud_projector.", "aud_pool.")):
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
