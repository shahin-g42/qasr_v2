"""Rank-zero summaries; detailed metrics still reach Trainer, W&B and disk."""

import json
import logging
import time
from pathlib import Path

from transformers import TrainerCallback

from .console import emit

LOG = logging.getLogger("audm.metrics")


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else f"{minutes:d}m {seconds:02d}s"


class ConsoleCallback(TrainerCallback):
    def __init__(self, variant, stage):
        self.variant, self.stage = variant, stage
        self.average_seconds = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()
        self.start_step = state.global_step
        batch = args.per_device_train_batch_size
        accum = args.gradient_accumulation_steps
        emit(
            f"{self.variant.upper()} | {self.stage} | {state.max_steps:,} updates | "
            f"batch {batch} x {accum} x {args.world_size} = {batch * accum * args.world_size} | "
            f"eval/WER batch {args.per_device_eval_batch_size}/GPU"
        )
        emit(f"Output: {args.output_dir}")

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.monotonic() - self.step_started
        elapsed += getattr(kwargs.get("model"), "aud_data_wait_seconds", 0.0)
        self.average_seconds = (
            elapsed if self.average_seconds is None else 0.8 * self.average_seconds + 0.2 * elapsed
        )
        if state.global_step == self.start_step + 1:
            emit(f"First update complete | {duration(elapsed)} (includes startup/warm-up)")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if args.process_index != 0 or not logs:
            return
        LOG.info("step=%s %s", state.global_step, json.dumps(logs, default=str))
        if "loss" in logs:
            parts = [
                f"Train {state.global_step:,}/{state.max_steps:,} "
                f"({100 * state.global_step / max(1, state.max_steps):.1f}%)",
                f"loss {logs['loss']:.4f}",
            ]
            for key, label, spec in (("grad_norm", "grad", ".2f"), ("learning_rate", "lr", ".2e")):
                if key in logs:
                    parts.append(f"{label} {logs[key]:{spec}}")
            if self.average_seconds:
                parts.append(f"{self.average_seconds:.2f}s/update")
                parts.append(
                    f"train ETA ~{duration(self.average_seconds * (state.max_steps - state.global_step))}"
                )
            emit(" | ".join(parts))
        if "eval_loss" in logs:
            emit(
                f"Eval step {state.global_step:,} | loss {logs['eval_loss']:.4f} | "
                f"{logs.get('eval_runtime', 0):.1f}s"
            )
        languages = sorted(
            k.removeprefix("eval_wer_")
            for k in logs
            if k.startswith("eval_wer_") and k != "eval_wer_macro"
        )
        for language in languages:
            parts = [f"  {language.upper():<3} | WER {logs[f'eval_wer_{language}'] * 100:.2f}%"]
            if f"eval_cer_{language}" in logs:
                parts.append(f"CER {logs[f'eval_cer_{language}'] * 100:.2f}%")
            emit(" | ".join(parts))
        if "train_runtime" in logs:
            emit(
                f"Training finished | {duration(logs['train_runtime'])} | final evaluation/save follows"
            )

    def on_save(self, args, state, control, **kwargs):
        emit(f"Checkpoint saved | {Path(args.output_dir) / f'checkpoint-{state.global_step}'}")
