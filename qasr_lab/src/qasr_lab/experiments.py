"""Small, single-device pilots with their own loop and fresh output directories."""

from __future__ import annotations

import copy
import math
import os
import time
from collections import Counter
from pathlib import Path

from .checkpoints import checkpoint_identity, use_qasr_source
from .io import environment, fresh_directory, positive_number, read_json, write_json
from .panel import identity_keys, load_panel, validate_samples

DEFAULTS = {
    "seed": 42,
    "device": "cpu",
    "precision": "float32",
    "max_steps": 200,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "encoder_lr": 1e-5,
    "projector_lr": 1e-4,
    "decoder_lr": 1e-6,
    "weight_decay": 0.01,
    "warmup_steps": 20,
    "max_grad_norm": 1.0,
    "max_audio_seconds": 35.0,
    "min_audio_seconds": 0.01,
    "max_target_tokens": 512,
    "trainable": "projector",
    "upper_layers": 4,
    "gradient_checkpointing": True,
    "bridge_dim": 0,
    "mask_pool_padding": False,
    "ctc_weight": 0.0,
    "ctc_max_vocabulary": 16384,
    "min_ctc_feasible_fraction": 0.95,
    "log_every": 10,
}
PATH_KEYS = ("train_panel", "eval_panel", "non_speech_panel", "qasr_source")
CONFIG_KEYS = (
    set(DEFAULTS)
    | set(PATH_KEYS)
    | {
        "id",
        "checkpoint",
        "revision",
        "output",
        "comparison_baseline",
    }
)


def load_experiment(path):
    path = Path(path).resolve()
    config = read_json(path)
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    config = {**DEFAULTS, **config}
    for key in ("id", "checkpoint", "output", "train_panel", "eval_panel", "qasr_source"):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError(f"{key} is required")
    for key in (*PATH_KEYS, "output"):
        if config.get(key):
            config[key] = str((path.parent / Path(config[key]).expanduser()).resolve())
    checkpoint = Path(config["checkpoint"]).expanduser()
    if (
        checkpoint.is_absolute()
        or config["checkpoint"].startswith(".")
        or (path.parent / checkpoint).exists()
    ):
        config["checkpoint"] = str((path.parent / checkpoint).resolve())
    for key in (
        "max_steps",
        "batch_size",
        "gradient_accumulation_steps",
        "upper_layers",
        "max_target_tokens",
        "ctc_max_vocabulary",
        "log_every",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("warmup_steps", "bridge_dim", "seed"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if config["warmup_steps"] >= config["max_steps"]:
        raise ValueError("warmup_steps must be less than max_steps")
    for key in ("encoder_lr", "projector_lr", "decoder_lr", "max_grad_norm", "max_audio_seconds"):
        positive_number(config[key], key)
    for key in ("weight_decay", "ctc_weight", "min_audio_seconds"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], (float, int))
            or not math.isfinite(config[key])
            or config[key] < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
    if config["max_audio_seconds"] <= config["min_audio_seconds"]:
        raise ValueError("max_audio_seconds must exceed min_audio_seconds")
    for key in ("gradient_checkpointing", "mask_pool_padding"):
        if not isinstance(config[key], bool):
            raise ValueError(f"{key} must be boolean")
    fraction = positive_number(config["min_ctc_feasible_fraction"], "min_ctc_feasible_fraction")
    if fraction > 1:
        raise ValueError("min_ctc_feasible_fraction must not exceed one")
    if config["precision"] not in ("float32", "bfloat16"):
        raise ValueError("Pilots support float32 or bfloat16 autocast with float32 parameters")
    if config["trainable"] not in ("projector", "upper_encoder", "full"):
        raise ValueError("Unknown trainable mode")
    if config["ctc_weight"] and config["trainable"] == "projector":
        raise ValueError("CTC needs a trainable encoder; choose upper_encoder or full")
    return config


def load_experiment_panels(config):
    train_meta, train = load_panel(config["train_panel"])
    eval_meta, evaluation = load_panel(config["eval_panel"])
    provenance = {"train": train_meta, "evaluation": eval_meta}
    if config.get("non_speech_panel"):
        silence_meta, silence = load_panel(config["non_speech_panel"])
        if any(row["text"].strip() for row in silence):
            raise ValueError("The non-speech panel must contain only empty transcripts")
        train = [*train, *silence]
        provenance["non_speech"] = silence_meta
    validate_samples(train)
    if len(train) < config["batch_size"]:
        raise ValueError("Training panel must contain at least one full batch")
    training_keys = set().union(*(identity_keys(row, "/") for row in train))
    for row in evaluation:
        if training_keys.intersection(identity_keys(row, "/")):
            raise ValueError(f"Train/evaluation identity overlap: {row['id']}")
    provenance["effective_training_samples"] = len(train)
    provenance["training_language_counts"] = dict(Counter(row["language"] for row in train))
    provenance["training_non_speech_samples"] = sum(not row["text"].strip() for row in train)
    return train, evaluation, provenance


def create_plan(base_config, output, *, non_speech_panel=None):
    base = load_experiment(base_config)
    # A controlled suite starts with a vanilla checkpoint and projector-only control.
    if (
        base["bridge_dim"]
        or base["ctc_weight"]
        or base["mask_pool_padding"]
        or base.get("non_speech_panel")
    ):
        raise ValueError("The base plan configuration must have all lab extensions disabled")
    if base["trainable"] != "projector":
        raise ValueError("The plan control must use trainable=projector")
    variants = {
        "control": {},
        "upper_encoder": {"trainable": "upper_encoder"},
        "residual_bridge": {"bridge_dim": 512},
        "ctc": {
            "trainable": "upper_encoder",
            "ctc_weight": 0.1,
            "comparison_baseline": "upper_encoder",
        },
    }
    if non_speech_panel:
        variants["non_speech"] = {"non_speech_panel": str(Path(non_speech_panel).resolve())}
    out = fresh_directory(output)
    entries = []
    for name, overrides in variants.items():
        config = {
            **copy.deepcopy(base),
            "id": name,
            "comparison_baseline": "control",
            **overrides,
            "output": str(out / "runs" / name),
        }
        config_path = out / f"{name}.json"
        write_json(config_path, config)
        entries.append(
            {
                "id": name,
                "config": str(config_path),
                "comparison_baseline": config["comparison_baseline"],
                "changed_settings": overrides,
            }
        )
    plan = {
        "schema": "qasr-lab-experiment-plan-v1",
        "experiments": entries,
        "fixed": [
            "initial checkpoint",
            "seed",
            "evaluation panel",
            "optimizer steps",
            "batch size",
            "gradient accumulation",
            "learning rates",
            "scheduler",
        ],
        "notes": [
            "CTC is compared with upper_encoder because CTC needs encoder gradients.",
            "The non-speech variant changes the data mixture, not the optimizer-step budget.",
            "Masked pooling can be tested separately only when the checkpoint pool stride exceeds one.",
            "Train/evaluation CE is diagnostic. Rank checkpoints using generated held-out transcripts.",
            "Repeat promising comparisons with multiple seeds; these pilots do not establish improvements.",
        ],
    }
    write_json(out / "plan.json", plan)
    return plan


def optimizer_groups(model, config):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("model.audio_tower."):
            component = "encoder"
        elif name.startswith(
            ("model.multi_modal_projector.", "model.pooling_conv.", "lab_ctc_head.")
        ):
            component = "projector"
        else:
            component = "decoder"
        decay = parameter.ndim >= 2 and not name.endswith(".bias")
        group = groups.setdefault(
            (component, decay),
            {
                "params": [],
                "lr": config[f"{component}_lr"],
                "weight_decay": config["weight_decay"] if decay else 0,
            },
        )
        group["params"].append(parameter)
    return list(groups.values())


def run_experiment(config_path):
    import json
    from contextlib import nullcontext

    import torch
    from torch.utils.data import DataLoader
    from transformers import set_seed

    config = load_experiment(config_path)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The lab pilot runner is single-process; do not launch with torchrun")
    device = torch.device(config["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Pilots require cpu or a single cuda device")
    train, evaluation, panels = load_experiment_panels(config)
    identity = checkpoint_identity(config["checkpoint"], revision=config.get("revision"))
    source = use_qasr_source(config["qasr_source"])
    from qasr import QASRConfig, QASRProcessor
    from qasr.collator import QASRDataCollator

    from .modeling import (
        LabQASRForConditionalGeneration,
        ResidualBridge,
        compact_targets,
        configure_trainable,
        transcript_vocabulary,
    )

    out = fresh_directory(config["output"])
    write_json(out / "experiment.json", config)
    write_json(
        out / "provenance.json",
        {
            "environment": environment(),
            "checkpoint": identity,
            "panels": panels,
            "qasr_source": source,
            "effective_batch_size": config["batch_size"] * config["gradient_accumulation_steps"],
            "scope": "Single-device pilot; no production trainer, distributed launcher or optimizer state is reused.",
        },
    )
    try:
        set_seed(config["seed"])
        common = {"revision": config.get("revision"), "trust_remote_code": False}
        original_config, _ = QASRConfig.get_config_dict(config["checkpoint"], **common)
        if original_config.get("model_type") != "qasr":
            raise ValueError("Pilot initialization requires a vanilla QASR checkpoint")
        model_config = QASRConfig.from_pretrained(config["checkpoint"], **common)
        if (
            getattr(model_config, "lab_bridge_dim", 0)
            or getattr(model_config, "lab_ctc_vocabulary", [])
            or getattr(model_config, "lab_mask_pool_padding", False)
        ):
            raise ValueError(
                "Start pilots from the same vanilla QASR checkpoint, not a previous lab variant"
            )
        processor = QASRProcessor.from_pretrained(config["checkpoint"], **common)
        if processor.projector_pool_stride != model_config.projector_pool_stride:
            raise ValueError("Processor/model pooling stride mismatch")
        if config["mask_pool_padding"] and model_config.projector_pool_stride == 1:
            raise ValueError("Masked pooling has no effect when projector_pool_stride=1")
        model_config.lab_mask_pool_padding = config["mask_pool_padding"]
        model_config.lab_bridge_dim = config["bridge_dim"]
        model_config.lab_ctc_weight = config["ctc_weight"]
        model_config.lab_ctc_vocabulary = (
            transcript_vocabulary(processor.tokenizer, train, maximum=config["ctc_max_vocabulary"])
            if config["ctc_weight"]
            else []
        )
        model = LabQASRForConditionalGeneration.from_pretrained(
            config["checkpoint"],
            config=model_config,
            dtype=torch.float32,
            attn_implementation="sdpa",
            **common,
        ).to(device)
        if isinstance(model.model.multi_modal_projector, ResidualBridge):
            # HF initializes missing checkpoint tensors; explicitly restore identity initialization.
            model.model.multi_modal_projector.reset_residual()
        counts = configure_trainable(
            model, config["trainable"], upper_layers=config["upper_layers"]
        )
        write_json(out / "parameters.json", counts)
        model.config.use_cache = False
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable({"use_reentrant": False})
        base_collator = QASRDataCollator(
            processor=processor,
            language="en",
            min_audio_seconds=config["min_audio_seconds"],
            max_audio_seconds=config["max_audio_seconds"],
            max_target_length=config["max_target_tokens"],
        )

        def collate(rows, with_ctc=True):
            features = [{**row, "audio_path": row["audio_filepath"]} for row in rows]
            batch = base_collator(features)
            if with_ctc and model_config.lab_ctc_vocabulary:
                batch["lab_ctc_targets"] = compact_targets(
                    processor.tokenizer,
                    [row["text"] for row in rows],
                    model_config.lab_ctc_vocabulary,
                )
            return {key: value.to(device) for key, value in batch.items()}

        def autocast():
            return (
                torch.autocast(device.type, dtype=torch.bfloat16)
                if config["precision"] == "bfloat16"
                else nullcontext()
            )

        def eval_ce():
            model.eval()
            total, tokens = 0.0, 0
            with torch.inference_mode():
                for offset in range(0, len(evaluation), config["batch_size"]):
                    batch = collate(
                        evaluation[offset : offset + config["batch_size"]], with_ctc=False
                    )
                    count = int((batch["labels"][:, 1:] != -100).sum())
                    with autocast():
                        loss = model(**batch, use_cache=False).loss
                    total += float(loss) * count
                    tokens += count
            if not tokens or not math.isfinite(total):
                raise ValueError("No finite evaluation CE")
            return {"cross_entropy": total / tokens, "supervised_tokens": tokens}

        initial_eval = eval_ce()
        # Reset data/dropout seeds after validation so its length cannot change training randomness.
        set_seed(config["seed"])
        loader = DataLoader(
            train,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True,
            generator=torch.Generator().manual_seed(config["seed"]),
            collate_fn=collate,
        )
        iterator = iter(loader)
        optimizer = torch.optim.AdamW(optimizer_groups(model, config))

        def schedule(step):
            warmup = config["warmup_steps"]
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, config["max_steps"] - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        model.train()
        # Frozen Conformer BatchNorm buffers must not drift during projector-only training.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                if not any(p.requires_grad for p in module.parameters()):
                    module.eval()
        totals = Counter()
        started = time.perf_counter()
        with (out / "training.jsonl").open("x", encoding="utf-8") as journal:
            for step in range(1, config["max_steps"] + 1):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(config["gradient_accumulation_steps"]):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        iterator = iter(loader)
                        batch = next(iterator)
                    with autocast():
                        loss = model(**batch, use_cache=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss / config["gradient_accumulation_steps"]).backward()
                    loss_sum += float(loss.detach())
                    totals.update(
                        {
                            key: value
                            for key, value in model.lab_ctc_stats.items()
                            if key != "ctc_loss"
                        }
                    )
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config["max_grad_norm"],
                    error_if_nonfinite=True,
                )
                optimizer.step()
                scheduler.step()
                if step == 1 or step % config["log_every"] == 0 or step == config["max_steps"]:
                    event = {
                        "step": step,
                        "loss": loss_sum / config["gradient_accumulation_steps"],
                        "elapsed_seconds": time.perf_counter() - started,
                        "ctc": dict(totals),
                    }
                    journal.write(json.dumps(event, allow_nan=False) + "\n")
                    journal.flush()
                    print(json.dumps(event), flush=True)
        final_eval = eval_ce()
        coverage = totals["ctc_feasible_rows"] / totals["ctc_rows"] if totals["ctc_rows"] else None
        if coverage is not None and coverage < config["min_ctc_feasible_fraction"]:
            raise ValueError(
                f"CTC feasible fraction {coverage:.4f} is below the configured acceptance threshold"
            )
        model.config.use_cache = True
        model.save_pretrained(out / "checkpoint", safe_serialization=True)
        processor.save_pretrained(out / "checkpoint")
        summary = {
            "status": "complete",
            "initial_eval": initial_eval,
            "final_eval": final_eval,
            "ctc": dict(totals),
            "ctc_feasible_fraction": coverage,
            "parameters": counts,
            "steps": config["max_steps"],
            "checkpoint": str(out / "checkpoint"),
            "next_step": "Generate transcripts on a separate held-out panel using backend=qasr-lab.",
        }
        write_json(out / "result.json", summary)
        return summary
    except Exception as exc:
        write_json(
            out / "failure.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
