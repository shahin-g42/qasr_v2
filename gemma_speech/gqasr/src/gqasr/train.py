"""Native DDP training from a GQASR base artifact and a prepared disk index."""

import json
import math
import os
import shutil
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .bootstrap import copy_assets
from .config import load_config
from .console import print_record
from .data import Collator, ManifestIndex, StratifiedUpdateSampler, data_contract
from .distributed import Runtime
from .gradients import clip_gradients
from .metrics import score_predictions
from .modeling import GQASRForSpeechRecognition
from .optimization import build_optimizer, training_scope
from .processing import GQASRProcessor
from .tracking import Tracker, training_config


def schedule(step, cfg):
    if step < cfg["warmup_steps"]:
        return (step + 1) / max(1, cfg["warmup_steps"])
    start = max(cfg["warmup_steps"], cfg["max_steps"] - cfg["decay_steps"])
    if step < start:
        return 1.0
    progress = min(1.0, (step - start) / max(1, cfg["max_steps"] - start))
    return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * (1 + math.cos(math.pi * progress)) / 2


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, dataset, collator, cfg, rt):
    model.eval()
    keys = []
    for language, count in sorted(dataset.counts.items()):
        generator = torch.Generator().manual_seed(cfg["data_seed"])
        # A bounded deterministic sample, without allocating a full-corpus permutation.
        target = min(count, cfg["eval_samples_per_language"])
        selected = set()
        while len(selected) < target:
            selected.add(int(torch.randint(1, count + 1, (1,), generator=generator)))
        keys.extend((language, i) for i in sorted(selected))
    keys = keys[rt.rank :: rt.world]
    records = []
    for start in range(0, len(keys), cfg["eval_batch_size"]):
        batch, rows = collator([dataset[k] for k in keys[start : start + cfg["eval_batch_size"]]])
        audio = {k: v for k, v in batch.items() if k not in {"targets", "target_mask"}}
        with rt.autocast():
            predicted = model.generate_transcripts(
                **move(audio, rt.device), max_new_tokens=cfg["eval_generation_max_new_tokens"]
            )
        records.extend(
            dict(
                id=row["id"],
                reference=row["text"],
                language=row["language"],
                dialect=row["dialect"],
                hypothesis=collator.processor.decode(ids),
            )
            for row, ids in zip(rows, predicted, strict=True)
        )
    records = [r for shard in rt.gather(records) for r in shard]
    model.train()
    return score_predictions(records), records


def save_checkpoint(model, optimizer, cfg, metadata, step, output, rng, final=False):
    name = "final" if final else f"checkpoint-{step:08d}"
    destination = output / name
    staging = Path(tempfile.mkdtemp(prefix=".gqasr-save-", dir=output))
    try:
        artifact = staging / "artifact"
        model.save_pretrained(artifact)
        copy_assets(cfg["base_model"], artifact)
        provenance = json.loads((artifact / "provenance.json").read_text())
        provenance["training"] = {
            "stage_output": str(output),
            "completed_optimizer_steps": step,
            "initialized_from": cfg["base_model"],
        }
        (artifact / "provenance.json").write_text(json.dumps(provenance, indent=2))
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": cfg,
                "index_metadata": metadata,
                "rng": rng,
            },
            artifact / "training_state.pt",
        )
        (artifact / "training_config.json").write_text(json.dumps(cfg, indent=2))
        os.rename(artifact, destination)
    finally:
        shutil.rmtree(staging)
    (output / "latest.json.tmp").write_text(json.dumps({"checkpoint": name, "step": step}))
    os.replace(output / "latest.json.tmp", output / "latest.json")
    print(f"[GQASR] Checkpoint saved at update {step:,}: {destination}", flush=True)
    for old in sorted(output.glob("checkpoint-*"))[: -cfg["save_total_limit"]]:
        shutil.rmtree(old)


def train(config, resume=None):
    cfg = load_config(config)
    rt = Runtime(cfg["precision"])
    tracker = Tracker(cfg, main=rt.main, resume=resume)
    exit_code = 1
    try:
        _train(cfg, rt, resume, tracker)
        exit_code = 0
    finally:
        tracker.finish(exit_code)
        rt.close()


def _train(cfg, rt, resume, tracker=None):
    if rt.world != cfg["expected_world_size"]:
        raise ValueError(f"config expects {cfg['expected_world_size']} ranks; launched {rt.world}")
    torch.manual_seed(cfg["seed"] + rt.rank)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    if rt.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
    dataset = ManifestIndex(cfg["index_path"], "train")
    validation = ManifestIndex(cfg["index_path"], "eval")
    if dataset.metadata["data_contract"] != data_contract(cfg):
        raise ValueError("index/config data contract mismatch; rebuild the index")
    processor = GQASRProcessor(cfg["base_model"])
    import hashlib

    tokenizer_hash = hashlib.sha256(
        processor.tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest()
    if dataset.metadata["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("tokenizer differs from the preflight index")
    model = GQASRForSpeechRecognition.from_pretrained(resume or cfg["base_model"])
    model.configure_training(
        cfg["acoustic_layers"],
        cfg["decoder_layers"],
        cfg["train_embeddings"],
        cfg["gradient_checkpointing"],
        cfg["label_smoothing"],
        train_acoustic_frontend=cfg.get("train_acoustic_frontend"),
    )
    model.to(rt.device)
    optimizer = build_optimizer(model, cfg)
    step = 0
    if resume:
        saved = torch.load(
            Path(resume) / "training_state.pt", map_location="cpu", weights_only=True
        )

        def comparable(c):
            return training_config(c)

        if (
            comparable(saved["config"]) != comparable(cfg)
            or saved["index_metadata"] != dataset.metadata
        ):
            raise ValueError(
                "resume requires identical config/index; only output_dir/workers/logging settings may change"
            )
        group_names = [group["name"] for group in optimizer.param_groups]
        optimizer.load_state_dict(saved["optimizer"])
        for group, name in zip(optimizer.param_groups, group_names, strict=True):
            group["name"] = name
        step = saved["step"]
        torch.set_rng_state(saved["rng"][rt.rank]["cpu"])
        if rt.device.type == "cuda":
            torch.cuda.set_rng_state(saved["rng"][rt.rank]["cuda"], rt.device)
    output = Path(cfg["output_dir"])
    if rt.main:
        output.mkdir(parents=True, exist_ok=False)
        (output / "config.json").write_text(json.dumps(cfg, indent=2))
        print(
            json.dumps(
                {
                    **training_scope(model, optimizer),
                    "world_size": rt.world,
                    "global_batch": cfg["batch_size"] * cfg["accumulation_steps"] * rt.world,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            ),
            flush=True,
        )
    rt.barrier()
    if cfg.get("eval_at_start", False) and not resume:
        if rt.main:
            print("[GQASR] Initial validation before any optimizer update", flush=True)
        scores, predictions = evaluate(model, validation, Collator(processor, cfg), cfg, rt)
        if rt.main:
            record = {"event": "initial_validation", "step": 0, "elapsed_seconds": 0.0,
                      "validation": scores}
            (output / "predictions-0.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
            )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print_record(record, cfg)
            if tracker is not None:
                tracker.log(record)
    wrapped = (
        DistributedDataParallel(
            model,
            device_ids=[rt.local_rank] if rt.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if rt.world > 1
        else model
    )
    sampler = StratifiedUpdateSampler(
        dataset.counts,
        cfg["language_batch_counts"],
        cfg["batch_size"],
        cfg["max_steps"],
        rt.rank,
        rt.world,
        cfg.get("training_data_seed", cfg["data_seed"]),
        step,
        augmentation_seeds=True,
    )
    collator = Collator(processor, cfg, training=True)
    eval_collator = Collator(processor, cfg)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collator,
        num_workers=cfg["workers"],
        multiprocessing_context="spawn" if cfg["workers"] else None,
        pin_memory=rt.device.type == "cuda",
        persistent_workers=cfg["workers"] > 0,
        generator=torch.Generator().manual_seed(cfg.get("training_data_seed", cfg["data_seed"])),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    for microbatch, (batch, rows) in enumerate(loader):
        sync = (microbatch + 1) % cfg["accumulation_steps"] == 0
        context = wrapped.no_sync() if rt.world > 1 and not sync else nullcontext()
        with context:
            with rt.autocast():
                result = wrapped(**move(batch, rt.device))
                loss = result["loss"] / cfg["accumulation_steps"]
            if not rt.all_finite(loss):
                raise FloatingPointError("nonfinite GQASR loss")
            loss.backward()
        if not sync:
            continue
        gradient_norm = clip_gradients(model, cfg["grad_clip"], rt, step)
        factor = schedule(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        record = {
            "step": step,
            "rank0_last_microbatch_loss": float(result["loss"].detach()),
            "rank0_gradient_norm": float(gradient_norm),
            "learning_rates": {g.get("name", str(i)): g["lr"] for i, g in enumerate(optimizer.param_groups)},
            "elapsed_seconds": time.monotonic() - started,
            "rank0_last_microbatch_augmentation": [r.get("augmentation_audit", {}) for r in rows],
        }
        if step % cfg["eval_steps"] == 0 or step == cfg["max_steps"]:
            if rt.main:
                print(f"[GQASR] Validation started at update {step:,}", flush=True)
            scores, predictions = evaluate(model, validation, eval_collator, cfg, rt)
            record["validation"] = scores
            if rt.main:
                (output / f"predictions-{step}.jsonl").write_text(
                    "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
                )
        if rt.main and (step == 1 or step % cfg["logging_steps"] == 0 or "validation" in record):
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print_record(record, cfg)
            if tracker is not None:
                tracker.log(record)
        if step % cfg["save_steps"] == 0 or step == cfg["max_steps"]:
            rng = {"cpu": torch.get_rng_state()}
            if rt.device.type == "cuda":
                rng["cuda"] = torch.cuda.get_rng_state(rt.device)
            rngs = rt.gather(rng)
            if rt.main:
                save_checkpoint(
                    model,
                    optimizer,
                    cfg,
                    dataset.metadata,
                    step,
                    output,
                    rngs,
                    final=step == cfg["max_steps"],
                )
            rt.barrier()
