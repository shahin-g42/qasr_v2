"""DDP training of the speaker adapter on cached Gemma/Qwen pairs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from gemma_speech.data import WeightedDistributedSampler, sampling_weights
from gemma_speech.distributed import Runtime
from gemma_speech.train import learning_rate_factor

from .speaker import SpeakerAdapter, collate_cached, distillation_loss, read_cache


@torch.no_grad()
def evaluate(adapter, rows, rt, batch_size):
    adapter.eval()
    stats = {}
    local = rows[rt.rank :: rt.world]
    for start in range(0, len(local), batch_size):
        chosen = local[start : start + batch_size]
        frames, mask, target = [v.to(rt.device) for v in collate_cached(chosen)]
        with rt.autocast():
            predictions = adapter(frames, mask)
        for row, predicted, teacher in zip(chosen, predictions, target, strict=True):
            loss, cosine = distillation_loss(predicted[None], teacher[None])
            values = stats.setdefault(row["language"], [0.0, 0.0, 0])
            values[0] += float(loss)
            values[1] += float(cosine)
            values[2] += 1
    all_stats = {}
    for shard in rt.gather(stats):
        for language, values in shard.items():
            total = all_stats.setdefault(language, [0.0, 0.0, 0])
            for i, value in enumerate(values):
                total[i] += value
    adapter.train()
    return {
        k: {"loss": v[0] / v[2], "cosine_distance": v[1] / v[2], "utterances": v[2]}
        for k, v in all_stats.items()
    }


def train_speaker(config, resume=None):
    path = Path(config).resolve()
    cfg = json.loads(path.read_text())
    allowed = {
        "train_cache",
        "validation_cache",
        "output_dir",
        "width",
        "batch_size",
        "max_steps",
        "learning_rate",
        "precision",
        "workers",
        "seed",
        "eval_every",
        "language_weights",
        "holdout_speakers",
    }
    if cfg.keys() - allowed:
        raise ValueError(f"unknown speaker settings: {cfg.keys() - allowed}")
    for key in ("train_cache", "validation_cache", "output_dir"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    defaults = dict(
        width=512,
        batch_size=32,
        max_steps=5000,
        learning_rate=3e-4,
        precision="bf16",
        workers=4,
        seed=42,
        eval_every=250,
        language_weights={},
        holdout_speakers=True,
    )
    cfg = defaults | cfg
    if (
        cfg["precision"] not in {"fp32", "bf16"}
        or min(cfg[k] for k in ("width", "batch_size", "max_steps", "eval_every", "learning_rate"))
        <= 0
    ):
        raise ValueError("invalid training settings")
    rt = Runtime(cfg["precision"])
    try:
        _train(cfg, rt, resume)
    finally:
        rt.close()


def _train(cfg, rt, resume):
    torch.manual_seed(cfg["seed"] + rt.rank)
    rows, metadata = read_cache(cfg["train_cache"])
    validation, val_metadata = read_cache(cfg["validation_cache"])
    if metadata != val_metadata:
        raise ValueError("train and validation cache identities differ")
    for key in ("id", "audio_sha256"):
        if {r[key] for r in rows} & {r[key] for r in validation}:
            raise ValueError(f"speaker train/validation leakage in {key}")
    if cfg["holdout_speakers"] and {r["speaker"] for r in rows} & {
        r["speaker"] for r in validation
    }:
        raise ValueError("speaker adapter validation must hold out speakers")
    if len({r["language"] for r in rows}) < 2:
        raise ValueError("multilingual speaker training requires at least two languages")
    first = torch.load(rows[0]["file"], map_location="cpu", weights_only=True)
    adapter = SpeakerAdapter(first["frames"].shape[-1], cfg["width"], first["teacher"].numel()).to(
        rt.device
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg["learning_rate"])
    step, epoch, offset = 0, 0, 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)

        def comparable(c):
            return {k: v for k, v in c.items() if k not in {"output_dir", "workers"}}

        if (
            comparable(saved["config"]) != comparable(cfg)
            or saved["world"] != rt.world
            or saved["metadata"] != metadata
        ):
            raise ValueError("resume requires the same config, caches and world size")
        adapter.load_state_dict(saved["adapter"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        step, epoch, offset = saved["step"], saved["epoch"], saved["offset"]
    output = Path(cfg["output_dir"])
    if rt.main:
        output.mkdir(parents=True, exist_ok=False)
    rt.barrier()
    model = (
        DistributedDataParallel(
            adapter, device_ids=[rt.local_rank] if rt.device.type == "cuda" else None
        )
        if rt.world > 1
        else adapter
    )
    weights = sampling_weights(
        [SimpleNamespace(language=r["language"]) for r in rows], cfg["language_weights"]
    )
    sampler = WeightedDistributedSampler(
        weights, len(rows), cfg["batch_size"], 1, rt.rank, rt.world, cfg["seed"]
    )
    loader = DataLoader(
        rows,
        batch_size=cfg["batch_size"],
        sampler=sampler,
        collate_fn=collate_cached,
        num_workers=cfg["workers"],
    )
    while step < cfg["max_steps"]:
        sampler.epoch, sampler.offset = epoch, offset * cfg["batch_size"]
        for batch_index, batch in enumerate(loader, offset):
            frames, mask, teacher = [v.to(rt.device) for v in batch]
            optimizer.zero_grad(set_to_none=True)
            with rt.autocast():
                loss, _ = distillation_loss(model(frames, mask), teacher)
            if not rt.all_finite(loss):
                raise FloatingPointError("nonfinite speaker loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            if not rt.all_finite(norm):
                raise FloatingPointError("nonfinite speaker gradients")
            optimizer.param_groups[0]["lr"] = cfg["learning_rate"] * learning_rate_factor(
                step, min(100, cfg["max_steps"] // 10), cfg["max_steps"]
            )
            optimizer.step()
            step += 1
            offset = batch_index + 1
            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"]:
                scores = evaluate(adapter, validation, rt, cfg["batch_size"])
                if rt.main:
                    report = {"step": step, "validation": scores}
                    print(json.dumps(report), flush=True)
                    with (output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(report) + "\n")
                    saved = {
                        "format": "gemma-qwen-speaker-v1",
                        "dimensions": adapter.dimensions,
                        "adapter": adapter.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metadata": metadata,
                        "config": cfg,
                        "world": rt.world,
                        "step": step,
                        "epoch": epoch,
                        "offset": offset,
                    }
                    torch.save(saved, output / "last.tmp")
                    os.replace(output / "last.tmp", output / "last.pt")
            if step >= cfg["max_steps"]:
                break
        epoch += 1
        offset = 0
