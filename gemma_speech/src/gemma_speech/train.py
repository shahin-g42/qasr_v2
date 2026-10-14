"""Single-node and multi-node DDP training with global language sampling."""

import hashlib
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .asr import greedy_tokens
from .assets import load_assets, processors_from_assets
from .backbone import load_component
from .batching import Collator, move_batch
from .checkpoint import build_model, load_checkpoint, save_checkpoint
from .config import TrainConfig
from .data import WeightedDistributedSampler, check_disjoint, read_manifest, sampling_weights
from .distributed import Runtime
from .metrics import score_predictions
from .text import SpeechTokenizer


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def learning_rate_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.1 + 0.9 * (1 + math.cos(math.pi * min(progress, 1))) / 2


@torch.no_grad()
def evaluate(model, rows, collator, runtime, batch_size):
    model.eval()
    local_rows = rows[runtime.rank :: runtime.world]
    predictions, losses = [], []
    # Use the unwrapped model: unequal validation shard sizes must never trigger DDP collectives.
    with torch.random.fork_rng(
        devices=[runtime.local_rank] if runtime.device.type == "cuda" else []
    ):
        torch.manual_seed(1729 + runtime.rank)
        for start in range(0, len(local_rows), batch_size):
            batch, selected = collator(local_rows[start : start + batch_size])
            with runtime.autocast():
                output = model(**move_batch(batch, runtime.device))
            losses.append((float(output["loss"]), len(selected)))
            if model.config.task == "asr":
                if model.config.asr_decoder == "qwen":
                    audio_batch = {
                        k: v for k, v in batch.items() if k not in {"targets", "target_mask"}
                    }
                    with runtime.autocast():
                        decoded = model.generate_transcripts(
                            **move_batch(audio_batch, runtime.device),
                            max_new_tokens=collator.config.max_text_tokens,
                        )
                else:
                    decoded = greedy_tokens(output["logits"], output["mask"])
                predictions.extend(
                    {
                        "id": row.id,
                        "language": row.language,
                        "dialect": row.dialect,
                        "reference": row.text,
                        "hypothesis": collator.tokenizer.decode(ids),
                    }
                    for row, ids in zip(selected, decoded, strict=True)
                )
    gathered = runtime.gather({"predictions": predictions, "losses": losses})
    all_predictions = [p for shard in gathered for p in shard["predictions"]]
    all_losses = [p for shard in gathered for p in shard["losses"]]
    result = {"loss": sum(loss * n for loss, n in all_losses) / sum(n for _, n in all_losses)}
    if all_predictions:
        result["scores"] = score_predictions(all_predictions)
    model.train()
    return result, all_predictions


def train(config_path, init_checkpoint=None, resume=None):
    cfg = TrainConfig.load(config_path)
    runtime = Runtime(cfg.precision)
    try:
        _train(cfg, runtime, init_checkpoint, resume)
    finally:
        runtime.close()


def _train(cfg, rt, init_checkpoint, resume):
    if init_checkpoint and resume:
        raise ValueError("use either init-checkpoint or resume")
    torch.manual_seed(cfg.seed + rt.rank)
    torch.set_num_threads(max(1, int(__import__("os").environ.get("OMP_NUM_THREADS", "1"))))
    train_rows = read_manifest(cfg.train_manifest, cfg.model.task)
    val_rows = read_manifest(cfg.validation_manifest, cfg.model.task)
    check_disjoint(train_rows, val_rows)
    languages = {v: i for i, v in enumerate(sorted({r.language for r in train_rows}))}
    speakers = {v: i for i, v in enumerate(sorted({r.speaker for r in train_rows}))}
    if cfg.require_multilingual and len(languages) < 2:
        raise ValueError("require_multilingual=true needs at least two training languages")
    if not {r.language for r in val_rows} <= languages.keys():
        raise ValueError("validation contains a language absent from training")
    if cfg.model.task == "tts" and not {r.speaker for r in val_rows} <= speakers.keys():
        raise ValueError(
            "this TTS implementation evaluates known speakers; hold out recordings, not voices"
        )
    weights = sampling_weights(train_rows, cfg.language_weights)
    metadata = {
        "world_size": rt.world,
        "torch": str(torch.__version__),
        "train_sha256": file_hash(cfg.train_manifest),
        "validation_sha256": file_hash(cfg.validation_manifest),
        "effective_batch": cfg.batch_size * cfg.accumulation_steps * rt.world,
    }
    is_qwen = cfg.model.task == "asr" and cfg.model.asr_decoder == "qwen"
    tokenizer = None if is_qwen else SpeechTokenizer(cfg.tokenizer)
    payload = None
    if init_checkpoint or resume:
        model, saved_tokenizer, payload = load_checkpoint(
            init_checkpoint or resume, model_override=cfg.model
        )
        if tokenizer is not None and saved_tokenizer.serialized != tokenizer.serialized:
            raise ValueError("checkpoint tokenizer differs from configured tokenizer")
        tokenizer = saved_tokenizer
        if payload["languages"] != languages or (
            cfg.model.task == "tts" and payload["speakers"] != speakers
        ):
            raise ValueError("checkpoint language/speaker mappings differ from training data")
        allowed = (
            {"train_backbone_layers", "qwen_train_layers", "local_files_only"}
            if init_checkpoint
            else {"local_files_only"}
        )
        before, after = dict(payload["config"]["model"]), cfg.to_dict()["model"]
        if {k: v for k, v in before.items() if k not in allowed} != {
            k: v for k, v in after.items() if k not in allowed
        }:
            raise ValueError(
                "warm start may only change backbone freezing, not model architecture or identity"
            )
        assets = payload["assets"]
    else:
        if is_qwen:
            from .qwen_asr import QwenTokenizer

            tokenizer = QwenTokenizer.from_pretrained(cfg.model)
        component = (
            load_component(cfg.model)
            if cfg.model.task == "asr" or cfg.model.use_semantic_text
            else None
        )
        assets = load_assets(cfg.model)
        model = build_model(cfg.model, component, tokenizer, speakers, languages)
    model.to(rt.device)
    extractor, hf_tokenizer = processors_from_assets(assets)
    collator = Collator(cfg, tokenizer, extractor, hf_tokenizer, speakers, languages)
    # Materialize at least one real batch before creating run artifacts.
    collator(train_rows[: cfg.batch_size])
    collator(val_rows[: cfg.batch_size])
    backbone_params, head_params = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (backbone_params if name.startswith(("backbone.", "decoder.")) else head_params).append(
                parameter
            )
    groups = [{"params": head_params, "lr": cfg.learning_rate, "base_lr": cfg.learning_rate}]
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": cfg.backbone_learning_rate,
                "base_lr": cfg.backbone_learning_rate,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    step, epoch, batch_offset = 0, 0, 0
    if resume:
        if payload["run_metadata"] != metadata:
            raise ValueError(
                "resume requires identical manifests, world size, PyTorch and effective batch"
            )
        old_config, new_config = dict(payload["config"]), cfg.to_dict()
        for key in ("output_dir", "workers"):
            old_config.pop(key, None)
            new_config.pop(key, None)
        if old_config != new_config:
            raise ValueError(
                "resume requires the original config except output_dir/workers; use init-checkpoint for a new stage"
            )
        optimizer.load_state_dict(payload["optimizer"])
        state = payload["training_state"]
        step, epoch, batch_offset = payload["step"], state["epoch"], state["batch_offset"]
        torch.set_rng_state(state["rng"][rt.rank]["cpu"])
        if rt.device.type == "cuda":
            torch.cuda.set_rng_state(state["rng"][rt.rank]["cuda"], rt.device)
    output_dir = Path(cfg.output_dir)
    if rt.main:
        output_dir.mkdir(parents=True, exist_ok=False)
        (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        print(
            json.dumps(
                {
                    **metadata,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            ),
            flush=True,
        )
    rt.barrier()
    train_model = (
        DistributedDataParallel(
            model,
            device_ids=[rt.local_rank] if rt.device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if rt.world > 1
        else model
    )
    sampler = WeightedDistributedSampler(
        weights,
        cfg.samples_per_epoch or len(train_rows),
        cfg.batch_size,
        cfg.accumulation_steps,
        rt.rank,
        rt.world,
        cfg.seed,
    )
    loader = DataLoader(
        train_rows,
        batch_size=cfg.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=cfg.workers,
        pin_memory=rt.device.type == "cuda",
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    while step < cfg.max_steps:
        sampler.epoch, sampler.offset = epoch, batch_offset * cfg.batch_size
        epoch_batches = sampler.total // rt.world // cfg.batch_size
        initial_offset = batch_offset
        for index, (batch, _) in enumerate(loader, initial_offset):
            sync = (index + 1) % cfg.accumulation_steps == 0
            context = train_model.no_sync() if not sync and rt.world > 1 else nullcontext()
            with context:
                with rt.autocast():
                    output = train_model(**move_batch(batch, rt.device))
                    loss = output["loss"] / cfg.accumulation_steps
                if not rt.all_finite(loss):
                    raise FloatingPointError("nonfinite loss; no update performed")
                loss.backward()
            if not sync:
                continue
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            if not rt.all_finite(norm):
                raise FloatingPointError("nonfinite gradients; no update performed")
            factor = learning_rate_factor(step, cfg.warmup_steps, cfg.max_steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            batch_offset = index + 1
            record = {
                "step": step,
                "rank0_microbatch_loss": float(output["loss"].detach()),
                "elapsed_seconds": time.monotonic() - started,
            }
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                metrics, predictions = evaluate(model, val_rows, collator, rt, cfg.batch_size)
                record["validation"] = metrics
                if rt.main:
                    (output_dir / f"validation-{step}.json").write_text(
                        json.dumps(metrics, indent=2)
                    )
                    if predictions:
                        with (output_dir / f"predictions-{step}.jsonl").open("w") as f:
                            for prediction in predictions:
                                f.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            if rt.main:
                with (output_dir / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if step == 1 or step % 10 == 0 or "validation" in record:
                    print(json.dumps(record), flush=True)
            if step % cfg.save_every == 0 or step == cfg.max_steps:
                rng = {"cpu": torch.get_rng_state()}
                if rt.device.type == "cuda":
                    rng["cuda"] = torch.cuda.get_rng_state(rt.device)
                training_state = {
                    "epoch": epoch,
                    "batch_offset": batch_offset,
                    "rng": rt.gather(rng),
                }
                if rt.main:
                    save_checkpoint(
                        output_dir / "last.pt",
                        model,
                        cfg,
                        tokenizer,
                        speakers,
                        languages,
                        assets,
                        step,
                        optimizer,
                        metadata,
                        training_state,
                    )
                rt.barrier()
            if step >= cfg.max_steps:
                break
        if batch_offset >= epoch_batches:
            epoch, batch_offset = epoch + 1, 0
