"""Single-GPU real-audio check; optional in-memory optimizer steps, never saves weights."""

import argparse
import json
from contextlib import nullcontext
from itertools import islice

import torch

from .config import load_config
from .data import Collator, ManifestIndex, StratifiedUpdateSampler
from .modeling import GQASRForSpeechRecognition
from .optimization import build_optimizer, training_scope
from .processing import GQASRProcessor


def audit_batches(counts, cfg, rank, rank_count, update, batches):
    for sampled_rank in range(rank, rank + rank_count):
        sampler = StratifiedUpdateSampler(
            counts, cfg["language_batch_counts"], cfg["batch_size"],
            update + batches, rank=sampled_rank, world=cfg["expected_world_size"],
            seed=cfg.get("training_data_seed", cfg["data_seed"]), start=update, augmentation_seeds=True,
        )
        for batch_number, keys in enumerate(islice(sampler, batches)):
            yield sampled_rank, batch_number, keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["bridge", "decoder", "acoustic", "full"], default="full")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true",
                        help="exercise AdamW/clipping in memory, including optimizer-state GPU memory; saves nothing")
    parser.add_argument("--rank", type=int, default=0, help="Replay this training sampler rank")
    parser.add_argument("--rank-count", type=int, default=1)
    parser.add_argument("--update", type=int, default=0, help="Zero-based sampler update")
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("streaming_datasets"):
        parser.error("This audit replays the local index only; use a config without streaming_datasets")
    if args.optimizer_step and cfg["accumulation_steps"] != 1:
        parser.error("--optimizer-step currently requires accumulation_steps=1")
    if not 0 <= args.rank < cfg["expected_world_size"] or args.update < 0 or args.batches < 1 or args.rank_count < 1 or args.rank + args.rank_count > cfg["expected_world_size"]:
        parser.error("invalid rank, update or batch count")
    if not torch.cuda.is_available():
        raise RuntimeError("Run the real-weight audit on a cluster CUDA GPU")
    device = torch.device("cuda", 0)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = cfg["tf32"]
    torch.backends.cudnn.allow_tf32 = cfg["tf32"]
    model = GQASRForSpeechRecognition.from_pretrained(cfg["base_model"]).to(device)
    model.configure_training(
        cfg["acoustic_layers"] if args.mode in ("acoustic", "full") else 0,
        cfg["decoder_layers"] if args.mode in ("decoder", "full") else 0,
        cfg["train_embeddings"] and args.mode in ("decoder", "full"),
        cfg["gradient_checkpointing"] and not args.no_checkpointing,
        cfg["label_smoothing"],
        train_acoustic_frontend=(cfg.get("train_acoustic_frontend")
                                 if args.mode in ("acoustic", "full") else False),
    )
    optimizer = build_optimizer(model, cfg) if args.optimizer_step else None
    model.train()
    index = ManifestIndex(cfg["index_path"], "train")
    processor = GQASRProcessor(cfg["base_model"])
    from .data import data_contract
    import hashlib
    if index.metadata["data_contract"] != data_contract(cfg):
        raise ValueError("index/config data contract mismatch")
    if index.metadata["tokenizer_sha256"] != hashlib.sha256(
        processor.tokenizer.backend_tokenizer.to_str().encode()
    ).hexdigest():
        raise ValueError("tokenizer differs from the preflight index")
    collator = Collator(processor, cfg, training=not args.no_augmentation)
    print(json.dumps({
        "event": "backward_audit_start", **vars(args), "model": cfg["base_model"],
        "optimizer_updates": args.optimizer_step, "writes_model_weights": False,
        "distributed_reduction": False,
        "training_scope": training_scope(model, optimizer or build_optimizer(model, cfg)),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }), flush=True)
    optimizer_updates = 0
    peak_memory, specaugment_changed = 0.0, 0
    for sampled_rank, batch_number, keys in audit_batches(
        index.counts, cfg, args.rank, args.rank_count, args.update, args.batches
    ):
        batch, rows = collator([index[key] for key in keys])
        specaugment_changed += sum(bool(row.get("augmentation_audit", {}).get("specaugment_changed"))
                                   for row in rows)
        print(json.dumps({
            "event": "audit_batch", "rank": sampled_rank, "batch": batch_number,
            "rows": [{k: r.get(k) for k in ("id", "audio", "duration", "language", "augmentation_audit")} for r in rows],
            "feature_shape": list(batch["input_features"].shape),
            "valid_frames": batch["input_features_mask"].sum(-1).tolist(),
        }), flush=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(device)
        # Anomaly mode includes the forward trace and may intentionally stop before the summary.
        with torch.autograd.detect_anomaly() if args.anomaly else nullcontext():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                loss = model(**batch)["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("audit loss is nonfinite")
            loss.backward()
        fp32_norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(p.grad.float()) for p in model.parameters() if p.grad is not None
        ]))
        groups = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = groups.setdefault(name.split(".")[0], {"missing": [], "nonfinite": [], "norms": []})
            if p.grad is None:
                group["missing"].append(name)
            elif not torch.isfinite(p.grad).all():
                group["nonfinite"].append(name)
            else:
                group["norms"].append(torch.linalg.vector_norm(p.grad, dtype=torch.float64))
        failed = not bool(torch.isfinite(fp32_norm))
        for group in groups.values():
            norms = group.pop("norms")
            group["finite_gradient_norm_fp64"] = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
            failed |= bool(group["nonfinite"] or group["missing"])
        if not failed and optimizer is not None:
            from .gradients import clip_gradients
            from .train import schedule

            class LocalRuntime:
                rank = sampled_rank

                @staticmethod
                def all_finite(value):
                    return bool(torch.isfinite(value).all())

            clip_gradients(model, cfg["grad_clip"], LocalRuntime(), optimizer_updates)
            factor = schedule(optimizer_updates, cfg)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.step()
            optimizer_updates += 1
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                raise FloatingPointError("nonfinite parameter after audit optimizer step")
        peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device) / 2**30)
        print(json.dumps({
            "event": "backward_audit_result", "rank": sampled_rank, "batch": batch_number, "loss": float(loss.detach()),
            "groups": groups, "passed": not failed,
            "original_fp32_norm_finite": bool(torch.isfinite(fp32_norm)),
            "optimizer_updates": optimizer_updates,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }), flush=True)
        if failed:
            raise SystemExit(1)
    print(json.dumps({"event": "backward_audit_complete", "passed": True,
                      "batches": args.rank_count * args.batches,
                      "optimizer_updates": optimizer_updates, "writes_model_weights": False,
                      "specaugment_changed": specaugment_changed,
                      "peak_allocated_gib": peak_memory}), flush=True)


if __name__ == "__main__":
    main()
