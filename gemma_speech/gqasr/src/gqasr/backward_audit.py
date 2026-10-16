"""Single-GPU backward isolation on real indexed audio; never updates or saves weights."""

import argparse
import json
from contextlib import nullcontext

import torch

from .config import load_config
from .data import Collator, ManifestIndex, StratifiedUpdateSampler
from .modeling import GQASRForSpeechRecognition
from .processing import GQASRProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["bridge", "decoder", "acoustic", "full"], default="full")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--anomaly", action="store_true")
    parser.add_argument("--rank", type=int, default=0, help="Replay this training sampler rank")
    parser.add_argument("--update", type=int, default=0, help="Zero-based sampler update")
    parser.add_argument("--batches", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not 0 <= args.rank < cfg["expected_world_size"] or args.update < 0 or args.batches < 1:
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
    )
    model.train()
    index = ManifestIndex(cfg["index_path"], "train")
    processor = GQASRProcessor(cfg["base_model"])
    collator = Collator(processor, cfg, training=not args.no_augmentation)
    sampler = StratifiedUpdateSampler(
        index.counts, cfg["language_batch_counts"], cfg["batch_size"],
        args.update + args.batches, rank=args.rank, world=cfg["expected_world_size"],
        seed=cfg["data_seed"], start=args.update, augmentation_seeds=True,
    )
    print(json.dumps({
        "event": "backward_audit_start", **vars(args), "model": cfg["base_model"],
        "optimizer_updates": False, "distributed_reduction": False,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
    }), flush=True)
    for batch_number, keys in enumerate(sampler):
        if batch_number >= args.batches:
            break
        batch, rows = collator([index[key] for key in keys])
        print(json.dumps({
            "event": "audit_batch", "batch": batch_number,
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
        failed = False
        for group in groups.values():
            norms = group.pop("norms")
            group["finite_gradient_norm_fp64"] = float(torch.linalg.vector_norm(torch.stack(norms))) if norms else 0.0
            failed |= bool(group["nonfinite"] or group["missing"])
        print(json.dumps({
            "event": "backward_audit_result", "batch": batch_number, "loss": float(loss.detach()),
            "groups": groups, "passed": not failed,
            "original_fp32_norm_finite": bool(torch.isfinite(fp32_norm)),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }), flush=True)
        if failed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
