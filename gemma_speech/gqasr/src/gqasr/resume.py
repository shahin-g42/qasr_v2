"""Validate a checkpoint continuation without silently changing its recipe."""

from .tracking import training_config


def validate_resume(saved, cfg, metadata, allow_microbatch_change=False, allow_batch_change=False):
    old, new = training_config(saved["config"]), training_config(cfg)
    changes = {k: {"from": old.get(k), "to": new.get(k)}
               for k in old.keys() | new.keys() if old.get(k) != new.get(k)}
    if saved["index_metadata"] != metadata:
        raise ValueError("resume requires identical index metadata; do not rescan for this restart")
    allowed = {"batch_size", "accumulation_steps"} if allow_microbatch_change else set()
    if allow_batch_change:
        allowed |= {"batch_size", "accumulation_steps", "language_batch_counts"}
    unexpected = changes.keys() - allowed
    if unexpected:
        raise ValueError(f"resume config mismatch: {sorted(unexpected)}; "
                         "regrouping requires --allow-microbatch-change; "
                         "effective-batch changes require --allow-batch-change")
    old_batch = old["batch_size"] * old["accumulation_steps"]
    new_batch = new["batch_size"] * new["accumulation_steps"]
    if old_batch != new_batch and not allow_batch_change:
        raise ValueError("microbatch continuation must preserve the effective batch per rank")
    if allow_batch_change:
        old_quotas, new_quotas = old["language_batch_counts"], new["language_batch_counts"]
        if (old_quotas.keys() != new_quotas.keys()
                or sum(old_quotas.values()) != old_batch
                or sum(new_quotas.values()) != new_batch
                or any(old_quotas[k] * new_batch != new_quotas[k] * old_batch
                       for k in old_quotas)):
            raise ValueError("batch continuation must preserve language proportions and valid quotas")
    if not 0 <= saved["step"] < cfg["max_steps"]:
        raise ValueError("resume checkpoint must precede max_steps")
    if len(saved["rng"]) != cfg["expected_world_size"]:
        raise ValueError("resume requires RNG state for every rank in the same world size")
    return {
        "event": "training_resume",
        "completed_updates": saved["step"],
        "next_update": saved["step"] + 1,
        "remaining_updates": cfg["max_steps"] - saved["step"],
        "microbatch_changed": bool(changes),
        "effective_batch_changed": old_batch != new_batch,
        "global_batch": {
            "from": old_batch * cfg["expected_world_size"],
            "to": new_batch * cfg["expected_world_size"],
        },
        "changes": changes,
        "optimizer_restored": True,
        "schedule_restarted": False,
        # Changing grouping changes token weighting and augmentation seeds.
        "same_microbatch_recipe": not bool(changes),
    }


def check_checkpoint(config, checkpoint, allow_microbatch_change=False, allow_batch_change=False):
    """CPU-only readability, compatibility and finiteness check; no writes."""
    import hashlib
    import json
    from pathlib import Path

    import torch

    from .config import load_config
    from .data import ManifestIndex, data_contract
    from .modeling import GQASRForSpeechRecognition
    from .optimization import build_optimizer
    from .processing import GQASRProcessor

    torch.set_num_threads(8)
    cfg = load_config(config)
    if Path(cfg["output_dir"]).exists():
        raise FileExistsError(f"Restart requires a fresh output directory: {cfg['output_dir']}")
    print(json.dumps({"event": "resume_check", "stage": "reading_optimizer_state"}), flush=True)
    saved = torch.load(Path(checkpoint) / "training_state.pt", map_location="cpu", weights_only=True)
    metadata = ManifestIndex(cfg["index_path"], "train").metadata
    result = validate_resume(saved, cfg, metadata, allow_microbatch_change, allow_batch_change)
    if metadata["data_contract"] != data_contract(cfg):
        raise ValueError("index/config data contract mismatch")
    processor = GQASRProcessor(cfg["base_model"])
    tokenizer_hash = hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if metadata["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("tokenizer differs from the preflight index")
    print(json.dumps({"event": "resume_check", "stage": "loading_model_weights"}), flush=True)
    model = GQASRForSpeechRecognition.from_pretrained(checkpoint)
    model.configure_training(
        cfg["acoustic_layers"], cfg["decoder_layers"], cfg["train_embeddings"],
        cfg["gradient_checkpointing"], cfg["label_smoothing"],
        train_acoustic_frontend=cfg.get("train_acoustic_frontend"),
    )
    optimizer = build_optimizer(model, cfg)
    optimizer.load_state_dict(saved["optimizer"])
    print(json.dumps({"event": "resume_check", "stage": "checking_finite_tensors"}), flush=True)
    for name, value in model.named_parameters():
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite checkpoint parameter: {name}")
    for parameter, state in optimizer.state.items():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                if not torch.isfinite(value).all():
                    raise ValueError(f"nonfinite optimizer state: {name}")
                if name in {"exp_avg", "exp_avg_sq"} and value.shape != parameter.shape:
                    raise ValueError(f"optimizer state shape mismatch: {name}")
    return {
        **result, "event": "resume_check_complete", "checkpoint": str(Path(checkpoint).resolve()),
        "passed": True, "writes_model_weights": False,
        "kind": "checkpoint readability, configuration and finite tensor check; no cluster health test",
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Check a training checkpoint on CPU before recovery")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--allow-microbatch-change", action="store_true")
    parser.add_argument("--allow-batch-change", action="store_true")
    print(json.dumps(check_checkpoint(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
