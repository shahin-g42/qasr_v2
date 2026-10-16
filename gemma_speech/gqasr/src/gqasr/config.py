import hashlib
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if cfg.get("format_version") != 1:
        raise ValueError("unsupported training config")
    for key in ("base_model", "output_dir", "index_path"):
        cfg[key] = str((path.parent / Path(cfg[key]).expanduser()).resolve())
    if cfg["precision"] not in {"fp32", "bf16"}:
        raise ValueError("only fp32/bf16 supported")
    if (
        min(
            cfg[k]
            for k in (
                "batch_size",
                "accumulation_steps",
                "max_steps",
                "eval_steps",
                "save_steps",
                "save_total_limit",
                "logging_steps",
                "expected_world_size",
                "eval_batch_size",
                "eval_samples_per_language",
            )
        )
        < 1
    ):
        raise ValueError("invalid training batch/steps")
    if sum(cfg["language_batch_counts"].values()) != cfg["batch_size"] * cfg["accumulation_steps"]:
        raise ValueError("language quotas must fill one per-rank optimizer update")
    if set(cfg["train_manifest"]) != set(cfg["eval_manifest"]) or set(cfg["train_manifest"]) != set(
        cfg["language_batch_counts"]
    ):
        raise ValueError("training, evaluation and quotas must cover the same languages")
    if cfg["workers"] < 0 or min(cfg["language_batch_counts"].values()) < 1:
        raise ValueError("workers must be nonnegative and language quotas positive")
    if not 0 <= cfg["min_lr_ratio"] <= 1 or min(cfg["warmup_steps"], cfg["decay_steps"]) < 0:
        raise ValueError("invalid learning-rate schedule")
    cfg.setdefault("code_switch_probability", 0.4)
    if not 0 <= cfg["code_switch_probability"] <= 1:
        raise ValueError("code_switch_probability must be in [0, 1]")
    if cfg.get("online_code_switching") and not cfg.get("code_switch_index"):
        raise ValueError("online_code_switching requires code_switch_index")
    if cfg.get("augmentation") and not isinstance(cfg["augmentation"], dict):
        raise ValueError("augmentation must be a mapping or false")
    if cfg.get("code_switch_index"):
        cfg["code_switch_index"] = str(
            (path.parent / Path(cfg["code_switch_index"]).expanduser()).resolve()
        )
    if cfg.get("code_switch_exclude_manifest"):
        cfg["code_switch_exclude_manifest"] = str(
            (path.parent / Path(cfg["code_switch_exclude_manifest"]).expanduser()).resolve()
        )
    return cfg


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    src = yaml.safe_load(source.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for name in ("01_bridge_12node.yaml", "02_full_12node.yaml", "02_full_b40_12node.yaml", "source_mapping.yaml"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    cfg = dict(
        format_version=1,
        model_name="gqasr",
        base_model="../models/gqasr-base",
        output_dir="../outputs/bridge",
        index_path="../indexes/full_v81.sqlite",
        train_manifest=src["train_manifest"],
        eval_manifest=src["eval_manifest"],
        audio_root=src.get("audio_root"),
        audio_path_rewrites={"/vast": "/lustrefs/taiga/vast40"},
        min_duration_seconds=src["min_duration_seconds"],
        max_duration_seconds=src["max_duration_seconds"],
        max_target_length=src["max_target_length"],
        eval_generation_max_new_tokens=src["eval_generation_max_new_tokens"],
        eval_samples_per_language=src["eval_wer_samples_per_language"],
        language_batch_counts=src["language_batch_counts"],
        expected_world_size=src["expected_world_size"],
        batch_size=20,
        accumulation_steps=1,
        eval_batch_size=src["per_device_eval_batch_size"],
        precision="bf16",
        tf32=True,
        workers=4,
        seed=src["seed"],
        data_seed=src["data_seed"],
        acoustic_layers=0,
        decoder_layers=0,
        train_embeddings=False,
        gradient_checkpointing=True,
        max_steps=1000,
        learning_rate=1e-4,
        acoustic_learning_rate=1e-5,
        decoder_learning_rate=1e-5,
        embedding_learning_rate=5e-6,
        weight_decay=src["weight_decay"],
        grad_clip=src["max_grad_norm"],
        warmup_steps=100,
        decay_steps=200,
        min_lr_ratio=0.1,
        label_smoothing=0.0,
        eval_steps=250,
        save_steps=250,
        save_total_limit=src["save_total_limit"],
        logging_steps=10,
        source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        online_code_switching=False,
        augmentation=False,
    )
    (output / "01_bridge_12node.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    cfg.update(
        online_code_switching=bool(src.get("code_switch_index")),
        code_switch_index=src.get("code_switch_index"),
        code_switch_probability=src.get("code_switch_probability", 0.4),
        augmentation=src.get("augmentation", False),
        base_model="../outputs/bridge/final",
        output_dir="../outputs/full_b40_frozen",
        batch_size=40,
        accumulation_steps=1,
        language_batch_counts={k: 2 * v for k, v in src["language_batch_counts"].items()},
        logging_steps=5,
        acoustic_layers=0,
        decoder_layers=28,
        train_embeddings=True,
        max_steps=(src["max_steps"] + 1) // 2,
        learning_rate=src["learning_rate"],
        acoustic_learning_rate=src["learning_rate"],
        decoder_learning_rate=src["learning_rate"] * src["llm_lr_factor"],
        embedding_learning_rate=src["learning_rate"] * src["embed_lr_factor"],
        warmup_steps=(src["warmup_steps"] + 1) // 2,
        decay_steps=(src["lr_scheduler_kwargs"]["num_decay_steps"] + 1) // 2,
        eval_steps=(src["eval_steps"] + 1) // 2,
        save_steps=(src["save_steps"] + 1) // 2,
        label_smoothing=src["label_smoothing"],
    )
    (output / "02_full_b40_12node.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    cfg.update(
        output_dir="../outputs/full_b20_frozen",
        batch_size=20,
        language_batch_counts=dict(src["language_batch_counts"]),
        max_steps=src["max_steps"],
        warmup_steps=src["warmup_steps"],
        decay_steps=src["lr_scheduler_kwargs"]["num_decay_steps"],
        eval_steps=src["eval_steps"],
        save_steps=src["save_steps"],
        logging_steps=10,
    )
    (output / "02_full_12node.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    )
    mapping = {
        "source": str(source),
        "source_sha256": cfg["source_config_sha256"],
        "preserved": [
            "train_manifest",
            "eval_manifest",
            "duration bounds",
            "target limit",
            "language proportions",
            "approximate full-stage example budget",
            "approximate full-stage schedule measured in examples",
        ],
        "changed": {
            "batching": "bridge: 20 x 1, global 1920; next stage: 40 x 1, global 3840, doubled quotas and halved step schedule",
            "initialization": "new GQASR pretrained composite and random bridge; warm-up stage precedes full adaptation",
            "workers": "4 per GPU instead of 12, leaving CPU headroom on 96-core nodes",
            "distributed": "native DDP, not DeepSpeed ZeRO-2",
            "logging": "local JSONL, not wandb",
        },
        "full_stage_augmentations": {
            "code_switch_probability": src.get("code_switch_probability"),
            "code_switch_index": src.get("code_switch_index"),
            "augmentation": src.get("augmentation"),
        },
        "data_access": "manifest paths copied; cluster-side scan still required",
    }
    (output / "source_mapping.yaml").write_text(
        yaml.safe_dump(mapping, sort_keys=False, allow_unicode=True)
    )
    return {
        "configs": str(output),
        "train_manifests": sum(map(len, src["train_manifest"].values())),
        "eval_manifests": sum(map(len, src["eval_manifest"].values())),
    }
