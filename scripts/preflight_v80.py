"""Check v8.0 inputs; --scan confirms post-filter counts and the epoch budget.

Run through run_v80_phase2_node.sh --preflight, or with PYTHONPATH=src in qasr3.
This never loads model weights onto a GPU or modifies the frozen manifests.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from qasr.config import TrainConfig, validate_local_checkpoint
from qasr.data import JsonlSpeechDataset
from qasr.processing import QASRProcessor


def check_inputs(config: TrainConfig, *, resume: str | None = None, smoke: bool = False) -> None:
    config.validate()
    validate_local_checkpoint(config.model_name_or_path)
    model = Path(config.model_name_or_path)
    for index in model.glob("*.safetensors.index.json"):
        shards = set(json.loads(index.read_text())["weight_map"].values())
        for name in shards:
            shard = model / name
            if not shard.is_file() or shard.stat().st_size == 0:
                raise ValueError(f"Missing/empty checkpoint shard: {shard}")
    for path in config.train_manifest_paths + config.eval_manifest_paths:
        if not Path(path).is_file() or Path(path).stat().st_size == 0:
            raise ValueError(f"Missing/empty manifest: {path}")
    if set(config.train_manifest_paths) & set(config.eval_manifest_paths):
        raise ValueError("A manifest is listed in both train and eval")
    ds = Path(config.deepspeed or "")
    if not ds.is_file():
        raise ValueError(f"Missing DeepSpeed config: {ds}")
    noise = (config.augmentation or {}).get("noise_injection", {})
    if noise.get("enabled") and not Path(noise["noise_dir"]).is_dir():
        raise ValueError(f"Missing MUSAN directory: {noise['noise_dir']}")
    resume = resume or config.resume_from_checkpoint
    if resume:
        validate_local_checkpoint(resume)
        if not (Path(resume) / "trainer_state.json").is_file():
            raise ValueError("Resume requires a complete training checkpoint with trainer_state.json")
        if Path(resume).resolve().parent != Path(config.output_dir).resolve():
            raise ValueError("Resume must point to a checkpoint in this v8.0 output directory")
    elif not smoke and any(Path(config.output_dir).glob("checkpoint-*")):
        raise ValueError("v8.0 checkpoints already exist; pass --resume-from-checkpoint to restart")


def scan_counts(config: TrainConfig, expected: dict) -> dict[str, int]:
    totals: Counter = Counter()
    differences = []
    for split, specs in (("train", config.train_manifest_specs), ("eval", config.eval_manifest_specs)):
        for path, lang in specs:
            data = JsonlSpeechDataset(path, language=lang, audio_root=config.audio_root,
                                      min_duration_seconds=config.min_duration_seconds,
                                      max_duration_seconds=config.max_duration_seconds,
                                      validate_audio_paths=False)
            count = len(data)
            print(f"{split} {lang}: {Path(path).name}: {count:,} eligible records", flush=True)
            wanted = expected[lang][Path(path).stem]
            if count != wanted:
                differences.append(f"{Path(path).name}: expected {wanted:,}, eligible {count:,}")
            if split == "train":
                totals[lang] += count
            del data
    world = config.expected_world_size
    quotas = config.language_batch_counts
    batches = max(math.ceil(n / (world * quotas[lang])) for lang, n in totals.items())
    steps = math.ceil(batches / config.gradient_accumulation_steps)
    print(f"Training totals: {dict(totals)}; full coverage requires {steps:,} updates", flush=True)
    if differences or steps != config.max_steps:
        raise ValueError("Counts/budget differ; review before launching:\n" + "\n".join(differences)
                         + f"\ncomputed steps={steps}, configured={config.max_steps}")
    return dict(totals)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v8.0/02_full_10node.yaml")
    parser.add_argument("--scan", action="store_true", help="scan all manifests once; CPU/I/O intensive")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = TrainConfig.from_yaml(args.config)
    check_inputs(config, resume=args.resume_from_checkpoint, smoke=args.smoke_test)
    processor = QASRProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model_config = json.loads((Path(config.model_name_or_path) / "config.json").read_text())
    if getattr(processor, "projector_pool_stride", 1) != model_config.get("projector_pool_stride", 1):
        raise ValueError("Processor/model pooling stride mismatch")
    if config.max_duration_seconds > processor.feature_extractor.max_audio_clip_s:
        raise ValueError("Configured duration exceeds the processor limit")
    if args.scan:
        expected = json.loads((Path(args.config).parent / "manifest_counts.json").read_text())["counts"]
        scan_counts(config, expected)
    print("v8.0 preflight passed. Audio decoding, target-token limits and GPU memory still require rehearsal.")


if __name__ == "__main__":
    main()
