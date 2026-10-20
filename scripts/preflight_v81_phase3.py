"""Stage 3 gates: finished Stage 2 export, existing index, and optional coverage scan."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


def check_stage2_export(model_path: str, output_path: str) -> None:
    """Require the final save_state written after final evaluation and model export."""
    model = Path(model_path)
    if model.resolve() == Path(output_path).resolve():
        raise ValueError("Stage 3 output must not overwrite Stage 2")
    names = ("training_config.json", "trainer_state.json", "train_results.json")
    missing = [name for name in names if not (model / name).is_file()]
    if missing:
        raise ValueError(
            f"Stage 2 final export is not ready: missing {missing} in {model}. "
            "Wait for training, final evaluation, model save and trainer-state save to finish."
        )
    config, state, result = [json.loads((model / name).read_text()) for name in names]
    target = config.get("max_steps", -1)
    actual = state.get("global_step", -1)
    if config.get("smoke_test") or target <= 0 or actual < target:
        raise ValueError(f"Stage 2 is incomplete: global_step={actual}, required={target}")
    if not math.isfinite(float(result.get("train_loss", float("nan")))):
        raise ValueError("Stage 2 final train_loss is missing or non-finite")
    print(f"Stage 2 final export: {actual:,}/{target:,} steps complete", flush=True)


def scan_coverage(cfg, *, exact=True) -> None:
    # Keep just one manifest's index in memory, not all ~46M records together.
    from qasr.data import JsonlSpeechDataset

    counts = Counter()
    for path, language in cfg.train_manifest_specs:
        part = JsonlSpeechDataset(
            path, language=language, audio_root=cfg.audio_root,
            min_duration_seconds=cfg.min_duration_seconds,
            max_duration_seconds=cfg.max_duration_seconds, validate_audio_paths=False,
        )
        counts[language] += len(part)
        print(f"coverage: {language} {Path(path).name}: {len(part):,}", flush=True)
        del part
    batches = max(math.ceil(n / (cfg.expected_world_size * cfg.language_batch_counts[lang]))
                  for lang, n in counts.items())
    steps = math.ceil(batches / cfg.gradient_accumulation_steps)
    print(f"Eligible anchors: {dict(counts)}; full weighted epoch: {steps:,} updates", flush=True)
    if (exact and steps != cfg.max_steps) or cfg.max_steps < steps:
        raise ValueError(f"Coverage differs: computed={steps}, configured={cfg.max_steps}; review config")
    print(f'Configured budget: {cfg.max_steps:,} updates ({cfg.max_steps / steps:.3f} weighted epochs)', flush=True)


def main() -> None:
    from preflight_v81 import main as index_preflight

    from qasr.config import TrainConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v8.1/03_hq_12node.yaml')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--resume-from-checkpoint')
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.validate()
    check_stage2_export(cfg.model_name_or_path, cfg.output_dir)
    # Reuse tokenizer/frozen-index/shard/MUSAN/resume checks without its all-parts scan.
    forwarded = ['preflight_v81.py', '--config', args.config]
    if args.smoke_test:
        forwarded.append('--smoke-test')
    if args.resume_from_checkpoint:
        forwarded.extend(['--resume-from-checkpoint', args.resume_from_checkpoint])
    original_argv = sys.argv
    try:
        sys.argv = forwarded
        index_preflight()
    finally:
        sys.argv = original_argv
    if args.scan:
        scan_coverage(cfg)
    print('Stage 3 preflight passed. Compare baseline recognition metrics before promotion.', flush=True)


if __name__ == '__main__':
    main()
