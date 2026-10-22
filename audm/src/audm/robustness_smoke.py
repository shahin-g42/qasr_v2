"""CPU augmentation rehearsal with listenable outputs and timing (no model weights)."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .robustness import RobustAugmenter, load_mono_audio
from .training_config import load_recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("Iterations must be positive")
    cfg, _, receipt = load_recipe(args.config, model="/unused", output=args.output)
    aug = RobustAugmenter(cfg.aud_robustness["augmentation"], cfg.augmentation)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    waves = [load_mono_audio(p, 16000) for p in args.audio]
    for i, wave in enumerate(waves):
        if not cfg.min_duration_seconds <= len(wave) / 16000 <= cfg.max_duration_seconds:
            raise ValueError(f"Input duration outside recipe limits: {args.audio[i]}")
        sf.write(root / f"original-{i}.wav", wave, 16000, subtype="FLOAT")
    start = time.perf_counter()
    for i in range(args.iterations):
        result = aug.augment_waveform(
            waves[i % len(waves)], cfg.min_duration_seconds, cfg.max_duration_seconds
        )
        # This rehearsal measures waveform augmentation only, not SpecAugment.
        aug.clean_rows.clear()
        sf.write(root / f"augmented-{i:03d}.wav", result, 16000, subtype="FLOAT")
    report = dict(
        seconds=time.perf_counter() - start,
        iterations=args.iterations,
        worker_local_counts=dict(aug.stats),
        recipe=receipt,
        scope="Single-process waveform augmentation plus output writing; excludes model and SpecAugment",
    )
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
