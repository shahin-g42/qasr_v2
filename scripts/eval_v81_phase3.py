"""Run the same deterministic offline evaluation suite before and after Stage 3."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

from qasr.config import TrainConfig


def evaluation_commands(cfg, model, output, samples, language=None):
    for manifest, lang in cfg.eval_manifest_specs:
        if language is not None and language != lang:
            continue
        result = output / f'{lang}-{Path(manifest).stem}.json'
        yield result, [
            sys.executable, 'evaluate.py', '--model', model, '--manifest', manifest,
            '--language', lang, '--modes', 'offline', '--num-samples', str(samples),
            '--seed', str(cfg.seed), '--min-duration', str(cfg.min_duration_seconds),
            '--max-duration', str(cfg.max_duration_seconds),
            '--max-new-tokens', str(cfg.eval_generation_max_new_tokens), '--output', str(result),
        ] + (['--audio-root', cfg.audio_root] if cfg.audio_root else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/v8.1/03_hq_12node.yaml')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=1000,
                        help='Per manifest; uses all eligible rows when fewer exist')
    parser.add_argument('--language', choices=['ar', 'en', 'hi', 'ml', 'zh'])
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error('--num-samples must be positive')
    cfg = TrainConfig.from_yaml(args.config)
    output = Path(args.output_dir)
    commands = list(evaluation_commands(cfg, args.model, output, args.num_samples, args.language))
    if not args.dry_run:
        if any(path.exists() for path, _ in commands):
            parser.error('Evaluation outputs already exist; use a new output directory')
        output.mkdir(parents=True, exist_ok=True)
    for _, command in commands:
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
