"""Command-line entry point for one rank of the distributed trainer."""

from __future__ import annotations

import argparse
from dataclasses import replace

import torch.distributed as dist

from .config import TrainConfig
from .trainer import build_trainer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jev.train",
        description="Train a decision model with deterministic global batches on an HSDP mesh",
    )
    parser.add_argument("--config", required=True, help="Path to a strict TOML training config")
    parser.add_argument(
        "--resume", default=None,
        help="Override the config's resume setting: a checkpoint directory or 'latest'",
    )
    args = parser.parse_args(argv)
    config = TrainConfig.from_toml(args.config)
    if args.resume is not None:
        config = replace(config, resume=args.resume)
    trainer = build_trainer(config)
    trainer.run()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
