"""GQTTS commands sharing the parent project's compatible runtime."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="GQTTS speaker adaptation and TTS")
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    for name in ("manifest", "output", "gemma", "qwen"):
        cache.add_argument("--" + name, required=True)
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--max-seconds", type=float, default=15)
    cache.add_argument("--shard", type=int, default=0)
    cache.add_argument("--shards", type=int, default=1)
    cache.add_argument("--include-codes", action="store_true")
    for command in ("speaker-train", "train"):
        p = commands.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--resume")
    p = commands.add_parser("synthesize")
    for name in ("checkpoint", "audio", "gemma", "qwen", "text", "language", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--talker-checkpoint")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "cache":
        from .speaker import cache_speaker

        print(json.dumps(cache_speaker(**args), indent=2))
    elif command == "speaker-train":
        from .speaker_train import train_speaker

        train_speaker(**args)
    elif command == "train":
        from .qwen_tts_train import train_qwen_tts

        train_qwen_tts(**args)
    else:
        from .speaker import synthesize_qwen

        print(json.dumps(synthesize_qwen(**args), indent=2))
