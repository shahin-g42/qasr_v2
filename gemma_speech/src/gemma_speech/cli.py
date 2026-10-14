import argparse
import json
from collections import Counter, defaultdict


def main():
    parser = argparse.ArgumentParser(
        description="Arabic-first multilingual EmbeddingGemma speech experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    tok = commands.add_parser(
        "tokenizer", help="Train a multilingual byte-fallback subword vocabulary"
    )
    tok.add_argument("--manifest", nargs="+", required=True)
    tok.add_argument("--output", required=True)
    tok.add_argument("--vocab-size", type=int, default=8192)
    tok.add_argument("--per-language", type=int, default=100000)
    tok.add_argument("--seed", type=int, default=42)
    tok.add_argument("--text-field", choices=["text", "tts_text"], default="text")
    for name in ("inspect", "validate", "train"):
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        if name == "train":
            group = p.add_mutually_exclusive_group()
            group.add_argument("--init-checkpoint", help="New fine-tuning stage; optimizer resets")
            group.add_argument("--resume", help="Continue a saved run into a new output directory")
    prefetch = commands.add_parser(
        "prefetch", help="Cache pinned model assets before distributed launch"
    )
    prefetch.add_argument("--output", required=True)
    choices = prefetch.add_mutually_exclusive_group()
    choices.add_argument("--vocoder", action="store_true")
    choices.add_argument("--qwen-asr", action="store_true")
    choices.add_argument("--qwen-tts", action="store_true")
    cache = commands.add_parser(
        "speaker-cache", help="Cache Gemma frames and Qwen speaker/codec targets in one environment"
    )
    for name in ("manifest", "output", "gemma", "qwen"):
        cache.add_argument(f"--{name}", required=True)
    cache.add_argument("--device", default="cpu")
    cache.add_argument("--max-seconds", type=float, default=15)
    cache.add_argument("--shard", type=int, default=0)
    cache.add_argument("--shards", type=int, default=1)
    cache.add_argument("--include-codes", action="store_true")
    speaker = commands.add_parser("speaker-train")
    speaker.add_argument("--config", required=True)
    speaker.add_argument("--resume")
    talker = commands.add_parser(
        "qwen-tts-train",
        help="Adapt Qwen's talker on cached codec targets with Gemma speaker conditioning",
    )
    talker.add_argument("--config", required=True)
    talker.add_argument("--resume")
    qtts = commands.add_parser(
        "qwen-synthesize",
        help="Use the trained Gemma speaker adapter with Qwen's speech generator and codec",
    )
    for name in ("checkpoint", "audio", "gemma", "qwen", "text", "language", "output"):
        qtts.add_argument(f"--{name}", required=True)
    qtts.add_argument("--device", default="cpu")
    qtts.add_argument("--seed", type=int, default=42)
    qtts.add_argument("--talker-checkpoint")
    asr = commands.add_parser("transcribe")
    asr.add_argument("--checkpoint", required=True)
    asr.add_argument("--manifest", required=True)
    asr.add_argument("--output", required=True)
    asr.add_argument("--device", default="cpu")
    asr.add_argument("--batch-size", type=int, default=1)
    tts = commands.add_parser("synthesize")
    tts.add_argument("--checkpoint", required=True)
    tts.add_argument("--text", required=True)
    tts.add_argument("--speaker", required=True)
    tts.add_argument("--language", required=True)
    tts.add_argument("--output", required=True)
    tts.add_argument("--device", default="cpu")
    tts.add_argument("--vocoder", help="Local vocos-mel-24khz directory; omit for mel output")
    tts.add_argument(
        "--seconds", type=float, help="Optional explicit duration; default is learned prediction"
    )
    tts.add_argument("--steps", type=int, default=32)
    tts.add_argument("--guidance", type=float, default=1.0)
    tts.add_argument("--seed", type=int, default=42)
    score = commands.add_parser(
        "score", help="Score saved predictions, including external baselines"
    )
    score.add_argument("--predictions", required=True)
    smoke = commands.add_parser(
        "smoke", help="Offline tiny real-HF-tower training/inference, not a quality benchmark"
    )
    smoke.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "tokenizer":
        from random import Random

        from .data import read_manifest
        from .text import train_tokenizer

        if args["per_language"] < 1:
            parser.error("per-language must be positive")
        groups = defaultdict(list)
        for manifest in args["manifest"]:
            for row in read_manifest(manifest):
                groups[row.language].append(getattr(row, args["text_field"]))
        rng = Random(args["seed"])
        texts = [
            t
            for language in sorted(groups)
            for t in rng.sample(groups[language], min(len(groups[language]), args["per_language"]))
        ]
        tokenizer = train_tokenizer(texts, args["output"], args["vocab_size"])
        print(
            json.dumps(
                {"vocabulary": len(tokenizer), "languages": sorted(groups), "sentences": len(texts)}
            )
        )
    elif command in {"inspect", "validate"}:
        inspect_config(args["config"], validate=command == "validate")
    elif command == "train":
        from .train import train

        train(args.pop("config"), **args)
    elif command == "prefetch":
        from huggingface_hub import snapshot_download

        from .config import MODEL_REVISION

        model = "charactr/vocos-mel-24khz" if args["vocoder"] else "google/embeddinggemma-2"
        revision = VOCOS_REVISION if args["vocoder"] else MODEL_REVISION
        if args["qwen_asr"]:
            from .config import ModelConfig

            model, revision = ModelConfig().qwen_model_id, ModelConfig().qwen_revision
        elif args["qwen_tts"]:
            from gqtts.speaker import QWEN_TTS_REVISION

            model, revision = "Qwen/Qwen3-TTS-12Hz-0.6B-Base", QWEN_TTS_REVISION
        path = snapshot_download(
            model,
            revision=revision,
            local_dir=args["output"],
            allow_patterns=[
                "*.json",
                "*.model",
                "*.safetensors",
                "*.yaml",
                "*.bin",
                "*.jinja",
                "*.txt",
            ],
        )
        print(path)
    elif command == "transcribe":
        from .inference import transcribe

        if args["batch_size"] < 1:
            parser.error("batch-size must be positive")
        print(json.dumps(transcribe(**args), indent=2, ensure_ascii=False))
    elif command == "synthesize":
        from .inference import synthesize

        print(json.dumps(synthesize(**args), indent=2, ensure_ascii=False))
    elif command == "score":
        from .metrics import score_predictions

        with open(args["predictions"]) as f:
            predictions = [json.loads(line) for line in f if line.strip()]
        print(json.dumps(score_predictions(predictions), indent=2, ensure_ascii=False))
    elif command == "smoke":
        from .smoke import smoke

        print(json.dumps(smoke(args["output"]), indent=2))
    elif command == "speaker-cache":
        from gqtts.speaker import cache_speaker

        print(json.dumps(cache_speaker(**args), indent=2))
    elif command == "speaker-train":
        from gqtts.speaker_train import train_speaker

        train_speaker(**args)
    elif command == "qwen-synthesize":
        from gqtts.speaker import synthesize_qwen

        print(json.dumps(synthesize_qwen(**args), indent=2))
    elif command == "qwen-tts-train":
        from gqtts.qwen_tts_train import train_qwen_tts

        train_qwen_tts(**args)


def inspect_config(config_path, validate=False):
    from .assets import load_assets, processors_from_assets
    from .batching import Collator
    from .config import TrainConfig
    from .data import check_disjoint, read_manifest, sampling_weights
    from .text import SpeechTokenizer

    cfg = TrainConfig.load(config_path)
    train = read_manifest(cfg.train_manifest, cfg.model.task)
    validation = read_manifest(cfg.validation_manifest, cfg.model.task)
    check_disjoint(train, validation)
    weights = sampling_weights(train, cfg.language_weights)
    counts = Counter(row.language for row in train)
    if cfg.require_multilingual and len(counts) < 2:
        raise ValueError("multilingual training requires at least two languages")
    masses = {
        language: float(weights[[r.language == language for r in train]].sum() / weights.sum())
        for language in counts
    }
    summary = {
        "task": cfg.model.task,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "language_counts": dict(counts),
        "sampling_probabilities": masses,
        "audio_checked": validate,
        "output_dir": cfg.output_dir,
    }
    if validate:
        if cfg.model.task == "asr" and cfg.model.asr_decoder == "qwen":
            from .qwen_asr import QwenTokenizer

            tokenizer = QwenTokenizer.from_pretrained(cfg.model)
        else:
            tokenizer = SpeechTokenizer(cfg.tokenizer)
        extractor, hf_tokenizer = processors_from_assets(load_assets(cfg.model))
        speakers = {v: i for i, v in enumerate(sorted({r.speaker for r in train}))}
        languages = {v: i for i, v in enumerate(sorted(counts))}
        if not {r.language for r in validation} <= languages.keys():
            raise ValueError("validation contains an unseen language")
        if cfg.model.task == "tts" and not {r.speaker for r in validation} <= speakers.keys():
            raise ValueError("TTS validation contains an unseen speaker")
        collator = Collator(cfg, tokenizer, extractor, hf_tokenizer, speakers, languages)
        for records in (train, validation):
            for start in range(0, len(records), cfg.batch_size):
                batch, selected = collator(records[start : start + cfg.batch_size])
                if cfg.model.task == "asr" and cfg.model.asr_decoder == "ctc":
                    # Exact mask slicing in the inspected Gemma4 subsampling implementation.
                    lengths = (
                        batch["input_features_mask"][:, ::2][:, ::2].sum(-1)
                        * cfg.model.asr_upsample
                    )
                    targets, mask = batch["targets"], batch["target_mask"]
                    repeats = ((targets[:, 1:] == targets[:, :-1]) & mask[:, 1:]).sum(-1)
                    invalid = (lengths < mask.sum(-1) + repeats) | (lengths == 0)
                    if invalid.any():
                        ids = [r.id for r, bad in zip(selected, invalid, strict=True) if bad]
                        raise ValueError(
                            f"CTC alignment infeasible for {ids}; try larger vocabulary or asr_upsample=2"
                        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


# Pinned after inspection of the upstream public model repository.
VOCOS_REVISION = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
