"""Offline integration experiment with tiny random instances of the actual HF towers."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .inference import synthesize, transcribe
from .text import train_tokenizer
from .train import train


def tiny_configs():
    from transformers import EmbeddingGemma2TextConfig, Gemma4AudioConfig

    audio = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        output_proj_dims=48,
        subsampling_conv_channels=[128, 4],
        attention_chunk_size=4,
        attention_context_left=5,
    )
    text = EmbeddingGemma2TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size_per_layer_input=16,
        embedding_dim=24,
        layer_types=["sliding_attention", "full_attention"],
        per_layer_config={},
        sliding_window=16,
    )
    return audio, text


def prepare_smoke(directory):
    import soundfile as sf
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        EmbeddingGemma2Config,
        EmbeddingGemma2Model,
        Gemma4AudioFeatureExtractor,
        PreTrainedTokenizerFast,
    )

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    audio_config, text_config = tiny_configs()
    config = EmbeddingGemma2Config(
        audio_config=audio_config, text_config=text_config, vision_config=None
    )
    base = directory / "tiny-backbone"
    EmbeddingGemma2Model(config).save_pretrained(base)
    Gemma4AudioFeatureExtractor().save_pretrained(base)
    backend = Tokenizer(
        models.WordLevel({"<pad>": 0, "<unk>": 1, "مرحبا": 2, "hello": 3}, unk_token="<unk>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", unk_token="<unk>"
    ).save_pretrained(base)
    examples = [("مرحبا", "ar"), ("hello", "en")]
    train_tokenizer([t for t, _ in examples] * 10, directory / "speech.model", vocab_size=300)
    for split, count in (("train", 4), ("validation", 2)):
        rows = []
        for i in range(count):
            text, language = examples[i % 2]
            filename = directory / f"{split}-{i}.wav"
            timeline = np.arange(16000, dtype=np.float32) / 16000
            sf.write(filename, 0.05 * np.sin(2 * np.pi * (220 + i * 40) * timeline), 16000)
            rows.append(
                {
                    "id": f"{split}-{i}",
                    "audio_filepath": str(filename),
                    "text": text,
                    "language": language,
                    "speaker_id": "test-speaker",
                }
            )
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
    configs = {}
    for task in ("asr", "tts"):
        model = ModelConfig(
            task=task,
            model_id=str(base),
            revision="local-smoke",
            local_files_only=True,
            width=32,
            layers=2,
            heads=4,
            dropout=0,
            condition_dropout=0,
            train_backbone_layers=1,
        )
        cfg = TrainConfig(
            model=model,
            train_manifest=str(directory / "train.jsonl"),
            validation_manifest=str(directory / "validation.jsonl"),
            tokenizer=str(directory / "speech.model"),
            output_dir=str(directory / task),
            batch_size=2,
            max_steps=4,
            eval_every=4,
            save_every=2,
            workers=0,
            precision="fp32",
            max_audio_seconds=2,
            warmup_steps=1,
        )
        path = directory / f"{task}.json"
        path.write_text(json.dumps(asdict(cfg), indent=2))
        configs[task] = path
    return directory, configs


def smoke(directory):
    directory, configs = prepare_smoke(directory)
    for path in configs.values():
        train(path)
    asr = transcribe(
        directory / "asr/last.pt",
        directory / "validation.jsonl",
        directory / "predictions.jsonl",
        batch_size=2,
    )
    tts = synthesize(
        directory / "tts/last.pt",
        "مرحبا",
        "test-speaker",
        "ar",
        directory / "generated-mel.pt",
        seconds=0.5,
        steps=2,
    )
    result = {
        "kind": "synthetic integration test, not speech quality evaluation",
        "real_hf_towers": True,
        "pretrained_weights": False,
        "training_steps_per_task": 4,
        "asr": asr,
        "tts": tts,
    }
    (directory / "smoke-report.json").write_text(json.dumps(result, indent=2))
    return result
