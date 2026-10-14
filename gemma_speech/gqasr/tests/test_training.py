import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from gqasr.config import load_config
from gqasr.data import ManifestIndex, StratifiedUpdateSampler, normalize_row, scan
from gqasr.modeling import (
    GQASRAcousticConfig,
    GQASRConfig,
    GQASRDecoderConfig,
    GQASRForSpeechRecognition,
)
from gqasr.processing import GQASRFeatureExtractor, GQASRProcessor
from gqasr.train import train
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def tiny_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    audio = GQASRAcousticConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        output_proj_dims=48,
        subsampling_conv_channels=[128, 4],
        attention_chunk_size=4,
        attention_context_left=5,
    )
    decoder = GQASRDecoderConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=True,
    )
    return GQASRForSpeechRecognition(
        GQASRConfig(audio.to_dict(), decoder.to_dict(), [4], [5], 1, 16)
    )


def artifact(path):
    model = tiny_model()
    model.save_pretrained(path)
    backend = Tokenizer(
        models.WordLevel(
            {"<unk>": 0, "<|im_end|>": 1, "<asr_text>": 2, "مرحبا": 3, "language": 4, "Arabic": 5},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<asr_text>"],
    ).save_pretrained(path / "tokenizer")
    (path / "preprocessor_config.json").write_text(GQASRFeatureExtractor().to_json_string())
    (path / "provenance.json").write_text("{}")
    (path / "MODEL_CARD.md").write_text("Test fixture")
    (path / "licenses").mkdir()
    return model


def fixture(root):
    root.mkdir(exist_ok=True)
    artifact(root / "base")
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/01_bridge_12node.yaml").read_text())
    for split in ("train", "eval"):
        sf.write(root / f"{split}.wav", np.sin(np.arange(4800) / 10).astype("float32") * 0.1, 16000)
        manifest = root / f"{split}.jsonl"
        manifest.write_text(
            json.dumps({"wav_path": f"{split}.wav", "transcript": "مرحبا", "duration": 0.3}) + "\n"
        )
        cfg[split + "_manifest"] = {"ar": [str(manifest)]}
    cfg.update(
        base_model=str(root / "base"),
        output_dir=str(root / "out"),
        index_path=str(root / "index.sqlite"),
        language_batch_counts={"ar": 2},
        batch_size=1,
        accumulation_steps=2,
        expected_world_size=1,
        max_steps=2,
        workers=0,
        precision="fp32",
        eval_steps=2,
        save_steps=1,
        eval_samples_per_language=1,
        eval_generation_max_new_tokens=2,
        warmup_steps=0,
        decay_steps=1,
    )
    path = root / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    cfg = load_config(path)
    scan(cfg, GQASRProcessor(root / "base"), True)
    return path, cfg


def test_weight_roundtrip_and_full_gradient_checkpointing(tmp_path):
    model = tiny_model()
    model.save_pretrained(tmp_path / "base")
    loaded = GQASRForSpeechRecognition.from_pretrained(tmp_path / "base")
    for key, value in model.state_dict().items():
        assert key.startswith("gqasr_")
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)
    assert not any(
        "Gemma" in type(m).__name__ or "Qwen" in type(m).__name__ for m in loaded.modules()
    )
    loaded.configure_training(2, 2, True, True)
    loss = loaded(
        torch.randn(2, 41, 128),
        torch.ones(2, 41, dtype=torch.bool),
        torch.tensor([[3, 1], [3, 1]]),
        torch.ones(2, 2, dtype=torch.bool),
    )["loss"]
    loss.backward()
    for module in (loaded.gqasr_acoustic, loaded.gqasr_bridge, loaded.gqasr_decoder):
        gradients = [p.grad for p in module.parameters() if p.requires_grad]
        assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)


def test_exact_quotas_and_resume():
    quotas = {"ar": 10, "en": 6, "hi": 2, "ml": 1, "zh": 1}
    counts = dict.fromkeys(quotas, 1000)
    batches = list(StratifiedUpdateSampler(counts, quotas, 2, 3, rank=3, world=96))
    for i in range(3):
        assert (
            Counter(lang for batch in batches[i * 10 : (i + 1) * 10] for lang, _ in batch) == quotas
        )
    assert (
        list(StratifiedUpdateSampler(counts, quotas, 2, 3, rank=3, world=96, start=1))
        == batches[10:]
    )


def test_path_rewrite_and_alias():
    cfg = {"audio_path_rewrites": {"/vast": "/lustrefs/taiga/vast40"}, "audio_root": None}
    row = normalize_row(
        {"wav_path": "/vast/x.wav", "text": "", "transcript": "مرحبا"}, "/tmp/x.jsonl", "ar", cfg
    )
    assert row[0] == "/lustrefs/taiga/vast40/x.wav"


def test_index_training_resume_and_full_stage(tmp_path):
    path, cfg = fixture(tmp_path)
    assert ManifestIndex(cfg["index_path"], "train")[("ar", 1)]["text"] == "مرحبا"
    train(path)
    cfg["output_dir"] = str(tmp_path / "resumed")
    path.write_text(yaml.safe_dump(cfg))
    train(path, tmp_path / "out/checkpoint-00000001")
    original = GQASRForSpeechRecognition.from_pretrained(tmp_path / "out/final")
    resumed = GQASRForSpeechRecognition.from_pretrained(tmp_path / "resumed/final")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    cfg.update(
        base_model=str(tmp_path / "out/final"),
        output_dir=str(tmp_path / "full"),
        acoustic_layers=2,
        decoder_layers=2,
        train_embeddings=True,
        max_steps=1,
    )
    path.write_text(yaml.safe_dump(cfg))
    train(path)
    assert (tmp_path / "full/final/model.safetensors").exists()


def test_index_leakage_and_overlong_filter(tmp_path):
    path, cfg = fixture(tmp_path)
    cfg["index_path"] = str(tmp_path / "bad.sqlite")
    cfg["eval_manifest"] = cfg["train_manifest"]
    with pytest.raises(ValueError, match="leakage"):
        scan(cfg, GQASRProcessor(tmp_path / "base"))
    assert not Path(cfg["index_path"]).exists()
    cfg = load_config(path)
    cfg["index_path"] = str(tmp_path / "short.sqlite")
    cfg["max_target_length"] = 1
    with pytest.raises(ValueError, match="no usable"):
        scan(cfg, GQASRProcessor(tmp_path / "base"))
