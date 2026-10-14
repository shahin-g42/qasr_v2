import json
from types import SimpleNamespace

import pytest
import torch
from gqtts.qwen_tts_train import QwenTTSFineTune
from gqtts.speaker import SpeakerAdapter, distillation_loss
from gqtts.speaker_train import train_speaker
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from gemma_speech._vendor.qwen_tts.core.models import (
    Qwen3TTSConfig,
    Qwen3TTSForConditionalGeneration,
)


def tiny_tts_api():
    sub = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_code_groups=16,
        layer_types=["full_attention"] * 2,
        max_position_embeddings=256,
    )
    talker = sub | dict(
        vocab_size=2048,
        text_vocab_size=128,
        text_hidden_size=32,
        code_predictor_config=sub,
        codec_bos_id=40,
        codec_eos_token_id=41,
        codec_think_id=42,
        codec_nothink_id=43,
        codec_think_bos_id=44,
        codec_think_eos_id=45,
        codec_pad_id=46,
        codec_language_id={},
        spk_id={},
        spk_is_dialect={},
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "interleaved": True},
    )
    cfg = Qwen3TTSConfig(
        talker_config=talker,
        speaker_encoder_config={"enc_dim": 32},
        tts_bos_token_id=50,
        tts_eos_token_id=51,
        tts_pad_token_id=52,
        tokenizer_type="qwen3_tts_tokenizer_12hz",
        tts_model_size="0b6",
        tts_model_type="base",
    )
    model = Qwen3TTSForConditionalGeneration(cfg)
    vocabulary = {"<unk>": 0, "<|im_start|>": 1, "assistant": 2, "مرحبا": 3, "hello": 4}
    tok = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")
    return SimpleNamespace(model=model, processor=SimpleNamespace(tokenizer=hf))


def test_speaker_pool_mask_and_distillation():
    adapter = SpeakerAdapter(input_dim=48, width=32, output_dim=32)
    frames = torch.randn(2, 7, 48)
    mask = torch.arange(7)[None] < torch.tensor([7, 4])[:, None]
    a = adapter(frames, mask)
    changed = frames.clone()
    changed[1, 4:] = 1e5
    torch.testing.assert_close(a, adapter(changed, mask))
    teacher = torch.randn(2, 32)
    loss, _ = distillation_loss(a, teacher)
    loss.backward()
    assert adapter.attention[0].weight.grad.abs().sum() > 0
    zero, _ = distillation_loss(teacher, teacher)
    assert abs(float(zero)) < 1e-6
    with pytest.raises(ValueError):
        adapter(frames, torch.zeros_like(mask))


def test_qwen_tts_codec_teacher_forcing_backward(tmp_path):
    model = QwenTTSFineTune(tiny_tts_api(), SpeakerAdapter(48, 32, 32))
    rows = []
    for i, n in enumerate([3, 5]):
        file = tmp_path / f"{i}.pt"
        torch.save({"frames": torch.randn(9, 48), "codes": torch.randint(0, 32, (n, 16))}, file)
        rows.append({"file": str(file), "text": ["مرحبا", "hello"][i]})
    embeds, mask, labels, bounds, codes = model.pack(rows)
    for i, (start, count) in enumerate(bounds):
        assert (labels[i, :start] == -100).all()
        assert torch.equal(labels[i, start : start + count], codes[i][:, 0])
        assert labels[i, start + count] == 41
    loss = model(rows)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.talker.text_projection.linear_fc1.weight.grad is not None
    assert all(p.grad is None for p in model.adapter.parameters())
    assert all(p.grad is not None for p in model.talker.parameters() if p.requires_grad)


def test_speaker_training_and_resume(tmp_path):
    metadata = {"shards": 1, "gemma_config": {"test": True}, "include_codes": False}
    for split in ["train", "val"]:
        directory = tmp_path / split
        directory.mkdir()
        records = []
        for i in range(4):
            filename = f"{i}.pt"
            torch.save(
                {"frames": torch.randn(5 + i, 48), "teacher": torch.randn(32)}, directory / filename
            )
            records.append(
                {
                    "id": f"{split}-{i}",
                    "audio_sha256": f"{split}-{i}",
                    "speaker": split,
                    "language": ["ar", "en"][i % 2],
                    "file": filename,
                }
            )
        (directory / "shard-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (directory / "metadata-00000.json").write_text(json.dumps(metadata))
    cfg = {
        "train_cache": "train",
        "validation_cache": "val",
        "output_dir": "out",
        "max_steps": 2,
        "batch_size": 2,
        "width": 32,
        "eval_every": 1,
        "precision": "fp32",
        "workers": 0,
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    train_speaker(tmp_path / "config.json")
    saved = torch.load(tmp_path / "out/last.pt", weights_only=True)
    assert saved["step"] == 2 and saved["dimensions"]["output_dim"] == 32


def test_qwen_talker_training_checkpoint_and_resume(tmp_path, monkeypatch):
    import importlib

    from gqtts.speaker import weights_digest

    module = importlib.import_module("gqtts.qwen_tts_train")
    qwen = tmp_path / "qwen"
    qwen.mkdir()
    (qwen / "model.safetensors").write_bytes(b"test identity; loader replaced by a tiny real model")
    metadata = {
        "shards": 1,
        "gemma_config": {"test": True},
        "include_codes": True,
        "qwen_weights": weights_digest(qwen),
    }
    for split in ["train", "val"]:
        directory = tmp_path / split
        directory.mkdir()
        records = []
        for i in range(2):
            filename = f"{i}.pt"
            torch.save(
                {
                    "frames": torch.randn(5 + i, 48),
                    "teacher": torch.randn(32),
                    "codes": torch.randint(0, 32, (3 + i, 16)),
                },
                directory / filename,
            )
            records.append(
                {
                    "id": f"{split}-{i}",
                    "audio_sha256": f"{split}-{i}",
                    "speaker": split,
                    "language": ["ar", "en"][i],
                    "text": ["مرحبا", "hello"][i],
                    "file": filename,
                }
            )
        (directory / "shard-00000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (directory / "metadata-00000.json").write_text(json.dumps(metadata))
    adapter = SpeakerAdapter(48, 32, 32)
    torch.save(
        {"dimensions": adapter.dimensions, "adapter": adapter.state_dict(), "metadata": metadata},
        tmp_path / "speaker.pt",
    )
    cfg = {
        "qwen": "qwen",
        "speaker_checkpoint": "speaker.pt",
        "train_cache": "train",
        "validation_cache": "val",
        "output_dir": "out",
        "max_steps": 2,
        "batch_size": 2,
        "eval_every": 1,
        "precision": "fp32",
        "workers": 0,
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(module, "load_qwen_tts", lambda path: tiny_tts_api())
    save = torch.save

    def keep_first(payload, destination):
        save(payload, destination)
        if payload.get("format") == "gemma-qwen-talker-v1" and payload["step"] == 1:
            save(payload, tmp_path / "step1.pt")

    monkeypatch.setattr(torch, "save", keep_first)
    module.train_qwen_tts(path)
    first = torch.load(tmp_path / "out/last.pt", weights_only=True)
    cfg["output_dir"] = "resumed"
    path.write_text(json.dumps(cfg))
    module.train_qwen_tts(path, resume=tmp_path / "step1.pt")
    resumed = torch.load(tmp_path / "resumed/last.pt", weights_only=True)
    for name, value in first["talker"].items():
        torch.testing.assert_close(value, resumed["talker"][name], rtol=0, atol=0)
