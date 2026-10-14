import json
from dataclasses import asdict

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from gemma_speech.backbone import component_from_config
from gemma_speech.checkpoint import load_checkpoint
from gemma_speech.config import ModelConfig
from gemma_speech.qwen_asr import QwenASRModel, QwenTokenizer, load_qwen_decoder
from gemma_speech.smoke import prepare_smoke, tiny_configs
from gemma_speech.train import train


def qwen_fixture(directory):
    directory.mkdir()
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    decoder = Qwen3ForCausalLM(config)
    state = {"thinker." + k: v.clone() for k, v in decoder.state_dict().items()}
    save_file(state, directory / "model.safetensors")
    # Reproduce nested config claiming tying while outer model has distinct head weights.
    text_config = config.to_dict()
    text_config["tie_word_embeddings"] = True
    (directory / "config.json").write_text(
        json.dumps({"thinker_config": {"text_config": text_config}})
    )
    vocabulary = {
        "<unk>": 0,
        "<|im_end|>": 1,
        "<|audio_pad|>": 2,
        "<asr_text>": 3,
        "language": 4,
        "Arabic": 5,
        "English": 6,
        "مرحبا": 7,
        "hello": 8,
        "user": 9,
        "assistant": 10,
    }
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|audio_pad|>", "<asr_text>"],
    )
    tok.save_pretrained(directory)
    (directory / "chat_template.json").write_text(
        json.dumps({"chat_template": "user <|audio_pad|> assistant "})
    )
    return decoder


def test_bridge_preserves_decoder_and_masking(tmp_path):
    original = qwen_fixture(tmp_path / "qwen")
    cfg = ModelConfig(
        asr_decoder="qwen",
        qwen_model_id=str(tmp_path / "qwen"),
        local_files_only=True,
        width=32,
        heads=4,
    )
    decoder = load_qwen_decoder(cfg)
    assert decoder.config.tie_word_embeddings is False
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, decoder.state_dict()[name])
    tokenizer = QwenTokenizer.from_pretrained(cfg)
    audio_cfg, _ = tiny_configs()
    model = QwenASRModel(component_from_config(audio_cfg.to_dict(), "asr"), decoder, cfg, tokenizer)
    audio, valid = model.audio_embeddings(
        torch.randn(2, 41, 128), torch.arange(41)[None, :] < torch.tensor([41, 25])[:, None]
    )
    assert valid.sum(-1).tolist() == [6, 4]
    ids = torch.tensor([[4, 5, 3, 7, 1], [4, 6, 3, 8, 1]])
    out = model(
        torch.randn(2, 41, 128),
        torch.ones(2, 41, dtype=torch.bool),
        ids,
        torch.ones_like(ids, dtype=torch.bool),
    )
    out["loss"].backward()
    assert model.bridge[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(p.grad is None for p in model.backbone.parameters())
    fused, mask, _ = model.fuse(audio, valid, left_pad=True)
    assert not mask[1, 0] and mask[1, -1]
    assert fused.shape[1] == int(mask[0].sum())
    model.eval()
    generated = model.generate_transcripts(
        torch.randn(2, 41, 128), torch.ones(2, 41, dtype=torch.bool), 3
    )
    assert generated.shape[0] == 2 and generated.shape[1] <= 3


def test_qwen_train_save_reload_without_upstream(tmp_path, monkeypatch):
    directory, configs = prepare_smoke(tmp_path / "run")
    qwen_fixture(directory / "qwen")
    config = json.loads(configs["asr"].read_text())
    config["model"].update(
        asr_decoder="qwen", qwen_model_id=str(directory / "qwen"), qwen_train_layers=1
    )
    config.update(max_steps=2, eval_every=2, max_text_tokens=12)
    configs["asr"].write_text(json.dumps(config))
    train(configs["asr"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (directory / "qwen").rename(directory / "removed-qwen")
    (directory / "tiny-backbone").rename(directory / "removed-gemma")
    model, tokenizer, payload = load_checkpoint(directory / "asr/last.pt")
    assert payload["tokenizer_kind"] == "qwen"
    assert tokenizer.decode(tokenizer.target("مرحبا", "ar")) == "مرحبا"
    assert asdict(model.config)["asr_decoder"] == "qwen"
