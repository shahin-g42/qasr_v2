from dataclasses import replace

import pytest
import torch
from transformers import EmbeddingGemma2TextModel, Gemma4AudioModel

from gemma_speech.asr import ASRModel, ctc_loss, greedy_tokens
from gemma_speech.audio import LogMel
from gemma_speech.config import ModelConfig
from gemma_speech.layers import average_upsample
from gemma_speech.smoke import tiny_configs
from gemma_speech.tts import TTSModel


def make_asr(train_last=0, upsample=1):
    audio, _ = tiny_configs()
    cfg = ModelConfig(
        width=32,
        heads=4,
        layers=2,
        dropout=0,
        train_backbone_layers=train_last,
        asr_upsample=upsample,
    )
    return ASRModel(Gemma4AudioModel(audio), cfg, 20)


@pytest.mark.parametrize("train_last", [0, 1, 2])
def test_asr_real_hf_tower_backprop_and_freezing(train_last):
    model = make_asr(train_last).train()
    result = model(
        torch.randn(2, 60, 128),
        torch.arange(60)[None] < torch.tensor([[60], [40]]),
        torch.tensor([[2, 3, 4], [3, 4, 0]]),
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    assert result["mask"].sum(-1).tolist() == [15, 10]
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.classifier.weight.grad is not None
    assert any(p.grad is not None for p in model.backbone.parameters()) == (train_last > 0)
    assert all(
        p.grad is None for p in model.backbone.component.layers[: 2 - train_last].parameters()
    )


def test_ctc_repeats_need_blank_and_masked_frames_do_not_decode():
    logits = torch.randn(1, 2, 5)
    with pytest.raises(ValueError, match="alignment impossible"):
        ctc_loss(
            logits,
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[2, 2]]),
            torch.ones(1, 2, dtype=torch.bool),
        )
    prediction = torch.nn.functional.one_hot(torch.tensor([[2, 2, 0, 2, 3, 4]]), 5).float()
    assert greedy_tokens(prediction, torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.bool)) == [
        [2, 2, 3]
    ]


def test_asr_learned_upsampling_doubles_valid_lengths():
    model = make_asr(upsample=2).eval()
    result = model(torch.randn(1, 40, 128), torch.ones(1, 40, dtype=torch.bool))
    assert result["mask"].sum().item() == 20


def test_ctc_accepts_silence_but_not_blank_targets():
    logits = torch.randn(1, 8, 5, requires_grad=True)
    loss = ctc_loss(
        logits,
        torch.ones(1, 8, dtype=torch.bool),
        torch.zeros(1, 1, dtype=torch.long),
        torch.zeros(1, 1, dtype=torch.bool),
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="exclude blank"):
        ctc_loss(
            logits,
            torch.ones(1, 8, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.long),
            torch.ones(1, 1, dtype=torch.bool),
        )


def make_tts(semantic=True, train_last=0):
    _, text = tiny_configs()
    cfg = ModelConfig(
        task="tts",
        width=32,
        heads=4,
        layers=2,
        dropout=0,
        condition_dropout=0,
        use_semantic_text=semantic,
        train_backbone_layers=train_last,
    )
    return TTSModel(EmbeddingGemma2TextModel(text) if semantic else None, cfg, 30, 2, 2)


def conditioning():
    return {
        "tokens": torch.tensor([[2, 3, 4], [5, 2, 0]]),
        "token_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
        "speaker_ids": torch.tensor([0, 1]),
        "language_ids": torch.tensor([0, 1]),
        "input_ids": torch.tensor([[2, 4, 1], [2, 7, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool),
    }


@pytest.mark.parametrize("semantic,train_last", [(True, 0), (True, 1), (False, 0)])
def test_tts_real_text_tower_backward_and_sampling(semantic, train_last):
    model = make_tts(semantic, train_last).train()
    output = model(torch.randn(2, 17, 100), torch.tensor([17, 11]), **conditioning())
    output["loss"].backward()
    assert torch.isfinite(output["loss"])
    assert model.duration[-1].weight.grad.abs().sum() > 0
    if semantic:
        assert model.semantic_projection.weight.grad.abs().sum() > 0
        assert any(p.grad is not None for p in model.backbone.parameters()) == (train_last > 0)
    model.eval()
    generated, lengths = model.sample(**conditioning(), mel_lengths=torch.tensor([17, 11]), steps=2)
    assert generated.shape == (2, 20, 100)
    assert lengths.tolist() == [17, 11]
    assert torch.isfinite(generated).all()
    assert torch.all(generated[1, 11:] == model.config.mel_mean)


def test_tts_padding_does_not_affect_loss_or_valid_output():
    model = make_tts().eval()
    mel = torch.randn(2, 17, 100)
    lengths = torch.tensor([17, 11])
    changed = mel.clone()
    changed[1, 11:] = 1e6
    torch.manual_seed(4)
    first = model(mel, lengths, **conditioning())["loss"]
    torch.manual_seed(4)
    second = model(changed, lengths, **conditioning())["loss"]
    torch.testing.assert_close(first, second)
    cond = conditioning()
    with torch.no_grad():
        x, _ = model.encode_condition(**cond)
        cond["tokens"][1, 2] = 13
        cond["input_ids"][1, 2] = 18
        y, _ = model.encode_condition(**cond)
    torch.testing.assert_close(x[1, :2], y[1, :2])


def test_duration_prediction_does_not_require_target_audio():
    model = make_tts(False).eval()
    torch.nn.init.zeros_(model.duration[-1].weight)
    torch.nn.init.zeros_(model.duration[-1].bias)
    _, lengths = model.sample(**conditioning(), steps=1)
    assert lengths.tolist() == [3, 2]
    with pytest.raises(ValueError, match="duration"):
        model.sample(**conditioning(), mel_lengths=torch.tensor([3000, 3]), max_frames=100)


def test_uniform_alignment_is_per_example():
    text = torch.tensor([[[1.0], [2], [99]], [[3.0], [4], [5]]])
    aligned = average_upsample(text, torch.tensor([2, 3]), torch.tensor([4, 6]), 6)
    assert aligned[0, :4, 0].tolist() == [1, 1, 2, 2]
    assert aligned[1, :, 0].tolist() == [3, 3, 4, 4, 5, 5]


def test_mel_contract_matches_torchaudio_when_installed():
    try:
        import torchaudio
    except (ImportError, OSError):
        pytest.skip("optional matching torchaudio binary is unavailable")
    wave = torch.randn(1, 24000)
    reference = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100, power=1
    )(wave)
    torch.testing.assert_close(
        LogMel()(wave), reference.clamp_min(1e-7).log().transpose(1, 2), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "kwargs", [{"width": 31}, {"asr_upsample": 0}, {"mel_std": 0}, {"dropout": 1}]
)
def test_invalid_model_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        replace(ModelConfig(), **kwargs)
