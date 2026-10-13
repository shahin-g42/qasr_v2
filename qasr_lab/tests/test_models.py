import copy
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("qasr")

import numpy as np
import soundfile as sf
from qasr import QASRConfig, QASRFeatureExtractor, QASRForConditionalGeneration, QASRProcessor
from qasr.configuration import CohereEncoderConfig
from qasr.modeling import QASRPoolingConv
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config

from qasr_lab.checkpoints import audit_transfer, checkpoint_identity
from qasr_lab.cli import run_inference
from qasr_lab.experiments import create_plan, load_experiment, run_experiment
from qasr_lab.io import read_json, write_json
from qasr_lab.modeling import (
    LabQASRForConditionalGeneration,
    compact_targets,
    configure_trainable,
    transcript_ctc_loss,
)
from qasr_lab.panel import canonical_sample, save_panel


def tiny_config():
    return QASRConfig(
        audio_config=CohereEncoderConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            num_mel_bins=128,
            subsampling_factor=8,
            subsampling_conv_channels=4,
            subsampling_conv_kernel_size=3,
            subsampling_conv_stride=2,
            conv_kernel_size=3,
            max_position_embeddings=256,
        ),
        text_config=Qwen3Config(
            vocab_size=128,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=256,
            pad_token_id=0,
            eos_token_id=2,
        ),
        audio_token_id=120,
        timestamp_token_id=121,
        pad_token_id=0,
        eos_token_id=2,
    )


def batch():
    ids = torch.tensor([[1, 120, 120, 5, 6]])
    labels = torch.full_like(ids, -100)
    labels[0, -2:] = ids[0, -2:]
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "input_features": torch.randn(1, 9, 128),
        "input_features_mask": torch.ones(1, 9, dtype=torch.long),
        "labels": labels,
    }


def test_default_lab_model_is_numerically_identical():
    baseline = QASRForConditionalGeneration(tiny_config()).eval()
    lab = LabQASRForConditionalGeneration(copy.deepcopy(baseline.config)).eval()
    lab.load_state_dict(baseline.state_dict(), strict=True)
    inputs = batch()
    with torch.no_grad():
        assert torch.equal(baseline(**inputs).logits, lab(**inputs).logits)


def test_masked_pooling_matches_unpadded_boundary_and_gradient():
    from qasr_lab.modeling import MaskedPoolingConv

    pool = MaskedPoolingConv(3, 2)
    short = torch.randn(1, 3, 3, requires_grad=True)
    padded = torch.cat([short.detach(), torch.full((1, 4, 3), 100.0)], dim=1).requires_grad_()
    expected, _, _ = pool(short, torch.ones(1, 3), torch.tensor([3]))
    actual, mask, lengths = pool(padded, torch.tensor([[1, 1, 1, 0, 0, 0, 0]]), torch.tensor([3]))
    assert torch.allclose(actual[:, :2], expected)
    assert lengths.tolist() == [2] and mask.sum().item() == 2
    actual[:, :2].sum().backward()
    assert torch.count_nonzero(padded.grad[:, 3:]) == 0
    legacy = QASRPoolingConv(3, 2)
    legacy.load_state_dict(pool.state_dict())
    old, _, _ = legacy(padded.detach(), mask, torch.tensor([3]))
    assert not torch.allclose(old[:, :2], expected)


def test_residual_bridge_preserves_initial_function_and_roundtrips(tmp_path):
    baseline = QASRForConditionalGeneration(tiny_config()).eval()
    config = copy.deepcopy(baseline.config)
    config.lab_bridge_dim = 7
    lab = LabQASRForConditionalGeneration(config).eval()
    lab.load_state_dict(baseline.state_dict(), strict=False)
    inputs = batch()
    with torch.no_grad():
        expected = baseline(**inputs).logits
        assert torch.equal(expected, lab(**inputs).logits)
    # A nonzero learned residual must survive saving, not get zeroed on reload.
    with torch.no_grad():
        lab.model.multi_modal_projector.lab_residual[-1].weight.fill_(0.01)
    lab.save_pretrained(tmp_path / "checkpoint")
    restored = LabQASRForConditionalGeneration.from_pretrained(tmp_path / "checkpoint").eval()
    with torch.no_grad():
        assert torch.equal(lab(**inputs).logits, restored(**inputs).logits)
    assert restored.lm_head.weight is restored.get_input_embeddings().weight


def test_ctc_repeat_feasibility_empty_targets_and_gradients():
    logits = torch.randn(3, 3, 4, requires_grad=True)
    targets = torch.tensor([[1, 1], [-100, -100], [1, 2]])
    loss, stats = transcript_ctc_loss(logits, torch.tensor([2, 2, 3]), targets)
    assert stats == {
        "ctc_rows": 3,
        "ctc_feasible_rows": 2,
        "ctc_infeasible_rows": 1,
        "ctc_empty_rows": 1,
    }
    loss.backward()
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[1]) > 0
    assert torch.count_nonzero(logits.grad[2]) > 0
    with pytest.raises(ValueError, match="No CTC-feasible"):
        transcript_ctc_loss(logits[:1], torch.tensor([2]), targets[:1])


def test_compact_targets_do_not_include_prompt_or_eos():
    class TokenizerStub:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            return [8, 9] if text else []

    assert compact_targets(TokenizerStub(), ["hello", ""], [8, 9]).tolist() == [
        [1, 2],
        [-100, -100],
    ]
    with pytest.raises(ValueError, match="absent"):
        compact_targets(TokenizerStub(), ["hello"], [8])


def test_ctc_joint_backward_checkpointing_and_reload(tmp_path):
    config = tiny_config()
    config.lab_ctc_vocabulary = [5, 6]
    config.lab_ctc_weight = 0.1
    model = LabQASRForConditionalGeneration(config).train()
    configure_trainable(model, "upper_encoder", upper_layers=1)
    model.gradient_checkpointing_enable({"use_reentrant": False})
    output = model(**batch(), lab_ctc_targets=torch.tensor([[1, 2]]), use_cache=False)
    output.loss.backward()
    assert model.lab_ctc_head.weight.grad is not None
    assert model.model.audio_tower.layers[-1].feed_forward1.linear1.weight.grad is not None
    assert model.model.audio_tower.layers[0].feed_forward1.linear1.weight.grad is None
    assert model.model.multi_modal_projector.linear_1.weight.grad is not None
    model.save_pretrained(tmp_path / "ctc")
    from transformers import AutoConfig

    assert read_json(tmp_path / "ctc/config.json")["model_type"] == "qasr_lab"
    with pytest.raises(ValueError, match="qasr_lab"):
        AutoConfig.from_pretrained(tmp_path / "ctc")
    restored = LabQASRForConditionalGeneration.from_pretrained(tmp_path / "ctc")
    assert torch.equal(restored.lab_ctc_head.weight, model.lab_ctc_head.weight)
    assert restored.lab_ctc_head.weight is not restored.lm_head.weight
    with pytest.raises(ValueError, match="transcript-only"):
        model(**batch())


def test_final_tied_embedding_overwrite_is_detected(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    write_json(source / "config.json", {"tie_word_embeddings": False})
    write_json(target / "config.json", {"model_type": "qasr", "tie_word_embeddings": True})
    save_file(
        {
            "thinker.model.embed_tokens.weight": torch.zeros(4, 3),
            "thinker.lm_head.weight": torch.ones(4, 3),
        },
        source / "model.safetensors",
    )
    save_file(
        {"model.language_model.embed_tokens.weight": torch.ones(4, 3)}, target / "model.safetensors"
    )
    result = audit_transfer(target, decoder=source)
    assert not result["passed"]
    assert result["mismatches"][0]["component"] == "decoder"
    assert result["verified_tensors"] == 2
    assert checkpoint_identity(str(target))["weights_sha256_verified"]
    with pytest.raises(ValueError, match="pinned"):
        checkpoint_identity("owner/model")


@pytest.fixture
def pilot(tmp_path):
    import qasr

    special = [
        "<unk>",
        "<pad>",
        "<|im_start|>",
        "<|im_end|>",
        "<|audio_start|>",
        "<|audio_pad|>",
        "<|audio_end|>",
        "<asr_text>",
    ]
    vocab = {
        word: i
        for i, word in enumerate(
            special
            + [
                "system",
                "user",
                "assistant",
                "language",
                "English",
                "hello",
                "world",
            ]
        )
    }
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        eos_token="<|im_end|>",
        additional_special_tokens=special[2:],
        audio_token="<|audio_pad|>",
        audio_bos_token="<|audio_start|>",
        audio_eos_token="<|audio_end|>",
    )
    template = (
        "{{ '<|im_start|>system English<|im_end|><|im_start|>user "
        "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>' }}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant ' }}{% endif %}"
    )
    processor = QASRProcessor(
        feature_extractor=QASRFeatureExtractor(feature_size=128, dither=0.0),
        tokenizer=tokenizer,
        chat_template=template,
    )
    config = tiny_config()
    config.audio_token_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")
    config.pad_token_id = config.text_config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = config.text_config.eos_token_id = tokenizer.eos_token_id
    model = QASRForConditionalGeneration(config)
    model.save_pretrained(tmp_path / "base")
    processor.save_pretrained(tmp_path / "base")
    for split in ("train", "eval"):
        samples = []
        for index in range(2):
            path = tmp_path / f"{split}{index}.wav"
            sf.write(path, np.sin(np.arange(3200) * 0.1).astype(np.float32) * 0.01, 16000)
            samples.append(
                canonical_sample(
                    {
                        "id": f"{split}{index}",
                        "audio_filepath": str(path),
                        "text": "hello",
                        "duration": 0.2,
                        "language": "en",
                    },
                    base="/",
                )
            )
        save_panel(samples, tmp_path / split)
    experiment = {
        "id": "tiny",
        "checkpoint": str(tmp_path / "base"),
        "qasr_source": str(Path(qasr.__file__).resolve().parents[2]),
        "train_panel": str(tmp_path / "train"),
        "eval_panel": str(tmp_path / "eval"),
        "output": str(tmp_path / "run"),
        "max_steps": 1,
        "warmup_steps": 0,
        "gradient_accumulation_steps": 2,
        "upper_layers": 1,
        "log_every": 1,
    }
    return tmp_path, experiment


@pytest.mark.parametrize("variant", ["control", "bridge", "ctc"])
def test_one_step_pilot_and_inference(pilot, variant):
    path, config = pilot
    if variant == "bridge":
        config["bridge_dim"] = 7
    if variant == "ctc":
        config.update(trainable="upper_encoder", ctc_weight=0.1)
    write_json(path / "experiment.json", config)
    result = run_experiment(path / "experiment.json")
    assert result["status"] == "complete" and result["steps"] == 1
    assert (path / "run/checkpoint/model.safetensors").is_file()
    if variant == "ctc":
        assert result["ctc_feasible_fraction"] == 1.0
    inference_config = {
        "id": variant,
        "checkpoint": result["checkpoint"],
        "backend": "qasr-lab",
        "qasr_source": config["qasr_source"],
        "max_new_tokens": 2,
        "warmup_samples": 1,
    }
    write_json(path / "model.json", inference_config)
    inference = run_inference(path / "model.json", path / "eval", path / "inference")
    assert inference["complete"]
    report = read_json(path / "inference/report.json")
    assert report["performance"]["timed_samples"] == 2
    if variant == "control":
        original = QASRForConditionalGeneration.from_pretrained(path / "base")
        trained = LabQASRForConditionalGeneration.from_pretrained(result["checkpoint"])
        for name, value in original.model.audio_tower.state_dict().items():
            assert torch.equal(value, trained.model.audio_tower.state_dict()[name]), name
        assert not torch.equal(
            original.model.multi_modal_projector.linear_2.weight,
            trained.model.multi_modal_projector.linear_2.weight,
        )
    else:
        inference_config["backend"] = "qasr"
        write_json(path / "wrong-backend.json", inference_config)
        with pytest.raises(ValueError, match="lab extensions"):
            run_inference(path / "wrong-backend.json", path / "eval", path / "wrong")


def test_plan_ctc_has_correct_control_and_runs_are_fresh(pilot):
    path, config = pilot
    write_json(path / "config.json", config)
    plan = create_plan(path / "config.json", path / "plan")
    ctc = next(row for row in plan["experiments"] if row["id"] == "ctc")
    assert ctc["comparison_baseline"] == "upper_encoder"
    loaded = load_experiment(ctc["config"])
    assert loaded["ctc_weight"] == 0.1
    assert loaded["max_steps"] == config["max_steps"]
    with pytest.raises(FileExistsError):
        create_plan(path / "config.json", path / "plan")
