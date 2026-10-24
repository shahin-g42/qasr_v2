import copy
import json

import numpy as np
import pytest
import torch
import yaml
from test_audio_types import artifact
from test_training import data_recipe
from tokenizers import Tokenizer, models, pre_tokenizers

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.collator import AUDDataCollator
from audm.ctc import (
    attach_head,
    auxiliary_loss,
    eligibility,
    export,
    inference_export,
    train_tokenizer,
)
from audm.train import run


def small_tokenizer():
    tok = Tokenizer(
        models.WordLevel(
            {"<ctc_blank>": 0, "<ctc_unk>": 1, "hello": 2, "world": 3}, unk_token="<ctc_unk>"
        )
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return tok


def test_ctc_repeats_padding_and_accumulation_gradients():
    torch.manual_seed(42)
    states = torch.randn(4, 9, 5, requires_grad=True)
    head = torch.nn.Linear(5, 8, bias=False)
    targets = torch.tensor([[2, 2, 3], [3, 4, -100], [2, 2, 2], [-100, -100, -100]])
    mask = torch.arange(9)[None] < torch.tensor([9, 4, 3, 9])[:, None]
    assert eligibility(targets, mask.sum(-1)).tolist() == [True, True, False, False]
    full, stats = auxiliary_loss(states, head, mask, targets)
    assert stats[1:].tolist() == [2, 3, 4]
    full.backward()
    other = states.detach().clone().requires_grad_()
    other_head = copy.deepcopy(head)
    other_head.zero_grad()
    pieces = [
        auxiliary_loss(other[i : i + 1], other_head, mask[i : i + 1], targets[i : i + 1], 2)[0]
        for i in range(4)
    ]
    sum(pieces).backward()
    torch.testing.assert_close(sum(pieces), full)
    torch.testing.assert_close(other.grad, states.grad)
    torch.testing.assert_close(other_head.weight.grad, head.weight.grad)
    assert not states.grad[2:].any()
    assert not states.grad[1, 4:].any()
    with pytest.raises(ValueError, match="blank"):
        auxiliary_loss(states[:1], head, mask[:1], torch.tensor([[0]]))


def test_ctc_export_keeps_all_pretrained_weights_and_logits(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audflash")
    original = AUDForConditionalGeneration.from_pretrained(source).eval()
    proc = AUDProcessor.from_pretrained(source)
    tok = tmp_path / "ctc"
    tok.mkdir()
    small_tokenizer().save(str(tok / "tokenizer.json"))
    report = export(source, tok, tmp_path / "new")
    assert report["stride"] == 1
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "new").eval()
    processor = AUDProcessor.from_pretrained(tmp_path / "new")
    assert processor.tokenizer.get_vocab() == proc.tokenizer.get_vocab()
    assert model.config.ctc_vocab_size == 4
    assert model.aud_ctc_head.in_features == model.config.audio_config.output_proj_dims
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    batch = processor.prepare_training_batch([np.zeros(16000)], ["hello world"], ["en"])
    with torch.no_grad():
        torch.testing.assert_close(original(**batch).logits, model(**batch).logits, rtol=0, atol=0)
    with pytest.raises(ValueError, match="already"):
        attach_head(model, processor, small_tokenizer())
    broken = json.loads((tmp_path / "new/ctc_tokenizer.json").read_text())
    broken["model"]["vocab"]["world"] = 5
    (tmp_path / "new/ctc_tokenizer.json").write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="mismatch"):
        AUDProcessor.from_pretrained(tmp_path / "new")


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("stage", ["decoder", "full"])
def test_ctc_stage_backprop_and_nonspeech_targets(tmp_path, tokenizer, stage, version):
    source = artifact(tmp_path, tokenizer, "audflash")
    model = AUDForConditionalGeneration.from_pretrained(source)
    processor = AUDProcessor.from_pretrained(source)
    from audm.audio_types import extend_audio_types

    extend_audio_types(model, processor)
    attach_head(model, processor, small_tokenizer() if version == 1 else None)
    model.configure_training(stage)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    collator = AUDDataCollator(
        processor=processor,
        max_target_length=512,
        language="en",
        min_audio_seconds=0.1,
        max_audio_seconds=30,
    )
    collator._load_waveform = lambda _: np.random.default_rng(42).normal(0, 0.01, 16000)
    features = [
        dict(text="hello world", language="en"),
        dict(text="", audio_type="noise", language="en"),
    ]
    batch = collator(features)
    expected = (
        [2, 3]
        if version == 1
        else processor.tokenizer.encode("hello world", add_special_tokens=False)
    )
    assert batch["ctc_targets"].tolist() == [expected, [-100] * len(expected)]
    loss = model(**batch, loss_only=True, ctc_weight=0.1).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.aud_ctc_head.weight.grad).all()
    assert model.aud_ctc_head.weight.grad.abs().sum() > 0
    assert any(p.grad is not None for p in model.aud_encoder.parameters()) == (stage == "full")
    assert model.aud_projector.up.weight.grad is not None
    assert model.aud_ctc_totals[1:].tolist() == [1, 1, 2]


@pytest.mark.parametrize("version", [1, 2])
def test_tokenizer_and_adaptation_to_full_workflow(tmp_path, tokenizer, version):
    source = artifact(tmp_path, tokenizer, "audflash")
    data = data_recipe(tmp_path)
    training = dict(
        model_name_or_path=str(source),
        output_dir=str(tmp_path / "adaptation"),
        max_duration_seconds=30,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=2,
        expected_world_size=1,
        num_train_epochs=1,
        max_steps=-1,
        warmup_steps=0,
        learning_rate=1e-4,
        save_steps=2,
        eval_steps=2,
        logging_steps=1,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        bf16=False,
        tf32=False,
        deepspeed=None,
        eval_log_samples=0,
        eval_wer_samples_per_language=0,
        ctc_loss_weight=0.1,
    )
    recipe = dict(
        data_recipe=str(data),
        stage="decoder",
        training=training,
        ctc={"warmup_steps": 0},
        efficiency={"index_cache": str(tmp_path / "cache"), "duration_bucketing": True},
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe))
    report = train_tokenizer(path, tmp_path / "ctc", vocab_size=300)
    assert report["counts"]["ar:transcripts"] == 2
    tok = Tokenizer.from_file(str(tmp_path / "ctc/tokenizer.json"))
    for text in ("മലയാളം", "中文", "العربية", "हिन्दी", "héllo"):
        encoded = tok.encode(text, add_special_tokens=False)
        assert 1 not in encoded.ids
        assert tok.decode(encoded.ids) == text
    if version == 2:
        recipe["ctc"]["tokenizer"] = "decoder"
    export(source, tmp_path / "ctc" if version == 1 else None, tmp_path / "new")
    before = AUDForConditionalGeneration.from_pretrained(tmp_path / "new")
    training["model_name_or_path"] = str(tmp_path / "new")
    path.write_text(yaml.safe_dump(recipe))
    run(path)
    adapted = AUDForConditionalGeneration.from_pretrained(tmp_path / "adaptation")
    assert not torch.equal(before.aud_ctc_head.weight, adapted.aud_ctc_head.weight)
    for name, tensor in before.aud_encoder.state_dict().items():
        torch.testing.assert_close(tensor, adapted.aud_encoder.state_dict()[name], rtol=0, atol=0)
    recipe["stage"] = "full"
    training.update(
        model_name_or_path=str(tmp_path / "adaptation"), output_dir=str(tmp_path / "full")
    )
    path.write_text(yaml.safe_dump(recipe))
    run(path)
    final = AUDForConditionalGeneration.from_pretrained(tmp_path / "full")
    assert any(
        not torch.equal(tensor, final.aud_encoder.state_dict()[name])
        for name, tensor in adapted.aud_encoder.state_dict().items()
    )
    assert all(torch.isfinite(p).all() for p in final.parameters())
    assert json.loads((tmp_path / "full/eval_results.json").read_text())["eval_loss"] > 0
    stats = [
        json.loads(line) for line in (tmp_path / "full/performance.jsonl").read_text().splitlines()
    ]
    assert sum(row["ctc"]["eligible"] for row in stats) == 4


def test_next_recipe_keeps_source_objective_and_batch_contract(tmp_path, tokenizer):
    from audm.next_run import prepare
    from audm.training_config import load_recipe

    source = artifact(tmp_path, tokenizer, "audflash")
    recipe_path = tmp_path / "old.yaml"
    recipe_path.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage="full",
                training=dict(
                    model_name_or_path=str(source),
                    output_dir=str(tmp_path / "old"),
                    per_device_train_batch_size=2,
                    gradient_accumulation_steps=1,
                    max_duration_seconds=30,
                ),
            )
        )
    )
    _, _, receipt = load_recipe(recipe_path)
    receipt["resolved"]["eval_manifest"]["zh"] = receipt["resolved"]["eval_manifest"]["en"]
    (source / "training_recipe.json").write_text(json.dumps(receipt))
    (source / "trainer_state.json").write_text(json.dumps({"global_step": 3700}))
    report = prepare(source, tmp_path / "next")
    assert report["source_step"] == 3700 and report["stride"] == 1
    for name, stage, expected_source in (
        ("adaptation", "decoder", "export"),
        ("full", "full", "adaptation"),
    ):
        cfg, actual_stage, _ = load_recipe(tmp_path / "next" / f"{name}.yaml")
        assert actual_stage == stage
        assert cfg.ctc_loss_weight == 0.1
        assert cfg.aud_ctc["tokenizer"] == "decoder"
        assert (
            cfg.per_device_train_batch_size
            * cfg.gradient_accumulation_steps
            * cfg.expected_world_size
            == 640
        )
        assert cfg.per_device_eval_batch_size == 40
        assert cfg.dataloader_num_workers == 4
        assert "zh" not in cfg.eval_manifest
        assert cfg.train_manifest == receipt["resolved"]["train_manifest"]
        assert cfg.model_name_or_path == str(tmp_path / "next" / expected_source)
        assert cfg.resume_from_checkpoint is None
        assert cfg.aud_efficiency["distributed_wer"]
        if name == "full":
            assert cfg.num_train_epochs == 3 and cfg.max_steps == -1
            assert cfg.eval_steps == cfg.save_steps == 3700
    with pytest.raises(FileExistsError):
        prepare(source, tmp_path / "next")


def test_decoder_ctc_blank_accepts_zero_and_matches_torch_loss():
    torch.manual_seed(42)
    hidden = torch.randn(2, 7, 3, requires_grad=True)
    head = torch.nn.Linear(3, 5, bias=False)
    targets = torch.tensor([[0, 0, 2], [1, 2, -100]])
    mask = torch.arange(7)[None] < torch.tensor([7, 5])[:, None]
    loss, _ = auxiliary_loss(hidden, head, mask, targets, blank_id=4)
    loss.backward()
    expected_hidden = hidden.detach().clone().requires_grad_()
    expected_head = copy.deepcopy(head)
    expected_head.zero_grad()
    logits = expected_head(expected_hidden).log_softmax(-1).transpose(0, 1)
    expected = torch.nn.functional.ctc_loss(
        logits,
        targets[targets != -100],
        mask.sum(-1),
        targets.ne(-100).sum(-1),
        blank=4,
        reduction="mean",
    )
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(head.weight.grad, expected_head.weight.grad)
    for bad in (4, -1, 5):
        with pytest.raises(ValueError, match="blank"):
            auxiliary_loss(hidden[:1], head, mask[:1], torch.tensor([[bad]]), blank_id=4)


def test_decoder_ctc_export_and_inference_roundtrip(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audflash")
    original = AUDForConditionalGeneration.from_pretrained(source).eval()
    original_processor = AUDProcessor.from_pretrained(source)
    report = export(source, output=tmp_path / "ctc")
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "ctc").eval()
    processor = AUDProcessor.from_pretrained(tmp_path / "ctc")
    assert report["ctc_version"] == 2 and report["ctc_tokenizer"] == "decoder"
    assert model.config.ctc_blank_id == original.config.text_config.vocab_size
    assert model.config.ctc_vocab_size == original.config.text_config.vocab_size + 1
    assert processor.ctc_tokenizer is processor.tokenizer
    assert processor.tokenizer.get_vocab() == original_processor.tokenizer.get_vocab()
    assert not (tmp_path / "ctc/ctc_tokenizer.json").exists()
    assert processor.encode_ctc("hello world") == processor.tokenizer.encode(
        "hello world", add_special_tokens=False
    )
    with pytest.raises(ValueError, match="reserved"):
        processor.encode_ctc("hello <|im_end|>")
    with pytest.raises(ValueError, match="reserved"):
        processor.encode_ctc("hello", transcript_ids=[model.config.ctc_blank_id])
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    batch = processor.apply_transcription_request([np.zeros(16000)], language="en")
    batch = {k: v for k, v in batch.items() if k in processor.model_input_names}
    # Exercise processor batch padding before save; transient tokenizer state
    # must not invalidate the stable token-ID/segmentation fingerprint.
    processor.tokenizer(["hello", "hello world"], padding=True)
    processor.save_pretrained(tmp_path / "ctc")
    AUDProcessor.from_pretrained(tmp_path / "ctc")
    model.aud_ctc_head.forward = lambda *a, **k: pytest.fail("CTC executed during inference")
    with torch.no_grad():
        expected_logits = model(**batch).logits
        expected_tokens = model.generate(**batch, max_new_tokens=3, do_sample=False)
    report = inference_export(tmp_path / "ctc", tmp_path / "deployment")
    deployed = AUDForConditionalGeneration.from_pretrained(tmp_path / "deployment").eval()
    deployed_processor = AUDProcessor.from_pretrained(tmp_path / "deployment")
    assert report["removed_ctc_parameters"] == model.aud_ctc_head.weight.numel()
    assert deployed.aud_ctc_head is None and deployed.config.ctc_version == 0
    assert deployed.config.ctc_blank_id is None
    assert deployed_processor.ctc_tokenizer is None
    assert deployed_processor.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    assert set(deployed.state_dict()) == set(original.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(deployed(**batch).logits, expected_logits, rtol=0, atol=0)
        assert torch.equal(
            deployed.generate(**batch, max_new_tokens=3, do_sample=False), expected_tokens
        )
    from audm.verify import verify

    assert verify(tmp_path / "deployment")["checksums"] == "verified"
    with pytest.raises(ValueError, match="no auxiliary"):
        inference_export(tmp_path / "deployment", tmp_path / "again")
    with pytest.raises(FileExistsError):
        export(source, output=tmp_path / "ctc")
    processor.tokenizer.add_tokens(["new_token_invalidates_fixed_ctc"])
    with pytest.raises(ValueError, match="mismatch"):
        processor.save_pretrained(tmp_path / "changed")


def test_decoder_ctc_rejects_changed_tokenizer_on_reload(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audflash")
    export(source, output=tmp_path / "ctc")
    path = tmp_path / "ctc/tokenizer.json"
    data = json.loads(path.read_text())
    vocab = data["model"]["vocab"]
    a, b = next(
        (a, b)
        for a in vocab
        for b in vocab
        if a != b and not a.startswith("<") and not b.startswith("<")
    )
    vocab[a], vocab[b] = vocab[b], vocab[a]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="mismatch"):
        AUDProcessor.from_pretrained(tmp_path / "ctc")


def test_decoder_ctc_recipe_rejects_legacy_ctc():
    from types import SimpleNamespace

    from audm.ctc import validate_training_tokenizer

    recipe = SimpleNamespace(ctc_loss_weight=0.1, aud_ctc={"tokenizer": "decoder"})
    with pytest.raises(ValueError, match="new export"):
        validate_training_tokenizer(recipe, SimpleNamespace(ctc_version=1))
    validate_training_tokenizer(recipe, SimpleNamespace(ctc_version=2))
