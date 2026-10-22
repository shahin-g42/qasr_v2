import json

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from test_models import encoder_source, gqasr_source, qasr_source
from tokenizers import decoders

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.audio_types import TOKENS, extend_audio_types, upgrade
from audm.export import export_model
from audm.robustness import NonSpeechPool, RobustAugmenter
from audm.verify import verify


def artifact(tmp_path, tokenizer, variant, tied=True):
    tokenizer.backend_tokenizer.decoder = decoders.ByteLevel()
    if variant == "audflash":
        gqasr_source(tmp_path / "text", tokenizer)
        kwargs = dict(gqasr_checkpoint=tmp_path / "text")
        encoder_source(tmp_path / "encoder", "audflash_gemma")
    else:
        qasr_source(tmp_path / "text", tokenizer, tied=tied)
        kwargs = dict(qasr_checkpoint=tmp_path / "text")
        if variant == "audturbo":
            encoder_source(tmp_path / "encoder", variant)
    if variant != "audpro":
        kwargs["encoder_source"] = tmp_path / "encoder"
    export_model(variant, tmp_path / "aud", dtype="float32", **kwargs)
    return tmp_path / "aud"


@pytest.mark.parametrize(
    "variant,tied", [("audpro", True), ("audpro", False), ("audturbo", True), ("audflash", True)]
)
def test_upgrade_preserves_native_speech_and_trains_tokens(tmp_path, tokenizer, variant, tied):
    source = artifact(tmp_path, tokenizer, variant, tied)
    old = AUDForConditionalGeneration.from_pretrained(source)
    native = AUDProcessor.from_pretrained(source)
    old_vocab = native.tokenizer.get_vocab().copy()
    old_specials = set(native.tokenizer.all_special_tokens)
    waves = [
        np.random.default_rng(i).normal(0, 0.03, 3200 + i * 160).astype(np.float32)
        for i in range(4)
    ]
    original = native.prepare_training_batch(waves[:1], ["hello world"], ["ml"])
    report = upgrade(source, tmp_path / "typed")
    model = AUDForConditionalGeneration.from_pretrained(tmp_path / "typed")
    proc = AUDProcessor.from_pretrained(tmp_path / "typed")
    for token, index in old_vocab.items():
        assert proc.tokenizer.convert_tokens_to_ids(token) == index
    assert old_specials.issubset(proc.tokenizer.all_special_tokens)
    valid_ids = list(old_vocab.values())
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        torch.testing.assert_close(
            getattr(model, getter)().weight[valid_ids],
            getattr(old, getter)().weight[valid_ids],
            rtol=0,
            atol=0,
        )
    assert (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    ) == tied
    for prefix in ("aud_encoder.", "aud_projector."):
        for key, value in old.state_dict().items():
            if key.startswith(prefix):
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    for name, token in TOKENS.items():
        assert proc.tokenizer.encode(token, add_special_tokens=False) == [report["token_ids"][name]]
    batch = proc.prepare_training_batch(
        waves, ["hello world", "", "", ""], ["ml"] * 4, audio_type=list(TOKENS)
    )
    labels = batch["labels"]
    target = labels[0][labels[0] != -100]
    native_target = original["labels"][0][original["labels"][0] != -100]
    torch.testing.assert_close(target[1:], native_target)
    event = proc.decode_audio_event(target)
    assert event == dict(
        audio_type="speech", language="Malayalam", text="hello world", format_valid=True
    )
    for i, name in enumerate(list(TOKENS)[1:], 1):
        target = labels[i][labels[i] != -100]
        assert target.tolist() == [report["token_ids"][name], model.config.eos_token_id]
        assert proc.decode_audio_event(target) == dict(
            audio_type=name, language=None, text="", format_valid=True
        )
    with pytest.raises(ValueError, match="unfrozen decoder"):
        model.configure_training("adapter")
    model.configure_training("decoder")
    model.train()
    loss = model(**batch).loss
    assert torch.isfinite(loss)
    loss.backward()
    new_ids = list(report["token_ids"].values())
    assert model.get_output_embeddings().weight.grad[new_ids].abs().sum() > 0
    assert model.get_input_embeddings().weight.grad[new_ids].abs().sum() > 0
    assert all(p.grad is None for p in model.aud_encoder.parameters())
    assert verify(tmp_path / "typed")["offline_reload"]
    with pytest.raises(FileExistsError):
        upgrade(source, tmp_path / "typed")
    with pytest.raises(ValueError, match="already"):
        extend_audio_types(model, proc)


def test_typed_validation_and_noise_metadata(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audflash")
    upgrade(source, tmp_path / "typed")
    proc = AUDProcessor.from_pretrained(tmp_path / "typed")
    wav = np.zeros(3200, dtype=np.float32)
    for text, category in [("", "speech"), ("hello", "noise"), ("", None), ("", "unknown")]:
        with pytest.raises(ValueError):
            proc.prepare_training_batch([wav], [text], ["en"], audio_type=[category])
    ids = (
        [proc.config.audio_type_token_ids["noise"]]
        + tokenizer.encode("hello")
        + [proc.config.eos_token_id]
    )
    parsed = proc.decode_audio_event(ids)
    assert not parsed["format_valid"] and parsed["text"] == "hello"
    assert not proc.decode_audio_event([proc.config.audio_type_token_ids["noise"]])["format_valid"]
    assert proc.decode_audio_event(tokenizer.encode("hello"))["audio_type"] is None
    malformed = (
        [proc.config.audio_type_token_ids["speech"]]
        + tokenizer.encode("language hello<asr_text>hello", add_special_tokens=False)
        + [proc.config.eos_token_id]
    )
    assert not proc.decode_audio_event(malformed)["format_valid"]
    manifest = tmp_path / "negatives.jsonl"
    record = dict(audio_filepath="unused.wav", duration=1, text="")
    manifest.write_text(json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="audio_type"):
        NonSpeechPool(
            dict(probability=0.03, manifests=[str(manifest)]), 0.1, 30, require_audio_type=True
        )
    manifest.write_text(json.dumps(dict(record, audio_type="noise")) + "\n")
    pool = NonSpeechPool(
        dict(probability=0.03, manifests=[str(manifest)]), 0.1, 30, require_audio_type=True
    )
    assert pool.record(0, 0)["audio_type"] == "noise"
    # Token-map drift must fail reload rather than silently swapping classes.
    config = json.loads((tmp_path / "typed/config.json").read_text())
    config["audio_type_token_ids"]["noise"], config["audio_type_token_ids"]["music"] = (
        config["audio_type_token_ids"]["music"],
        config["audio_type_token_ids"]["noise"],
    )
    (tmp_path / "typed/config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatch"):
        AUDProcessor.from_pretrained(tmp_path / "typed")


def test_padded_decoder_keeps_size_and_seed_initialization(tmp_path, tokenizer):
    source = artifact(tmp_path, tokenizer, "audpro", tied=False)
    model = AUDForConditionalGeneration.from_pretrained(source)
    proc = AUDProcessor.from_pretrained(source)
    padded_size = len(proc.tokenizer) + 64
    model.aud_decoder.resize_token_embeddings(padded_size, mean_resizing=False)
    model.config.text_config.vocab_size = padded_size
    seeds = {name: proc.tokenizer.encode(name, add_special_tokens=False) for name in TOKENS}
    before = [
        model.get_input_embeddings().weight.clone(),
        model.get_output_embeddings().weight.clone(),
    ]
    report = extend_audio_types(model, proc)
    assert report["vocab_size"] == padded_size
    for old, embedding in zip(
        before, [model.get_input_embeddings(), model.get_output_embeddings()]
    ):
        for name, index in report["token_ids"].items():
            torch.testing.assert_close(embedding.weight[index], old[seeds[name]].mean(0))
        unchanged = [i for i in range(padded_size) if i not in report["token_ids"].values()]
        torch.testing.assert_close(embedding.weight[unchanged], old[unchanged], rtol=0, atol=0)


def test_category_safe_augmentation(monkeypatch):
    aug = RobustAugmenter(
        dict(clean_probability=0, effects={"white_noise": {"p": 1, "db": [-46, -46]}}),
        {"spec_augment": {"enabled": True, "p": 1}},
    )
    zero = np.zeros(3200, dtype=np.float32)
    np.testing.assert_array_equal(aug.augment_waveform(zero, audio_type="silence"), zero)
    features = torch.ones(1, 50, 128)
    torch.testing.assert_close(aug.augment_features(features, torch.ones(1, 50)), features)
    assert not aug.clean_rows
    # Inject an effect after construction so the test needs no external noise corpus.
    aug.effects = {"noise": {"p": 1}, "gain": {"p": 1}}
    calls = []
    monkeypatch.setattr(aug, "apply", lambda name, wave, cfg: calls.append(name) or wave)
    for name in ("noise", "music"):
        calls.clear()
        aug.augment_waveform(zero, audio_type=name)
        assert calls == ["gain"]


def test_category_metrics_preserve_errors():
    from audm.eval_audio_types import metrics

    def row(expected, prediction, text="", valid=True):
        return dict(
            expected=expected,
            event=dict(audio_type=prediction, text=text, format_valid=valid),
            hit_token_cap=not valid,
        )

    report = metrics(
        [
            row("speech", "speech", "hello"),
            row("speech", "noise"),
            row("noise", "noise", "hallucinated", False),
            row("noise", None, "oops", False),
            row("music", "music"),
            row("silence", "silence"),
        ]
    )
    assert report["accuracy"] == 4 / 6
    assert report["invalid_format_rate"] == 2 / 6
    assert report["missed_speech_rate"] == 0.5
    assert report["non_speech_text_rate"] == 0.5
    assert report["confusion"]["noise"]["invalid"] == 1
    assert report["per_class"]["noise"]["precision"] == 0.5
    assert report["token_cap_hits"] == 2


def test_decoder_warmup_checkpoint_and_category_evaluation(tmp_path, tokenizer):
    from test_training import data_recipe

    from audm.eval_audio_types import evaluate
    from audm.train import preflight, run

    source = artifact(tmp_path, tokenizer, "audflash")
    upgrade(source, tmp_path / "typed")
    sf.write(tmp_path / "silence.wav", np.zeros(3200), 16000)
    manifest = tmp_path / "negative.jsonl"
    manifest.write_text(
        json.dumps(dict(audio_filepath="silence.wav", duration=0.2, text="", audio_type="silence"))
        + "\n"
    )
    recipe = tmp_path / "decoder.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                data_recipe=str(data_recipe(tmp_path)),
                stage="decoder",
                robustness=dict(non_speech=dict(probability=0.5, manifests=[str(manifest)])),
                training=dict(
                    max_duration_seconds=30,
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    per_device_eval_batch_size=1,
                    expected_world_size=1,
                    max_steps=1,
                    warmup_steps=0,
                    save_steps=1,
                    eval_steps=1,
                    report_to="none",
                    dataloader_num_workers=0,
                    dataloader_pin_memory=False,
                    bf16=False,
                    tf32=False,
                    deepspeed=None,
                    gradient_checkpointing=True,
                    label_smoothing=0.0,
                    eval_log_samples=0,
                    eval_wer_samples_per_language=0,
                ),
            )
        )
    )
    report = preflight(recipe, tmp_path / "typed", tmp_path / "run", scan=True)
    assert report["updates_per_coverage_epoch"] == 2  # Warm-up intentionally stops earlier.
    original = AUDForConditionalGeneration.from_pretrained(tmp_path / "typed")
    run(recipe, tmp_path / "typed", tmp_path / "run")
    trained = AUDForConditionalGeneration.from_pretrained(tmp_path / "run")
    proc = AUDProcessor.from_pretrained(tmp_path / "run/checkpoint-1")
    assert proc.config.audio_type_version == 1
    for key, value in original.aud_encoder.state_dict().items():
        torch.testing.assert_close(trained.aud_encoder.state_dict()[key], value, rtol=0, atol=0)
    new_ids = list(proc.config.audio_type_token_ids.values())
    assert not torch.equal(
        original.get_input_embeddings().weight[new_ids],
        trained.get_input_embeddings().weight[new_ids],
    )
    result = evaluate(tmp_path / "run", manifest, tmp_path / "eval.json", max_new_tokens=2, limit=1)
    assert result["samples"] == 1 and result["per_class"]["silence"]["support"] == 1
