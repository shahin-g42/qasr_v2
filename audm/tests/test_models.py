import numpy as np
import pytest
import torch
from conftest import CHAT_TEMPLATE
from qasr.configuration import QASRConfig
from qasr.feature_extraction import QASRFeatureExtractor
from qasr.modeling import QASRForConditionalGeneration
from qasr.processing import QASRProcessor
from safetensors.torch import save_file
from transformers import (
    Gemma4AudioConfig,
    Gemma4AudioFeatureExtractor,
    Gemma4AudioModel,
    Qwen3Config,
    Qwen3ForCausalLM,
    WhisperFeatureExtractor,
)
from transformers.models.moonshine_streaming.configuration_moonshine_streaming import (
    MoonshineStreamingEncoderConfig,
)
from transformers.models.moonshine_streaming.modeling_moonshine_streaming import (
    MoonshineStreamingEncoder,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoderConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    _get_feat_extract_output_lengths,
)

from audm import AUDForConditionalGeneration, AUDProcessor
from audm.export import WeightSource, build_export, export_model, write_json
from audm.verify import verify


def text_config(tokenizer, tied=True):
    return dict(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=tied,
    )


def qasr_source(path, tokenizer, depth=1, stride=1, ctc=False, tied=True):
    config = QASRConfig(
        audio_config=dict(
            model_type="parakeet_encoder",
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_mel_bins=128,
            subsampling_factor=8,
            subsampling_conv_channels=4,
            conv_kernel_size=3,
            max_position_embeddings=512,
        ),
        text_config=text_config(tokenizer, tied),
        audio_token_id=tokenizer.convert_tokens_to_ids("<|audio_pad|>"),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=tied,
        projector_depth=depth,
        projector_pool_stride=stride,
    )
    model = QASRForConditionalGeneration(config).eval()
    if ctc:
        model.add_ctc_head()
    model.save_pretrained(path)
    processor = QASRProcessor(
        QASRFeatureExtractor(return_attention_mask=True),
        tokenizer,
        CHAT_TEMPLATE,
        projector_pool_stride=stride,
    )
    processor.save_pretrained(path)
    return model, processor


def encoder_source(path, variant):
    path.mkdir()
    if variant == "audturbo":
        config = Qwen3OmniMoeAudioEncoderConfig(
            num_mel_bins=128,
            d_model=16,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=32,
            output_dim=16,
            downsample_hidden_size=4,
            max_source_positions=128,
            n_window=50,
            n_window_infer=800,
        )
        model = Qwen3OmniMoeAudioEncoder(config).eval()
        data, prefix = (
            {"thinker_config": {"audio_config": config.to_dict()}},
            "thinker.audio_tower.",
        )
        WhisperFeatureExtractor(feature_size=128, return_attention_mask=True).save_pretrained(path)
    elif variant == "audflash_gemma":
        config = Gemma4AudioConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            output_proj_dims=16,
            subsampling_conv_channels=[128, 2],
            attention_chunk_size=4,
            attention_context_left=5,
            use_clipped_linears=True,
        )
        model = Gemma4AudioModel(config).eval()
        data, prefix = (
            {"model_type": "gemma4", "audio_config": config.to_dict()},
            "model.audio_tower.",
        )
        frontend = Gemma4AudioFeatureExtractor()
        # Match the upstream multimodal processor layout (no standalone frontend file).
        write_json(
            path / "processor_config.json",
            {"feature_extractor": __import__("json").loads(frontend.to_json_string())},
        )
    else:
        config = MoonshineStreamingEncoderConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            sliding_windows=[[2, 1], [2, 0]],
            frame_ms=5.0,
        )
        model = MoonshineStreamingEncoder(config).eval()
        data, prefix = {"encoder_config": config.to_dict()}, "model.encoder."
    write_json(path / "config.json", data)
    weights = {prefix + k: v for k, v in model.state_dict().items()}
    # Unrelated upstream decoder weights must not be imported.
    weights["model.unused_decoder.weight"] = torch.ones(4, 4)
    save_file(weights, path / "model.safetensors")
    return model


def gqasr_source(path, tokenizer, tied=True):
    path.mkdir()
    config = Qwen3Config(**text_config(tokenizer, tied))
    decoder = Qwen3ForCausalLM(config)
    write_json(
        path / "config.json",
        dict(
            model_type="gqasr",
            decoder_config=config.to_dict(),
            prompt_prefix=[2, 4],
            prompt_suffix=[6, 2],
            eos_id=tokenizer.eos_token_id,
            audio_stride=2,
            bridge_width=384,
        ),
    )
    weights = {"gqasr_decoder." + k: v.clone() for k, v in decoder.state_dict().items()}
    if tied:
        del weights["gqasr_decoder.model.embed_tokens.weight"]
    weights["gqasr_acoustic.core.weight"] = torch.ones(4, 4)
    bridge = torch.nn.Sequential(
        torch.nn.LayerNorm(16), torch.nn.Linear(16, 384), torch.nn.GELU(), torch.nn.Linear(384, 16)
    )
    weights.update({"gqasr_bridge." + k: v for k, v in bridge.state_dict().items()})
    save_file(weights, path / "model.safetensors")
    tokenizer.save_pretrained(path / "tokenizer")
    return decoder


@pytest.mark.parametrize("tied", [True, False])
def test_pro_exact_repack_with_trained_bridge_pool_ctc(tmp_path, tokenizer, tied):
    old, processor = qasr_source(
        tmp_path / "source", tokenizer, depth=2, stride=2, ctc=True, tied=tied
    )
    export_model("audpro", tmp_path / "aud", qasr_checkpoint=tmp_path / "source", dtype="float32")
    new = AUDForConditionalGeneration.from_pretrained(
        tmp_path / "aud", local_files_only=True
    ).eval()
    proc = AUDProcessor.from_pretrained(tmp_path / "aud")
    waves = [np.linspace(-0.2, 0.2, n, dtype=np.float32) for n in (3200, 4960)]
    old_inputs = dict(
        processor.prepare_training_batch(
            waves,
            ["hello", "hello world"],
            ["en", "ar"],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
    )
    new_inputs = dict(proc.prepare_training_batch(waves, ["hello", "hello world"], ["en", "ar"]))
    for key in new_inputs:
        torch.testing.assert_close(new_inputs[key], old_inputs[key], rtol=0, atol=0)
    with torch.no_grad():
        a, b = old(**old_inputs), new(**new_inputs)
    torch.testing.assert_close(
        a.logits[old_inputs["attention_mask"].bool()],
        b.logits[new_inputs["attention_mask"].bool()],
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(a.loss, b.loss)
    torch.testing.assert_close(old.model.ctc_head.weight, new.aud_ctc_head.weight)
    new.configure_training("full")
    assert new.aud_ctc_head.weight.requires_grad is False
    assert (
        new.get_input_embeddings().weight is new.get_output_embeddings().weight
        if tied
        else new.get_input_embeddings().weight is not new.get_output_embeddings().weight
    )
    # Single-row generation parity avoids irrelevant padded-position logits.
    batch = proc.apply_transcription_request(waves[:1], language="en")
    assert torch.equal(
        old.generate(**batch, max_new_tokens=3), new.generate(**batch, max_new_tokens=3)
    )
    report = verify(tmp_path / "aud")
    assert report["offline_reload"]
    assert len(list((tmp_path / "aud").glob("*.safetensors"))) == 1
    assert all(k.startswith("aud_") for k in new.state_dict())


@pytest.mark.parametrize("variant", ["audturbo", "audflash"])
def test_hybrid_preload_forward_backward_offline(tmp_path, tokenizer, variant):
    native = encoder_source(tmp_path / "encoder", variant)
    if variant == "audturbo":
        source, _ = qasr_source(tmp_path / "text", tokenizer)
        kwargs = {"qasr_checkpoint": tmp_path / "text"}
    else:
        source = gqasr_source(tmp_path / "text", tokenizer)
        kwargs = {"gqasr_checkpoint": tmp_path / "text"}
    export_model(
        variant, tmp_path / "aud", encoder_source=tmp_path / "encoder", dtype="float32", **kwargs
    )
    model = AUDForConditionalGeneration.from_pretrained(
        tmp_path / "aud", local_files_only=True
    ).eval()
    for key, weight in native.state_dict().items():
        torch.testing.assert_close(model.aud_encoder.state_dict()[key], weight, rtol=0, atol=0)
    expected = (
        source.model.language_model.embed_tokens.weight
        if variant == "audturbo"
        else source.model.embed_tokens.weight
    )
    torch.testing.assert_close(
        model.aud_decoder.model.embed_tokens.weight, expected, rtol=0, atol=0
    )
    processor = AUDProcessor.from_pretrained(tmp_path / "aud")
    waves = [np.random.default_rng(7).normal(0, 0.01, n).astype(np.float32) for n in (3200, 4961)]
    batch = processor.prepare_training_batch(waves, ["hello", "world"], ["en", "ml"])
    projected, lengths = model.encode_audio(batch["input_features"], batch["input_features_mask"])
    mask = batch["input_features_mask"].bool()
    if variant == "audturbo":
        packed = batch["input_features"].transpose(1, 2)[mask].T
        native_states = native(packed, feature_lens=mask.sum(-1)).last_hidden_state
        # The new residual adapter starts exactly at identity.
        torch.testing.assert_close(projected, native_states, rtol=0, atol=0)
    else:
        # Native padded batches overcount tail frames. Compare against native
        # single utterances on their own exact frame grid instead.
        offset = 0
        for i, wave in enumerate(waves):
            frames = len(wave) // 80
            raw = torch.as_tensor(wave[: frames * 80])[None]
            native_out = native(
                input_values=raw, attention_mask=torch.ones_like(raw, dtype=torch.bool)
            )
            expected = model.aud_projector(native_out.last_hidden_state[0])
            count = int(lengths[i])
            torch.testing.assert_close(
                projected[offset : offset + count], expected, atol=1e-6, rtol=1e-5
            )
            offset += count
    assert lengths.tolist() == processor.audio_token_lengths(mask).tolist()
    model.configure_training("adapter")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    loss = model(**batch).loss
    assert loss.isfinite()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.aud_projector.parameters()
    )
    assert all(p.grad is None for p in model.aud_encoder.parameters())
    assert all(p.grad is None for p in model.aud_decoder.parameters())
    assert verify(tmp_path / "aud")["smoke"]["quality_evaluation"] is False


def test_omni_chunk_boundary_lengths(tmp_path, tokenizer):
    qasr_source(tmp_path / "text", tokenizer)
    encoder_source(tmp_path / "audio", "audturbo")
    text, audio = WeightSource(tmp_path / "text"), WeightSource(tmp_path / "audio")
    try:
        model, _ = build_export("audturbo", text, audio, dtype="float32")
        lengths = torch.tensor([99, 100, 101, 200])
        mask = torch.arange(200)[None] < lengths[:, None]
        hidden, actual = model.encode_audio(torch.randn(4, 128, 200), mask)
        assert actual.tolist() == [13, 13, 14, 26]
        assert hidden.shape[0] == int(_get_feat_extract_output_lengths(lengths).sum())
        # A different sample's count cannot compensate for a wrong per-row placeholder count.
        bad = torch.full((4, 20), 5)
        with pytest.raises(ValueError, match="Per-row"):
            model(input_ids=bad, input_features=torch.randn(4, 128, 200), input_features_mask=mask)
    finally:
        text.close()
        audio.close()


def test_missing_weights_and_overwrite_fail(tmp_path, tokenizer):
    qasr_source(tmp_path / "source", tokenizer)
    with pytest.raises(FileExistsError):
        export_model("audpro", tmp_path / "source", qasr_checkpoint=tmp_path / "source")
    source = WeightSource(tmp_path / "source")
    del source.weight_map["model.multi_modal_projector.linear_1.weight"]
    try:
        with pytest.raises(KeyError, match="missing pretrained"):
            build_export("audpro", source)
    finally:
        source.close()


def test_corruption_detected(tmp_path, tokenizer):
    qasr_source(tmp_path / "source", tokenizer)
    export_model("audpro", tmp_path / "aud", qasr_checkpoint=tmp_path / "source", dtype="float32")
    config = tmp_path / "aud" / "config.json"
    config.write_text(config.read_text() + " ")
    with pytest.raises(ValueError, match="Checksum"):
        verify(tmp_path / "aud", smoke=False)


@pytest.mark.parametrize("variant", ["audpro", "audturbo", "audflash"])
def test_default_bf16_export_and_cpu_inference(tmp_path, tokenizer, variant):
    if variant == "audflash":
        gqasr_source(tmp_path / "text", tokenizer)
        kwargs = {"gqasr_checkpoint": tmp_path / "text"}
    else:
        qasr_source(tmp_path / "text", tokenizer)
        kwargs = {"qasr_checkpoint": tmp_path / "text"}
    if variant != "audpro":
        encoder_source(tmp_path / "encoder", variant)
        kwargs["encoder_source"] = tmp_path / "encoder"
    export_model(variant, tmp_path / "aud", **kwargs)
    assert verify(tmp_path / "aud")["smoke"]["device"] == "cpu"


def test_remote_index_downloads_only_required_encoder_shard(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import audm.export as exporter

    write_json(tmp_path / "config.json", {"model_type": "test"})
    write_json(
        tmp_path / "model.safetensors.index.json",
        {
            "weight_map": {
                "thinker.audio_tower.test.weight": "audio.safetensors",
                "thinker.language_model.large.weight": "decoder.safetensors",
            }
        },
    )
    save_file({"thinker.audio_tower.test.weight": torch.ones(2)}, tmp_path / "audio.safetensors")
    reads = []

    class API:
        def model_info(self, repo_id, revision):
            return SimpleNamespace(
                sha="immutable-test-sha",
                siblings=[SimpleNamespace(rfilename=p.name) for p in tmp_path.iterdir()]
                + [SimpleNamespace(rfilename="decoder.safetensors")],
            )

    def download(repo_id, filename, revision):
        assert revision == "immutable-test-sha"
        reads.append(filename)
        return str(tmp_path / filename)

    monkeypatch.setattr(exporter, "HfApi", API)
    monkeypatch.setattr(exporter, "hf_hub_download", download)
    source = WeightSource("example/encoder")
    try:
        torch.testing.assert_close(source.tensor("thinker.audio_tower.test.weight"), torch.ones(2))
        assert "decoder.safetensors" not in reads
        assert source.revision == "immutable-test-sha"
    finally:
        source.close()


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_gemma_flash_pretrained_bridge_pooling_and_reload(tmp_path, tokenizer, dtype, stride):
    gqasr_source(tmp_path / "text", tokenizer)
    native = encoder_source(tmp_path / "encoder", "audflash_gemma")
    export_model(
        "audflash",
        tmp_path / "aud",
        gqasr_checkpoint=tmp_path / "text",
        encoder_source=tmp_path / "encoder",
        dtype=dtype,
        flash_audio_stride=stride,
    )
    new = AUDForConditionalGeneration.from_pretrained(
        tmp_path / "aud", dtype=getattr(torch, dtype)
    ).eval()
    proc = AUDProcessor.from_pretrained(tmp_path / "aud")
    assert new.config.flash_encoder == "gemma4"
    assert new.config.flash_audio_stride == stride
    assert new.config.requires_alignment
    assert all(k.startswith("aud_") for k in new.state_dict())
    native.to(getattr(torch, dtype))
    for key, value in native.state_dict().items():
        torch.testing.assert_close(new.aud_encoder.state_dict()[key], value, rtol=0, atol=0)
    source = WeightSource(tmp_path / "text")
    for name, number in (("norm", "0"), ("down", "1"), ("up", "3")):
        for suffix in ("weight", "bias"):
            torch.testing.assert_close(
                new.state_dict()[f"aud_projector.{name}.{suffix}"],
                source.tensor(f"gqasr_bridge.{number}.{suffix}").to(getattr(torch, dtype)),
                rtol=0,
                atol=0,
            )
    source.close()
    waves = [
        np.random.default_rng(i).normal(0, 0.03, n).astype(np.float32)
        for i, n in enumerate((3200, 4961, 8000))
    ]
    batch = proc.prepare_training_batch(waves, ["hello"] * 3, ["en", "ar", "ml"])
    with torch.no_grad():
        output = native(
            input_features=batch["input_features"].to(getattr(torch, dtype)),
            attention_mask=batch["input_features_mask"],
            return_dict=True,
        )
        valid = output.attention_mask.bool()
        hidden = new.aud_projector(output.last_hidden_state).masked_fill(~valid[..., None], 0)
        # Independent per-row reference: GQASR valid-frame averaging after projection.
        expected = []
        for row, mask in zip(hidden, valid, strict=True):
            row = row[mask]
            expected.extend(row[i : i + stride].mean(0) for i in range(0, len(row), stride))
        encoded, lengths = new.encode_audio(batch["input_features"], batch["input_features_mask"])
        torch.testing.assert_close(encoded, torch.stack(expected))
        torch.testing.assert_close(lengths, proc.audio_token_lengths(batch["input_features_mask"]))
        assert torch.isfinite(new(**batch).loss)
    new.configure_training("adapter")
    new(**batch).loss.backward()
    assert any(p.grad is not None for p in new.aud_projector.parameters())
    assert all(p.grad is None for p in new.aud_encoder.parameters())
    assert all(p.grad is None for p in new.aud_decoder.parameters())
    assert verify(tmp_path / "aud")["offline_reload"]


def test_gemma_bridge_mismatch_requires_explicit_new_init(tmp_path, tokenizer):
    gqasr_source(tmp_path / "text", tokenizer)
    encoder_source(tmp_path / "encoder", "audflash_gemma")
    with pytest.raises(ValueError, match="Shape mismatch"):
        export_model(
            "audflash",
            tmp_path / "bad",
            gqasr_checkpoint=tmp_path / "text",
            encoder_source=tmp_path / "encoder",
            adapter_width=32,
        )
    assert not (tmp_path / "bad").exists()
    export_model(
        "audflash",
        tmp_path / "new",
        gqasr_checkpoint=tmp_path / "text",
        encoder_source=tmp_path / "encoder",
        adapter_width=32,
        bridge_init="new",
    )
    import json

    provenance = json.loads((tmp_path / "new/provenance.json").read_text())
    assert (
        provenance["tensor_mapping"]["aud_projector.down.weight"]["initialization"] == "new_adapter"
    )
