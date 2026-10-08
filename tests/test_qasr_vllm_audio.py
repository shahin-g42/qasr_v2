"""Parity of the vLLM plugin's audio path with the training code.

The plugin (``vllm_plugin/qasr_vllm``) re-implements the QASR front end and
audio tower without importing ``qasr``, because the vLLM image's transformers
may not match the training pin. These tests pin it to the training numerics:
any drift here would silently change every served transcript.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vllm_plugin"))

from qasr_vllm.audio import (
    QASRAudioTower,
    QASRMelExtractor,
    subsampling_output_lengths,
)
from qasr_vllm.config import QASRVllmConfig

from qasr.configuration import QASRConfig
from qasr.feature_extraction import QASRFeatureExtractor
from qasr.modeling import QASRForConditionalGeneration
from qasr.utils import get_subsampling_output_lengths

TINY_AUDIO = {
    "model_type": "parakeet_encoder", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
    "intermediate_size": 128, "num_mel_bins": 128, "subsampling_factor": 8, "subsampling_conv_channels": 16,
    "conv_kernel_size": 9, "attention_bias": True, "convolution_bias": True,
    "subsampling_conv_kernel_size": 3, "subsampling_conv_stride": 2, "scale_input": False,
}
TINY_TEXT = {
    "model_type": "qwen3", "hidden_size": 48, "num_hidden_layers": 1, "num_attention_heads": 4,
    "num_key_value_heads": 2, "intermediate_size": 96, "vocab_size": 151936, "head_dim": 12,
}


def _waveforms() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16_000 * 3 + 123, 16_000 * 7 + 5)]


class MelParityTest(unittest.TestCase):
    def test_matches_training_extractor_per_utterance(self) -> None:
        reference = QASRFeatureExtractor()
        ours = QASRMelExtractor()
        for wav in _waveforms():
            ref = reference(wav, sampling_rate=16_000, chunk_long_audio=False)
            frames = int(ref["attention_mask"].sum())
            features, n = ours(wav)
            self.assertEqual(n, frames)
            torch.testing.assert_close(features, ref["input_features"][0, :frames], rtol=0, atol=1e-5)

    def test_matches_training_extractor_inside_padded_batch(self) -> None:
        # Training extracts whole padded batches; per-utterance extraction must agree.
        wavs = _waveforms()
        ref = QASRFeatureExtractor()(wavs, sampling_rate=16_000, chunk_long_audio=False)
        ours = QASRMelExtractor()
        for i, wav in enumerate(wavs):
            frames = int(ref["attention_mask"][i].sum())
            features, n = ours(wav)
            self.assertEqual(n, frames)
            torch.testing.assert_close(features, ref["input_features"][i, :frames], rtol=0, atol=1e-5)

    def test_token_count_formula_matches_training(self) -> None:
        lengths = torch.arange(1, 4000)
        torch.testing.assert_close(subsampling_output_lengths(lengths), get_subsampling_output_lengths(lengths))


class TowerParityTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.tmp = tempfile.TemporaryDirectory()
        cfg = QASRConfig(audio_config=TINY_AUDIO, text_config=TINY_TEXT, tie_word_embeddings=True)
        self.model = QASRForConditionalGeneration(cfg).eval()
        self.model.save_pretrained(self.tmp.name, safe_serialization=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _plugin_config(self) -> QASRVllmConfig:
        return QASRVllmConfig(**json.loads((Path(self.tmp.name) / "config.json").read_text()))

    def test_config_round_trip(self) -> None:
        cfg = self._plugin_config()
        self.assertEqual(cfg.audio_config.hidden_size, 64)
        self.assertEqual(cfg.text_config.hidden_size, 48)
        self.assertTrue(cfg.text_config.tie_word_embeddings)
        self.assertEqual(cfg.audio_token_id, self.model.config.audio_token_id)

    def test_audio_embeddings_match_training_model(self) -> None:
        tower = QASRAudioTower(self._plugin_config()).eval()
        state = self.model.state_dict()
        mapped = {}
        for key, value in state.items():
            if key.startswith("model.audio_tower."):
                mapped["encoder." + key[len("model.audio_tower."):]] = value
            elif key.startswith("model.multi_modal_projector."):
                mapped["projector." + key[len("model.multi_modal_projector."):]] = value
        missing, unexpected = tower.load_state_dict(mapped, strict=False)
        self.assertEqual(unexpected, [])
        # Non-persistent buffers (rotary/rel-pos tables) are rebuilt at init.
        self.assertEqual([m for m in missing if "running" in m or "weight" in m or "bias" in m], [])

        wavs = _waveforms()
        fe = QASRFeatureExtractor()
        ref_in = fe(wavs, sampling_rate=16_000, chunk_long_audio=False)
        with torch.no_grad():
            ref = self.model.model.get_audio_features(
                ref_in["input_features"], ref_in["attention_mask"]
            ).pooler_output
            ours = tower([QASRMelExtractor()(w)[0] for w in wavs])
        self.assertEqual(sum(o.shape[0] for o in ours), ref.shape[0])
        torch.testing.assert_close(torch.cat(ours), ref, rtol=1e-4, atol=1e-4)
        for wav, emb in zip(wavs, ours, strict=False):
            self.assertEqual(emb.shape[0], tower.num_tokens(QASRMelExtractor().num_frames(len(wav))))


class MelFallbackTest(unittest.TestCase):
    def test_transformers_mel_bank_matches_librosa(self) -> None:
        # The vllm-openai image ships no librosa, so production uses the fallback.
        import builtins

        real_import = builtins.__import__

        def no_librosa(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_librosa
        try:
            fallback = QASRMelExtractor()
        finally:
            builtins.__import__ = real_import
        torch.testing.assert_close(fallback.mel_filters, QASRMelExtractor().mel_filters, rtol=0, atol=1e-7)
        for wav in _waveforms():
            torch.testing.assert_close(fallback(wav)[0], QASRMelExtractor()(wav)[0], rtol=0, atol=1e-5)


# Qwen3-1.7B decoder dims as welded by convert_weights (Audar-ASR-V1.2-Turbo).
FULL_TEXT = {
    "model_type": "qwen3", "hidden_size": 2048, "num_hidden_layers": 28, "num_attention_heads": 16,
    "num_key_value_heads": 8, "head_dim": 128, "intermediate_size": 6144, "vocab_size": 151936,
    "tie_word_embeddings": True,
}
# vLLM's WeightsMapper in qasr_vllm.model, reproduced so this test needs no vLLM.
PLUGIN_PREFIXES = {
    "model.audio_tower.": "audio_tower.encoder.",
    "model.multi_modal_projector.": "audio_tower.projector.",
    "model.language_model.": "language_model.model.",
    "lm_head.": "language_model.lm_head.",
}


def _map(name: str) -> str:
    for old, new in PLUGIN_PREFIXES.items():
        if name.startswith(old):
            return new + name[len(old):]
    raise AssertionError(f"checkpoint key {name!r} matches no plugin prefix")


class FullSizeLayoutTest(unittest.TestCase):
    """Every production-checkpoint tensor must land on a plugin tensor, and vice versa."""

    def setUp(self) -> None:
        cfg = QASRConfig(text_config=FULL_TEXT, tie_word_embeddings=True)
        with torch.device("meta"):
            self.model = QASRForConditionalGeneration(cfg)
            self.tower = QASRAudioTower(QASRVllmConfig(**json.loads(cfg.to_json_string())))
        # What save_pretrained writes: the state dict minus the tied lm_head.
        self.saved = [k for k in self.model.state_dict() if k != "lm_head.weight"]

    def test_parameter_counts(self) -> None:
        def count(module: torch.nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        self.assertEqual(len(self.model.model.audio_tower.layers), 48)
        self.assertEqual(len(self.model.model.language_model.layers), 28)
        self.assertEqual(count(self.model.model.multi_modal_projector), 4_263_168)
        self.assertEqual(round(count(self.model) / 1e9, 2), 3.62)
        self.assertEqual(count(self.tower), count(self.model.model.audio_tower) + 4_263_168)

    def test_audio_keys_cover_tower_exactly(self) -> None:
        mapped = {_map(k)[len("audio_tower."):] for k in self.saved if k.startswith(("model.audio_tower.", "model.multi_modal_projector."))}
        # vLLM's AutoWeightsLoader fills parameters plus persistent buffers (BatchNorm stats).
        expected = set(self.tower.state_dict())
        self.assertEqual(mapped - expected, set(), "checkpoint tensors with no plugin target")
        self.assertEqual(expected - mapped, set(), "plugin tensors the checkpoint never fills")

    def test_decoder_keys_are_qwen3_layout(self) -> None:
        decoder = [_map(k) for k in self.saved if k.startswith("model.language_model.")]
        self.assertEqual(len(decoder), 2 + 28 * 11)  # embed + final norm, 11 tensors per layer
        allowed = re.compile(
            r"language_model\.model\.(embed_tokens\.weight|norm\.weight|layers\.\d+\.("
            r"input_layernorm|post_attention_layernorm|self_attn\.(q|k|v|o)_proj|"
            r"self_attn\.(q|k)_norm|mlp\.(gate|up|down)_proj)\.weight)"
        )
        self.assertEqual([k for k in decoder if not allowed.fullmatch(k)], [])
        self.assertNotIn("lm_head.weight", self.saved)


if __name__ == "__main__":
    unittest.main()
