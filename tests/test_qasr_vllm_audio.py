"""Parity of the vLLM plugin's audio path with the training code.

The plugin (``vllm_plugin/qasr_vllm``) re-implements the QASR front end and
audio tower without importing ``qasr``, because the vLLM image's transformers
may not match the training pin. These tests pin it to the training numerics:
any drift here would silently change every served transcript.
"""

from __future__ import annotations

import json
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


if __name__ == "__main__":
    unittest.main()
