"""Offline integration of all three original checkpoints in one installed runtime.

Run after prefetch. Uses a sine wave: results test tensor/model compatibility,
not recognition accuracy, voice similarity, intelligibility or production speed.
"""

import argparse
import gc
import json
import platform
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from gqtts.qwen_tts_train import QwenTTSFineTune
from gqtts.speaker import SpeakerAdapter, audio_frames, distillation_loss, load_qwen_tts

from gemma_speech._vendor.qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
from gemma_speech.assets import load_assets, processors_from_assets
from gemma_speech.backbone import load_component
from gemma_speech.config import ModelConfig
from gemma_speech.data import pad_sequences
from gemma_speech.qwen_asr import QwenASRModel, QwenTokenizer, load_qwen_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gemma", "qwen-asr", "qwen-tts", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(4)
    torch.manual_seed(7)
    cfg = ModelConfig(
        asr_decoder="qwen", model_id=args.gemma, qwen_model_id=args.qwen_asr, local_files_only=True
    )
    component = load_component(cfg)
    extractor, _ = processors_from_assets(load_assets(cfg))
    wave16 = (0.05 * np.sin(np.arange(16000) / 16000 * 440 * 2 * np.pi)).astype("float32")
    wave24 = (0.05 * np.sin(np.arange(24000) / 24000 * 440 * 2 * np.pi)).astype("float32")
    tokenizer = QwenTokenizer.from_pretrained(cfg)
    decoder = load_qwen_decoder(cfg)
    model = QwenASRModel(component, decoder, cfg, tokenizer)
    features = extractor(
        [wave16],
        sampling_rate=16000,
        padding=True,
        truncation=False,
        max_length=None,
        return_tensors="pt",
    )
    ids, mask = pad_sequences([tokenizer.target("مرحبا بالعالم", "ar")])
    result = model(features["input_features"], features["input_features_mask"].bool(), ids, mask)
    result["loss"].backward()
    report = {
        "kind": "pretrained integration with synthetic audio; no quality/speed benchmark",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "device": "cpu",
        "asr": {
            "parameters": sum(p.numel() for p in model.parameters()),
            "bridge_parameters": sum(p.numel() for p in model.bridge.parameters()),
            "tied_decoder_embeddings": decoder.config.tie_word_embeddings,
            "loss": float(result["loss"].detach()),
            "bridge_gradient_norm": float(model.bridge[1].weight.grad.norm()),
        },
    }
    model.eval()
    report["asr"]["generated_tokens"] = model.generate_transcripts(
        features["input_features"], features["input_features_mask"].bool(), max_new_tokens=4
    ).shape[1]
    del result, model, decoder
    gc.collect()
    api = load_qwen_tts(args.qwen_tts)
    with torch.no_grad():
        teacher = api.model.extract_speaker_embedding(wave24, sr=24000)
        codes = api.model.speech_tokenizer.encode(wave24, sr=24000).audio_codes[0]
        reconstructed, rate = api.model.speech_tokenizer.decode([{"audio_codes": codes}])
        frames = audio_frames(component, extractor, wave16, "cpu")
    adapter = SpeakerAdapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    valid = torch.ones(1, len(frames), dtype=torch.bool)
    predicted = adapter(frames[None], valid)
    loss, _ = distillation_loss(predicted, teacher[None])
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        vector = adapter(frames[None], valid)[0]
        prompt = VoiceClonePromptItem(
            ref_code=None,
            ref_spk_embedding=vector,
            x_vector_only_mode=True,
            icl_mode=False,
            ref_text=None,
        )
        # Check the training prompt against the actual retained Qwen generation path.
        captured = {}

        class PrefillCaptured(Exception):
            pass

        def capture(**kwargs):
            captured["embeddings"] = kwargs["inputs_embeds"]
            raise PrefillCaptured

        with patch.object(api.model.talker, "generate", capture):
            try:
                api.generate_voice_clone(
                    text="Hello.",
                    language="Auto",
                    voice_clone_prompt=[prompt],
                    non_streaming_mode=True,
                )
            except PrefillCaptured:
                pass
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "frames.pt"
            torch.save({"frames": frames, "codes": codes}, cache)
            training = QwenTTSFineTune(api, adapter)
            packed, _, _, bounds, _ = training.pack([{"file": str(cache), "text": "Hello."}])
            torch.testing.assert_close(packed[:, : bounds[0][0]], captured["embeddings"])
        waves, sr = api.generate_voice_clone(
            text="Hello.",
            language="English",
            voice_clone_prompt=[prompt],
            non_streaming_mode=True,
            max_new_tokens=4,
            do_sample=False,
            subtalker_dosample=False,
        )
    report["tts"] = {
        "training_prefill_matches_generation": True,
        "gemma_frame_shape": list(frames.shape),
        "speaker_shape": list(vector.shape),
        "adapter_parameters": sum(p.numel() for p in adapter.parameters()),
        "distillation_loss": float(loss.detach()),
        "codec_shape": list(codes.shape),
        "codec_reconstruction_samples": len(reconstructed[0]),
        "codec_sample_rate": rate,
        "generated_samples": len(waves[0]),
        "sample_rate": sr,
        "finite": bool(np.isfinite(waves[0]).all()),
    }
    assert all(torch.isfinite(v).all() for v in (frames, predicted, loss))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
