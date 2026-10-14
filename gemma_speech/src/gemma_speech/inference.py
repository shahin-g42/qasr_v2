import json
import time
from pathlib import Path

import torch

from .asr import greedy_tokens
from .assets import processors_from_assets
from .audio import load_vocoder
from .batching import Collator, move_batch
from .checkpoint import load_checkpoint
from .config import TrainConfig
from .data import pad_sequences, read_manifest
from .metrics import score_predictions
from .text import normalize_text


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def transcribe(checkpoint, manifest, output, device="cpu", batch_size=1):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model, tokenizer, payload = load_checkpoint(checkpoint, device)
    if model.config.task != "asr":
        raise ValueError("transcribe requires an ASR checkpoint")
    cfg = TrainConfig(**payload["config"])
    rows = read_manifest(manifest)
    extractor, _ = processors_from_assets(payload["assets"])
    collator = Collator(cfg, tokenizer, extractor)
    model.eval()
    predictions = []
    # Timer includes file loading, resampling, features and decoding; model load is excluded.
    # No hardware-comparable speed claim is inferred from this single invocation.
    synchronize(device)
    started = time.monotonic()
    for start in range(0, len(rows), batch_size):
        batch, selected = collator(rows[start : start + batch_size])
        batch.pop("targets")
        batch.pop("target_mask")
        if model.config.asr_decoder == "qwen":
            ids = model.generate_transcripts(
                **move_batch(batch, device), max_new_tokens=cfg.max_text_tokens
            )
        else:
            result = model(**move_batch(batch, device))
            ids = greedy_tokens(result["logits"], result["mask"])
        predictions.extend(
            {
                "id": row.id,
                "reference": row.text,
                "language": row.language,
                "dialect": row.dialect,
                "hypothesis": tokenizer.decode(tokens),
            }
            for row, tokens in zip(selected, ids, strict=True)
        )
    synchronize(device)
    elapsed = time.monotonic() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "elapsed_seconds": elapsed,
        "batch_size": batch_size,
        "device": str(device),
        "precision": "fp32",
        "scores": score_predictions(predictions),
    }


@torch.no_grad()
def synthesize(
    checkpoint,
    text,
    speaker,
    language,
    output,
    device="cpu",
    vocoder=None,
    seconds=None,
    steps=32,
    guidance=1.0,
    seed=42,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if output.suffix != (".wav" if vocoder else ".pt"):
        raise ValueError("use .wav with --vocoder, or .pt for mel-only output")
    model, tokenizer, payload = load_checkpoint(checkpoint, device)
    if model.config.task != "tts":
        raise ValueError("synthesize requires a TTS checkpoint")
    cfg = TrainConfig(**payload["config"])
    if speaker not in payload["speakers"] or language not in payload["languages"]:
        raise ValueError("speaker and language must appear in the trained checkpoint mappings")
    text = normalize_text(text)
    tokens, mask = pad_sequences([tokenizer.encode(text)])
    if tokens.shape[1] > cfg.max_text_tokens or not mask.any():
        raise ValueError("empty or overlong text; segment text before synthesis")
    batch = {
        "tokens": tokens,
        "token_mask": mask,
        "speaker_ids": torch.tensor([payload["speakers"][speaker]]),
        "language_ids": torch.tensor([payload["languages"][language]]),
    }
    _, hf_tokenizer = processors_from_assets(payload["assets"])
    if hf_tokenizer is not None:
        encoded = hf_tokenizer([text], return_tensors="pt", truncation=False)
        if encoded["input_ids"].shape[1] > cfg.max_text_tokens:
            raise ValueError("Gemma token limit exceeded; segment text")
        batch.update(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"].bool()
        )
    if seconds is not None:
        if not 0 < seconds <= cfg.max_audio_seconds:
            raise ValueError("duration must be positive and within the trained audio limit")
        batch["mel_lengths"] = torch.tensor([max(1, round(seconds * 24000 / 256))])
    model.eval()
    mel, lengths = model.sample(
        **move_batch(batch, device),
        steps=steps,
        guidance=guidance,
        max_frames=int(cfg.max_audio_seconds * 24000 / 256) + 1,
        generator=torch.Generator(device=device).manual_seed(seed),
    )
    if not torch.isfinite(mel).all():
        raise FloatingPointError("nonfinite synthesized mel")
    mel = mel[0, : int(lengths[0])]
    output.parent.mkdir(parents=True, exist_ok=True)
    if vocoder:
        import soundfile as sf

        decoder = load_vocoder(vocoder, device)
        waveform = decoder.decode(mel.T[None]).squeeze(0).float().cpu().numpy()
        # Model quality, pronunciation and audible artifacts require real-data evaluation.
        if not __import__("numpy").isfinite(waveform).all():
            raise FloatingPointError("vocoder produced nonfinite audio")
        sf.write(output, waveform, 24000, subtype="FLOAT")
    else:
        with output.open("xb") as f:
            torch.save({"log_mel": mel.cpu(), "sample_rate": 24000, "hop_length": 256}, f)
    return {
        "output": str(output.resolve()),
        "frames": int(lengths[0]),
        "seconds": int(lengths[0]) * 256 / 24000,
        "seed": seed,
        "steps": steps,
    }
