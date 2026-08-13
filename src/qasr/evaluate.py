"""Comprehensive quality/accuracy/performance evaluation for the QASR stack.

Evaluates the complete inference setup on a JSONL manifest across up to four
modes, all sharing one loaded model:

    offline          greedy batch-of-one transcription (quality baseline)
    eagle            EAGLE-2 speculative decoding (must match offline output)
    streaming        rolling-window PCM16 simulation of the WebSocket server
    streaming-eagle  the same simulation decoded through the EAGLE path

Reported metrics:
    quality      corpus WER / CER against the manifest references
    accuracy     EAGLE-vs-offline exact-match rate, draft acceptance stats
    performance  latency (mean/p50/p90), RTF, tokens/sec, peak GPU memory

Usage:
    python evaluate.py \
        --model /path/to/output/full \
        --manifest /path/to/validated_manifests/train_en_commentary.json \
        --eagle /path/to/output/eagle/checkpoint-5000 \
        --language en --num-samples 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio import load_mono_audio
from .modeling import QASRForConditionalGeneration
from .processing import QASRProcessor
from .streaming import PCM16Buffer, merge_windowed_transcript

LOGGER = logging.getLogger("qasr.evaluate")


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest_samples(
    manifest_path: str | Path,
    *,
    num_samples: int,
    min_duration: float,
    max_duration: float,
    seed: int,
    audio_root: str | None = None,
) -> list[dict[str, Any]]:
    """Read a JSONL manifest and pick a deterministic random subset."""
    records: list[dict[str, Any]] = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed JSON at line %d", line_number)
                continue
            audio_path = record.get("audio_filepath") or record.get("wav_path")
            text = record.get("text") or record.get("transcript")
            if not audio_path or not text:
                continue
            duration = record.get("duration")
            if duration is not None and not (min_duration <= float(duration) <= max_duration):
                continue
            # Same cluster path rewrite the training data loader applies.
            audio_path = re.sub(r"^/vast", "/lustrefs/taiga/vast40", audio_path)
            if audio_root and not Path(audio_path).is_absolute():
                audio_path = str(Path(audio_root) / audio_path)
            records.append(
                {"audio_filepath": audio_path, "text": text, "duration": duration}
            )
    if not records:
        raise ValueError(f"No usable samples found in {manifest_path}")
    rng = random.Random(seed)
    if num_samples < len(records):
        records = rng.sample(records, num_samples)
    LOGGER.info("Selected %d samples from %s", len(records), manifest_path)
    return records


# ---------------------------------------------------------------------------
# Text metrics (WER / CER)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Casefold and strip punctuation so WER measures word identity only."""
    cleaned = "".join(
        character if character.isalnum() or character.isspace() or character == "'"
        else " "
        for character in text.casefold()
    )
    return " ".join(cleaned.split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance with O(min(len)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitution))
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str) -> dict[str, int]:
    """Word- and character-level edit counts for corpus-level aggregation."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    ref_chars = list(normalize_text(reference).replace(" ", ""))
    hyp_chars = list(normalize_text(hypothesis).replace(" ", ""))
    return {
        "word_errors": edit_distance(ref_words, hyp_words),
        "word_count": len(ref_words),
        "char_errors": edit_distance(ref_chars, hyp_chars),
        "char_count": len(ref_chars),
    }


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def greedy_transcribe(
    model: Any,
    processor: QASRProcessor,
    audio: np.ndarray,
    *,
    language: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Plain greedy generation; returns (transcript, generated_token_count)."""
    inputs = processor.apply_transcription_request(
        audio=np.ascontiguousarray(audio, dtype=np.float32),
        language=language,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids[0], return_format="transcription_only").strip()
    return text, generated_ids.shape[1]


def eagle_transcribe(
    decoder: Any,
    processor: QASRProcessor,
    audio: np.ndarray,
    *,
    language: str,
    max_new_tokens: int,
    num_draft_tokens: int,
) -> tuple[str, int]:
    """Speculative generation; returns (transcript, generated_token_count)."""
    inputs = processor.apply_transcription_request(
        audio=np.ascontiguousarray(audio, dtype=np.float32),
        language=language,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(decoder.device, decoder.dtype)
    output_ids = decoder.generate(
        input_ids=inputs["input_ids"],
        input_features=inputs["input_features"],
        input_features_mask=inputs["input_features_mask"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        num_draft_tokens=num_draft_tokens,
    )
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated_ids[0], return_format="transcription_only").strip()
    return text, generated_ids.shape[1]


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def run_utterance_eval(
    samples: list[dict[str, Any]],
    decode_fn: Callable[[np.ndarray], tuple[str, int]],
    *,
    sample_rate: int,
    device: torch.device,
    mode: str,
) -> list[dict[str, Any]]:
    """Time one full-utterance decode per sample and collect quality metrics."""
    results = []
    for index, sample in enumerate(samples, start=1):
        audio = load_mono_audio(sample["audio_filepath"], target_sampling_rate=sample_rate)
        audio_seconds = audio.size / sample_rate
        _sync_if_cuda(device)
        started = time.perf_counter()
        hypothesis, token_count = decode_fn(audio)
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - started
        record = {
            "audio_filepath": sample["audio_filepath"],
            "reference": sample["text"],
            "hypothesis": hypothesis,
            "audio_seconds": round(audio_seconds, 3),
            "latency_seconds": round(elapsed, 4),
            "rtf": round(elapsed / audio_seconds, 4) if audio_seconds else None,
            "generated_tokens": token_count,
            "tokens_per_second": round(token_count / elapsed, 2) if elapsed else None,
            **error_counts(sample["text"], hypothesis),
        }
        results.append(record)
        LOGGER.info(
            "[%s %d/%d] WER=%.1f%% latency=%.2fs rtf=%.3f | %s",
            mode, index, len(samples),
            100 * record["word_errors"] / max(record["word_count"], 1),
            elapsed, record["rtf"], Path(sample["audio_filepath"]).name,
        )
    return results


def run_streaming_eval(
    samples: list[dict[str, Any]],
    decode_fn: Callable[[np.ndarray], tuple[str, int]],
    *,
    sample_rate: int,
    device: torch.device,
    mode: str,
    chunk_ms: float,
    partial_interval_seconds: float,
    window_seconds: float,
) -> list[dict[str, Any]]:
    """Replay each file through the rolling-window pipeline the server uses.

    Audio is fed as PCM16 chunks into the bounded buffer; a decode fires each
    time ``partial_interval_seconds`` of new audio has arrived, and partials
    are merged with :func:`merge_windowed_transcript` exactly like the
    WebSocket server does. Runs unpaced: real-time viability is judged by RTF.
    """
    chunk_samples = max(1, round(chunk_ms / 1000 * sample_rate))
    interval_samples = max(1, round(partial_interval_seconds * sample_rate))
    results = []
    for index, sample in enumerate(samples, start=1):
        audio = load_mono_audio(sample["audio_filepath"], target_sampling_rate=sample_rate)
        audio_seconds = audio.size / sample_rate
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        buffer = PCM16Buffer(max_samples=int(window_seconds * sample_rate))

        merged = ""
        last_partial = ""
        decode_latencies: list[float] = []
        first_partial: dict[str, float] | None = None
        last_decode_total = 0

        def decode_window(active_buffer: PCM16Buffer, latencies: list[float]) -> None:
            nonlocal merged, last_partial, first_partial
            waveform = active_buffer.waveform()
            if waveform.size < sample_rate // 20:  # < 50 ms, matches the server
                return
            _sync_if_cuda(device)
            started = time.perf_counter()
            partial, _ = decode_fn(waveform)
            _sync_if_cuda(device)
            elapsed = time.perf_counter() - started
            latencies.append(elapsed)
            if partial and partial != last_partial:
                last_partial = partial
                merged = merge_windowed_transcript(merged, partial)
                if first_partial is None:
                    first_partial = {
                        "audio_seconds": active_buffer.total_samples / sample_rate,
                        "latency_seconds": elapsed,
                    }

        for offset in range(0, len(pcm), chunk_samples * 2):
            buffer.append(pcm[offset:offset + chunk_samples * 2])
            if buffer.total_samples - last_decode_total >= interval_samples:
                last_decode_total = buffer.total_samples
                decode_window(buffer, decode_latencies)
        if buffer.total_samples != last_decode_total:
            decode_window(buffer, decode_latencies)  # flush the tail

        inference_seconds = sum(decode_latencies)
        record = {
            "audio_filepath": sample["audio_filepath"],
            "reference": sample["text"],
            "hypothesis": merged,
            "audio_seconds": round(audio_seconds, 3),
            "decode_count": len(decode_latencies),
            "latency_seconds": round(inference_seconds, 4),
            "mean_decode_seconds": round(
                statistics.fmean(decode_latencies), 4
            ) if decode_latencies else None,
            "max_decode_seconds": round(max(decode_latencies), 4) if decode_latencies else None,
            "rtf": round(inference_seconds / audio_seconds, 4) if audio_seconds else None,
            "first_partial": first_partial,
            **error_counts(sample["text"], merged),
        }
        results.append(record)
        LOGGER.info(
            "[%s %d/%d] WER=%.1f%% decodes=%d rtf=%.3f | %s",
            mode, index, len(samples),
            100 * record["word_errors"] / max(record["word_count"], 1),
            len(decode_latencies), record["rtf"], Path(sample["audio_filepath"]).name,
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[position]


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [r["latency_seconds"] for r in results]
    rtfs = [r["rtf"] for r in results if r["rtf"] is not None]
    word_errors = sum(r["word_errors"] for r in results)
    word_count = sum(r["word_count"] for r in results)
    char_errors = sum(r["char_errors"] for r in results)
    char_count = sum(r["char_count"] for r in results)
    summary: dict[str, Any] = {
        "samples": len(results),
        "audio_hours": round(sum(r["audio_seconds"] for r in results) / 3600, 4),
        "wer": round(word_errors / word_count, 4) if word_count else None,
        "cer": round(char_errors / char_count, 4) if char_count else None,
        "latency_mean_s": round(statistics.fmean(latencies), 3),
        "latency_p50_s": round(_percentile(latencies, 0.5), 3),
        "latency_p90_s": round(_percentile(latencies, 0.9), 3),
        "rtf_mean": round(statistics.fmean(rtfs), 4) if rtfs else None,
    }
    token_rates = [r["tokens_per_second"] for r in results if r.get("tokens_per_second")]
    if token_rates:
        summary["tokens_per_second_mean"] = round(statistics.fmean(token_rates), 1)
    decode_counts = [r["decode_count"] for r in results if "decode_count" in r]
    if decode_counts:
        summary["decodes_per_sample_mean"] = round(statistics.fmean(decode_counts), 1)
        max_decodes = [r["max_decode_seconds"] for r in results if r.get("max_decode_seconds")]
        summary["decode_latency_max_s"] = round(max(max_decodes), 3) if max_decodes else None
    return summary


def print_report(report: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print("QASR EVALUATION REPORT")
    print("=" * 72)
    for mode, block in report["modes"].items():
        print(f"\n--- {mode} ---")
        for key, value in block["summary"].items():
            print(f"  {key:>26}: {value}")
    if "eagle_stats" in report:
        print("\n--- eagle speculation ---")
        for key, value in report["eagle_stats"].items():
            print(f"  {key:>26}: {value}")
    if "comparison" in report:
        print("\n--- offline vs eagle ---")
        for key, value in report["comparison"].items():
            print(f"  {key:>26}: {value}")
    print("\n" + "=" * 72)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the QASR inference stack.")
    parser.add_argument("--model", required=True, help="Path to the QASR checkpoint")
    parser.add_argument("--manifest", required=True, help="JSONL manifest with references")
    parser.add_argument("--eagle", default=None, help="EAGLE head dir (enables eagle modes)")
    parser.add_argument(
        "--modes", nargs="+", default=None,
        choices=["offline", "eagle", "streaming", "streaming-eagle"],
        help="Modes to run (default: all applicable)",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=35.0)
    parser.add_argument("--chunk-ms", type=float, default=250.0)
    parser.add_argument("--partial-interval", type=float, default=0.75)
    parser.add_argument("--window-seconds", type=float, default=30.0)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="eval_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    modes = args.modes or (
        ["offline", "eagle", "streaming", "streaming-eagle"] if args.eagle
        else ["offline", "streaming"]
    )
    if any(mode.startswith("eagle") or mode.endswith("eagle") for mode in modes) and not args.eagle:
        parser.error("--eagle is required for the eagle / streaming-eagle modes")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    LOGGER.info("Loading QASR model from %s", args.model)
    processor = QASRProcessor.from_pretrained(args.model)
    model = QASRForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype, attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to(device).eval()
    sample_rate = int(processor.feature_extractor.sampling_rate)

    decoder = None
    if args.eagle:
        from .eagle import EagleHead, EagleSpeculativeDecoder

        eagle_head = EagleHead.from_pretrained(args.eagle)
        decoder = EagleSpeculativeDecoder(model=model, eagle_head=eagle_head)

    samples = load_manifest_samples(
        args.manifest,
        num_samples=args.num_samples,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        seed=args.seed,
        audio_root=args.audio_root,
    )

    def offline_decode(audio: np.ndarray) -> tuple[str, int]:
        return greedy_transcribe(
            model, processor, audio,
            language=args.language, max_new_tokens=args.max_new_tokens,
        )

    def eagle_decode(audio: np.ndarray) -> tuple[str, int]:
        result = eagle_transcribe(
            decoder, processor, audio,
            language=args.language, max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
        )
        # last_stats is overwritten on every generate call, so capture it now.
        collected_stats.append(decoder.last_stats)
        return result

    streaming_kwargs = {
        "sample_rate": sample_rate,
        "device": device,
        "chunk_ms": args.chunk_ms,
        "partial_interval_seconds": args.partial_interval,
        "window_seconds": args.window_seconds,
    }

    report: dict[str, Any] = {
        "model": args.model,
        "eagle": args.eagle,
        "manifest": args.manifest,
        "language": args.language,
        "num_draft_tokens": args.num_draft_tokens,
        "seed": args.seed,
        "modes": {},
    }
    collected_stats: list[Any] = []

    for mode in modes:
        LOGGER.info("=== Running mode: %s ===", mode)
        _reset_peak_memory(device)
        if mode == "offline":
            results = run_utterance_eval(
                samples, offline_decode, sample_rate=sample_rate, device=device, mode=mode
            )
        elif mode == "eagle":
            results = run_utterance_eval(
                samples, eagle_decode, sample_rate=sample_rate, device=device, mode=mode
            )
        elif mode == "streaming":
            results = run_streaming_eval(
                samples, offline_decode, mode=mode, **streaming_kwargs
            )
        else:  # streaming-eagle
            results = run_streaming_eval(
                samples, eagle_decode, mode=mode, **streaming_kwargs
            )
        summary = summarize(results)
        summary["peak_gpu_memory_gb"] = _peak_memory_gb(device)
        report["modes"][mode] = {"summary": summary, "results": results}

    # Speculation quality: aggregate the per-call telemetry captured while
    # running the eagle modes (acceptance decides whether EAGLE pays off).
    if collected_stats:
        totals = {"rounds": 0, "drafted": 0, "accepted": 0, "emitted": 0, "forwards": 0}
        positions = [0] * args.num_draft_tokens
        for stats in collected_stats:
            totals["rounds"] += stats.rounds
            totals["drafted"] += stats.drafted_tokens
            totals["accepted"] += stats.accepted_tokens
            totals["emitted"] += stats.emitted_tokens
            totals["forwards"] += stats.target_forwards
            for i, count in enumerate(stats.accepted_by_position[: len(positions)]):
                positions[i] += count
        report["eagle_stats"] = {
            "acceptance_rate": round(totals["accepted"] / totals["drafted"], 4)
            if totals["drafted"] else None,
            "tokens_per_target_forward": round(totals["emitted"] / totals["forwards"], 3)
            if totals["forwards"] else None,
            "mean_accepted_per_round": round(totals["accepted"] / totals["rounds"], 3)
            if totals["rounds"] else None,
            "acceptance_by_position": [
                round(count / totals["rounds"], 4) if totals["rounds"] else None
                for count in positions
            ],
        }
        # A healthy head trained on the matching checkpoint accepts >= 50% of
        # drafts. Near-zero acceptance is the signature of pairing an EAGLE
        # head with the wrong base checkpoint (or an architecture change such
        # as a new projector/token rate), which silently turns speculative
        # decoding into pure overhead.
        acceptance = report["eagle_stats"]["acceptance_rate"]
        if acceptance is not None and acceptance < 0.3:
            print(
                "\n"
                "=" * 78 + "\n"
                f"WARNING: EAGLE acceptance rate is {acceptance:.3f} (< 0.30).\n"
                "This almost always means the EAGLE head was trained against a\n"
                "different base checkpoint than the one being evaluated, or the\n"
                "head architecture does not match the checkpoint (e.g. v1 head\n"
                "loaded onto a v2 fc1 4096x2048 layout). Retrain the head on\n"
                "the exact checkpoint you intend to serve, or pass the matching\n"
                "--eagle path. Speculative decoding at this acceptance rate is\n"
                "slower than plain greedy decoding.\n"
                + "=" * 78
            )

    # Offline vs EAGLE: greedy speculative decoding must be lossless, so any
    # transcript mismatch indicates a decoder bug rather than model quality.
    if "offline" in report["modes"] and "eagle" in report["modes"]:
        offline_results = report["modes"]["offline"]["results"]
        eagle_results = report["modes"]["eagle"]["results"]
        matches = sum(
            base["hypothesis"] == spec["hypothesis"]
            for base, spec in zip(offline_results, eagle_results, strict=True)
        )
        base_latency = report["modes"]["offline"]["summary"]["latency_mean_s"]
        spec_latency = report["modes"]["eagle"]["summary"]["latency_mean_s"]
        report["comparison"] = {
            "exact_transcript_match": f"{matches}/{len(offline_results)}",
            "eagle_speedup": round(base_latency / spec_latency, 3) if spec_latency else None,
        }

    print_report(report)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Full per-sample results written to %s", output_path.resolve())


__all__ = [
    "edit_distance",
    "error_counts",
    "load_manifest_samples",
    "main",
    "normalize_text",
    "run_streaming_eval",
    "run_utterance_eval",
    "summarize",
]


if __name__ == "__main__":
    main()
