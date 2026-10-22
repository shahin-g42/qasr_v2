"""Standalone non-speech evaluation, separate from WER and training callbacks."""

import argparse
import json
from pathlib import Path

import torch

from .modeling import AUDForConditionalGeneration
from .processing import AUDProcessor
from .robustness import NonSpeechPool, load_mono_audio


def evaluate(
    model_path, manifests, output, device="cpu", language="en", max_new_tokens=64, limit=200
):
    if limit < 1 or max_new_tokens < 1:
        raise ValueError("Evaluation limits must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    processor = AUDProcessor.from_pretrained(model_path)
    pool = NonSpeechPool(
        dict(probability=0.5, manifests=manifests),
        0.1,
        processor.config.max_audio_seconds,
        require_audio_type=bool(processor.config.audio_type_version),
    )
    model = (
        AUDForConditionalGeneration.from_pretrained(model_path, local_files_only=True, dtype="auto")
        .to(device)
        .eval()
    )
    rows = []
    for part_number, part in enumerate(pool.parts):
        for index in range(len(part)):
            record = pool.record(part_number, index)
            wave = load_mono_audio(record["audio_path"], 16000)
            batch = processor.apply_transcription_request([wave], language=[language])
            batch = {k: v.to(device) for k, v in batch.items() if k in processor.model_input_names}
            with torch.inference_mode():
                result = model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )
            tokens = result[0, batch["input_ids"].shape[1] :].tolist()
            text = processor.decode(tokens)
            event = (
                processor.decode_audio_event(tokens)
                if processor.config.audio_type_version
                else None
            )
            rows.append(
                dict(
                    audio_path=record["audio_path"],
                    expected_audio_type=record.get("audio_type"),
                    event=event,
                    prediction=text,
                    nonempty=bool(text.strip()),
                    generated_tokens=len(tokens),
                    hit_token_cap=len(tokens) == max_new_tokens
                    and tokens[-1] != model.config.eos_token_id,
                )
            )
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    report = dict(
        samples=len(rows),
        nonempty_transcript_rate=sum(r["nonempty"] for r in rows) / len(rows),
        mean_generated_tokens=sum(r["generated_tokens"] for r in rows) / len(rows),
        token_cap_hits=sum(r["hit_token_cap"] for r in rows),
        language=language,
        selection="first eligible records in manifest order",
        examples=rows,
    )
    if processor.config.audio_type_version:
        report["category_accuracy"] = sum(
            r["event"]["audio_type"] == r["expected_audio_type"] for r in rows
        ) / len(rows)
        report["invalid_format_rate"] = sum(not r["event"]["format_valid"] for r in rows) / len(
            rows
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return {k: v for k, v in report.items() if k != "examples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=200)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
