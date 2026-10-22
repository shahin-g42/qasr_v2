"""Evaluate clip-level audio categories on a held-out, explicitly labeled manifest."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from qasr.audio import load_mono_audio
from qasr.data import parse_record

from .audio_types import TOKENS
from .modeling import AUDForConditionalGeneration
from .processing import AUDProcessor


def metrics(rows):
    if not rows:
        raise ValueError("No labeled examples evaluated")
    confusion = {name: Counter() for name in TOKENS}
    for row in rows:
        confusion[row["expected"]][row["event"]["audio_type"] or "invalid"] += 1
    per_class = {}
    for name in TOKENS:
        support = sum(confusion[name].values())
        predicted = sum(counts[name] for counts in confusion.values())
        correct = confusion[name][name]
        per_class[name] = dict(
            support=support,
            predicted=predicted,
            precision=correct / predicted if predicted else None,
            recall=correct / support if support else None,
        )
    speech = [r for r in rows if r["expected"] == "speech"]
    negatives = [r for r in rows if r["expected"] != "speech"]
    return dict(
        samples=len(rows),
        confusion={k: dict(v) for k, v in confusion.items()},
        per_class=per_class,
        accuracy=sum(r["expected"] == r["event"]["audio_type"] for r in rows) / len(rows),
        invalid_format_rate=sum(not r["event"]["format_valid"] for r in rows) / len(rows),
        missed_speech_rate=sum(r["event"]["audio_type"] != "speech" for r in speech) / len(speech)
        if speech
        else None,
        non_speech_text_rate=sum(bool(r["event"]["text"].strip()) for r in negatives)
        / len(negatives)
        if negatives
        else None,
        token_cap_hits=sum(r["hit_token_cap"] for r in rows),
    )


def evaluate(model_path, manifest, output, device="cpu", limit=400, max_new_tokens=768):
    if limit < 1 or max_new_tokens < 1:
        raise ValueError("Evaluation limits must be positive")
    output, manifest = Path(output), Path(manifest).resolve()
    if output.exists():
        raise FileExistsError(output)
    processor = AUDProcessor.from_pretrained(model_path)
    if not processor.config.audio_type_version:
        raise ValueError("Evaluate an audio-type-enabled checkpoint")
    model = (
        AUDForConditionalGeneration.from_pretrained(model_path, local_files_only=True, dtype="auto")
        .to(device)
        .eval()
    )
    rows = []
    with manifest.open() as source:
        for line, raw in enumerate(source, 1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            kind = record.get("audio_type")
            if kind not in TOKENS:
                raise ValueError(f"Missing/invalid audio_type at line {line}")
            parsed = parse_record(
                record,
                manifest_path=manifest,
                line_number=line,
                audio_root=None,
                language=record.get("language"),
                empty_target_ok=True,
            )
            if (kind == "speech") != bool(parsed["text"].strip()):
                raise ValueError(f"Transcript/category mismatch at line {line}")
            wave = load_mono_audio(parsed["audio_path"], 16000)
            # No category or reference-language hint is supplied at inference.
            batch = processor.apply_transcription_request([wave])
            batch = {k: v.to(device) for k, v in batch.items() if k in processor.model_input_names}
            with torch.inference_mode():
                tokens = model.generate(
                    **batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
                )[0, batch["input_ids"].shape[1] :]
            rows.append(
                dict(
                    expected=kind,
                    event=processor.decode_audio_event(tokens),
                    reference_text=parsed["text"],
                    audio_path=parsed["audio_path"],
                    hit_token_cap=len(tokens) == max_new_tokens
                    and tokens[-1].item() != model.config.eos_token_id,
                )
            )
            if len(rows) == limit:
                break
    report = dict(
        **metrics(rows),
        examples=rows,
        selection="First eligible rows in manifest order; use a balanced held-out manifest",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return {k: v for k, v in report.items() if k != "examples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    print(json.dumps(evaluate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
