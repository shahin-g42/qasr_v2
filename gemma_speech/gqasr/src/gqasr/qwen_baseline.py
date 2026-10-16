"""Compare the official Qwen3-ASR baseline on exactly the saved Chinese eval IDs."""
import argparse
import hashlib
import json
import random
import sqlite3
import time
import unicodedata
from pathlib import Path

from .metrics import edit_distance, score_predictions
from .text import normalize_text

MODEL = "Qwen/Qwen3-ASR-1.7B-hf"


def read_rows(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def select_samples(index, predictions):
    rows = [r for r in read_rows(predictions) if r["language"] == "zh"]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty, unique Chinese evaluation IDs")
    samples = []
    with sqlite3.connect(f"file:{Path(index).resolve()}?mode=ro", uri=True) as db:
        for row in rows:
            split, language, ordinal = row["id"].split(":")
            if (split, language) != ("eval", "zh"):
                raise ValueError("Only eval:zh IDs are accepted")
            found = db.execute(
                "SELECT audio,text,duration,source,line FROM samples "
                "WHERE split=? AND language=? AND ordinal=?", (split, language, int(ordinal))
            ).fetchone()
            if found is None or normalize_text(found[1]) != normalize_text(row["reference"]):
                raise ValueError(f"Missing ID or reference mismatch: {row['id']}")
            if not Path(found[0]).is_file():
                raise FileNotFoundError(found[0])
            samples.append(dict(id=row["id"], language="zh", reference=row["reference"],
                                audio=found[0], duration=found[2], source=found[3],
                                source_line=found[4], gqasr_hypothesis=row["hypothesis"]))
    return samples



def select_training_samples(index, per_group=500, seed=42):
    """Uniform reservoir sample within each script stratum; never use ORDER BY RANDOM."""
    pools = {"han_only": [], "han_latin": []}
    counts = {key: 0 for key in pools}
    rng = random.Random(seed)
    with sqlite3.connect(f"file:{Path(index).resolve()}?mode=ro", uri=True) as db:
        for ordinal, audio, text, duration, source, line in db.execute(
            "SELECT ordinal,audio,text,duration,source,line FROM samples "
            "WHERE split='train' AND language='zh' ORDER BY ordinal"
        ):
            has_han = any("\u3400" <= c <= "\u9fff" for c in text)
            if not has_han:
                raise ValueError(f"Chinese training record has no Han characters: {ordinal}")
            group = "han_latin" if any(c.isascii() and c.isalpha() for c in text) else "han_only"
            counts[group] += 1
            sample = dict(id=f"train:zh:{ordinal}", language="zh", reference=text,
                          audio=audio, duration=duration, source=source, source_line=line,
                          stratum=group)
            pool = pools[group]
            if len(pool) < per_group:
                pool.append(sample)
            else:
                position = rng.randrange(counts[group])
                if position < per_group:
                    pool[position] = sample
    if any(n < per_group for n in counts.values()):
        raise ValueError(f"Not enough rows for {per_group} per group: {counts}")
    samples = sorted([r for pool in pools.values() for r in pool], key=lambda r: int(r["id"].split(":")[-1]))
    for row in samples:
        if not Path(row["audio"]).is_file():
            raise FileNotFoundError(row["audio"])
    return samples, counts


def review_rows(rows):
    reviewed = []
    for row in rows:
        ref = normalize_text(row["reference"]).replace(" ", "")
        hyp = normalize_text(row["hypothesis"]).replace(" ", "")
        def plain(text):
            return "".join(c for c in text if not unicodedata.category(c).startswith("P"))
        clean_ref, clean_hyp = plain(ref), plain(hyp)
        errors = edit_distance(ref, hyp)
        clean_errors = edit_distance(clean_ref, clean_hyp)
        reviewed.append({**row, "strict_character_errors": errors,
                         "strict_cer": errors / len(ref) if ref else None,
                         "punctuation_insensitive_cer": clean_errors / len(clean_ref) if clean_ref else None,
                         "reference_has_latin": any(c.isascii() and c.isalpha() for c in ref),
                         "hypothesis_has_latin": any(c.isascii() and c.isalpha() for c in hyp)})
    return sorted(reviewed, key=lambda r: r["punctuation_insensitive_cer"] or 0, reverse=True)


def summary(rows):
    strict = score_predictions(rows)["strict/all"]
    def no_punctuation(text):
        return "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    plain = [
        {**r, "reference": no_punctuation(r["reference"]),
         "hypothesis": no_punctuation(r["hypothesis"])} for r in rows
    ]
    return {
        "strict": strict,
        "punctuation_insensitive": score_predictions(plain)["strict/all"],
        "no_han_predictions": sum(not any("\u3400" <= c <= "\u9fff" for c in r["hypothesis"]) for r in rows),
        "generation_limit_hits": sum(r.get("generation_limit_hit", False) for r in rows),
        "note": "WER uses whitespace splitting; prioritize Chinese CER. No ITN normalization.",
    }


def run(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

    from ._qasr.audio import load_mono_audio

    cfg = json.loads(Path(args.run_config).read_text())
    population = None
    if args.samples:
        samples = read_rows(args.samples)
    elif args.train:
        samples, population = select_training_samples(cfg["index_path"], args.per_group, args.seed)
    else:
        samples = select_samples(cfg["index_path"], args.predictions)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples)
    )
    print(json.dumps({"event": "samples_selected", "count": len(samples),
                      "split": "train" if args.train else "eval", "population": population}), flush=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    processor.tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model, loading = Qwen3ASRForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(loading.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Baseline weights did not load cleanly: {loading}")
    model.to(args.device).eval()
    config_file = Path(args.model) / "config.json"
    (output / "provenance.json").write_text(json.dumps({
        "model": str(Path(args.model).resolve()), "official_repository": json.loads((Path(args.model) / "baseline_source.json").read_text())["repository"]
        if (Path(args.model) / "baseline_source.json").exists() else None,
        "source_snapshot": json.loads((Path(args.model) / "baseline_source.json").read_text())
        if (Path(args.model) / "baseline_source.json").exists() else None,
        "model_config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest() if args.predictions else None,
        "sample_manifest_sha256": hashlib.sha256((output / "samples.jsonl").read_bytes()).hexdigest(),
        "selection": {"split": "train" if args.train else "eval", "seed": args.seed,
                      "per_group": args.per_group if args.train else None, "population": population},
        "torch": str(torch.__version__), "transformers": transformers.__version__,
        "max_new_tokens": args.max_new_tokens, "batch_size": args.batch_size,
        "modes": args.modes, "device": args.device, "dtype": str(dtype),
        "decode": "greedy; no reference text or hotwords supplied",
    }, indent=2))
    scores = {} if args.train else {
        "gqasr": summary([{**r, "hypothesis": r["gqasr_hypothesis"]} for r in samples])
    }
    started = time.monotonic()
    for mode in args.modes:
        records = []
        with (output / f"predictions-{mode}.jsonl").open("x") as destination:
            for start in range(0, len(samples), args.batch_size):
                batch = samples[start:start + args.batch_size]
                waves = [np.asarray(load_mono_audio(r["audio"], 16000), dtype=np.float32) for r in batch]
                inputs = processor.apply_transcription_request(
                    audio=waves, language="Chinese" if mode == "chinese" else None,
                    prompt="", processor_kwargs={"return_tensors": "pt", "padding": True, "sampling_rate": 16000},
                ).to(model.device, model.dtype)
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                        num_beams=1, return_dict_in_generate=False,
                    )
                generated = generated[:, inputs["input_ids"].shape[1]:]
                for sample, ids in zip(batch, generated, strict=True):
                    raw = processor.tokenizer.decode(ids, skip_special_tokens=False)
                    parsed = processor.decode(ids, return_format="parsed")
                    eos = model.generation_config.eos_token_id
                    eos = set(eos if isinstance(eos, list) else [eos])
                    token_ids = ids.tolist()
                    hit = len(token_ids) >= args.max_new_tokens and not any(t in eos for t in token_ids)
                    record = {**sample, "hypothesis": parsed["transcription"],
                              "predicted_language": parsed["language"],
                              "forced_language": "Chinese" if mode == "chinese" else None,
                              "raw_output": raw, "generation_limit_hit": hit}
                    records.append(record)
                    destination.write(json.dumps(record, ensure_ascii=False) + "\n")
                destination.flush()
                print(json.dumps({"event": "baseline_progress", "mode": mode,
                                  "done": len(records), "total": len(samples),
                                  "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
        scores[mode] = summary(records)
        if args.train:
            scores[mode]["by_stratum"] = {
                group: summary([r for r in records if r["stratum"] == group])
                for group in ("han_only", "han_latin") if any(r["stratum"] == group for r in records)
            }
            scores[mode]["sampling_note"] = (
                "Equal-size script strata, not a population-weighted corpus estimate. "
                "Han+Latin is a text heuristic, not proof of spoken code switching. "
                "Forced Chinese may suppress correctly spoken English; use auto for label review."
            )
        (output / f"review-{mode}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_rows(records))
        )
        (output / "scores.json").write_text(json.dumps(scores, indent=2))
        print(f"{mode}: strict CER={scores[mode]['strict']['cer']:.2%}; "
              f"no-Han={scores[mode]['no_han_predictions']}; "
              f"token-limit hits={scores[mode]['generation_limit_hits']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--predictions")
    selection.add_argument("--train", action="store_true")
    parser.add_argument("--per-group", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", help=argparse.SUPPRESS)
    parser.add_argument("--gpus", help="Comma-separated GPU IDs; one independent worker per GPU")
    parser.add_argument("--model", required=True, help="Local official -hf snapshot")
    parser.add_argument("--output", required=True, help="New directory; never overwritten")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--modes", nargs="+", choices=["auto", "chinese"], default=["auto", "chinese"])
    args = parser.parse_args()
    if args.per_group < 1 or args.batch_size < 1 or args.max_new_tokens < 1 or len(set(args.modes)) != len(args.modes):
        parser.error("Positive batch/token counts and unique modes are required")
    if args.gpus:
        from .baseline_parallel import run_parallel
        run_parallel(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
