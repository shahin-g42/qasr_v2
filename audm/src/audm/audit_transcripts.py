"""Stream manifest transcripts through the exported tokenizer; never load audio/weights."""

import argparse
import hashlib
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from qasr.data import _UnusableTranscript, parse_record
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from .training_config import load_recipe


def summarize(counts):
    result = dict(counts)
    for scope in ("all", "duration_eligible"):
        n = counts[f"{scope}_transcripts"]
        for limit in (512, 1024):
            result[f"{scope}_above_{limit}_percent"] = (
                100 * counts[f"{scope}_above_{limit}"] / n if n else 0.0
            )
    return result


def audit(config, model, output, language="ml", split="train", batch_size=1024, filter_above=None):
    if batch_size < 1 or (filter_above is not None and filter_above < 1):
        raise ValueError("Batch size and filter threshold must be positive")
    model, output = Path(model).resolve(), Path(output).resolve()
    cfg, _, receipt = load_recipe(config, model=model, output=output)
    specs = cfg.train_manifest_specs if split == "train" else cfg.eval_manifest_specs
    paths = [Path(path).resolve() for path, lang in specs if lang == language]
    if not paths:
        raise ValueError(f"No {language} manifests in {split}")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate manifest paths; audit them separately to avoid double counting")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model, local_files_only=True)
    output.mkdir(parents=True, exist_ok=False)
    total, manifests = Counter(), []
    keys = ["records", "unusable_text", "missing_duration", "filtered_out", "written"]
    keys += [
        f"{scope}_{name}"
        for scope in ("all", "duration_eligible")
        for name in ("transcripts", "above_512", "above_1024")
    ]
    with ExitStack() as stack:
        rejected = stack.enter_context((output / "over_limit.jsonl").open("w", encoding="utf-8"))
        for number, path in enumerate(paths):
            counts = Counter(dict.fromkeys(keys, 0))
            digest = hashlib.sha256()
            maximum = 0
            filtered_path = output / f"{number:03d}-{path.name}" if filter_above else None
            filtered = (
                stack.enter_context(filtered_path.open("w", encoding="utf-8"))
                if filtered_path
                else None
            )
            pending = []

            def flush():
                nonlocal maximum
                if not pending:
                    return
                encoded = tokenizer(
                    [item[2]["text"] for item in pending],
                    add_special_tokens=False,
                    truncation=False,
                    padding=False,
                    return_attention_mask=False,
                )["input_ids"]
                for (line_number, record, parsed, eligible), ids in zip(
                    pending, encoded, strict=True
                ):
                    n = len(ids)
                    maximum = max(maximum, n)
                    for scope in ["all", "duration_eligible"] if eligible else ["all"]:
                        counts[f"{scope}_transcripts"] += 1
                        for limit in (512, 1024):
                            counts[f"{scope}_above_{limit}"] += int(n > limit)
                    if n > 512 or (filter_above is not None and n > filter_above):
                        rejected.write(
                            json.dumps(
                                dict(
                                    manifest=str(path),
                                    line=line_number,
                                    tokens=n,
                                    audio_path=parsed["audio_path"],
                                    duration=parsed["duration"],
                                    duration_eligible=eligible,
                                    text=parsed["text"],
                                ),
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    if filtered is not None:
                        if n > filter_above:
                            counts["filtered_out"] += 1
                        else:
                            # Relocation must not change resolution of relative audio paths.
                            record["audio_filepath"] = parsed["audio_path"]
                            filtered.write(json.dumps(record, ensure_ascii=False) + "\n")
                            counts["written"] += 1
                pending.clear()

            with path.open("rb") as source, tqdm(desc=path.name, unit=" records") as progress:
                for line_number, raw in enumerate(source, 1):
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    counts["records"] += 1
                    progress.update(1)
                    record = json.loads(raw)
                    try:
                        parsed = parse_record(
                            record,
                            manifest_path=path,
                            line_number=line_number,
                            audio_root=Path(cfg.audio_root).expanduser()
                            if cfg.audio_root
                            else None,
                            language=language,
                        )
                    except _UnusableTranscript:
                        flush()
                        counts["unusable_text"] += 1
                        # Only filter on token length; preserve already skipped text records.
                        if filtered is not None:
                            filtered.write(raw.decode("utf-8").rstrip("\r\n") + "\n")
                            counts["written"] += 1
                        continue
                    duration = parsed["duration"]
                    counts["missing_duration"] += int(duration is None)
                    eligible = (
                        duration is None
                        or cfg.min_duration_seconds <= duration <= cfg.max_duration_seconds
                    )
                    pending.append((line_number, record, parsed, eligible))
                    if len(pending) >= batch_size:
                        flush()
                flush()
            total.update(counts)
            manifests.append(
                dict(
                    path=str(path),
                    sha256=digest.hexdigest(),
                    max_tokens=maximum,
                    counts=summarize(counts),
                    filtered_path=str(filtered_path) if filtered_path else None,
                )
            )
    report = dict(
        language=language,
        split=split,
        model=str(model),
        tokenizer_sha256=hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
        recipe=receipt,
        filter_above=filter_above,
        notes=[
            "Token counts exclude special tokens, language markers, audio placeholders and EOS.",
            "Duration eligibility uses manifest metadata; unknown durations remain eligible.",
            "No audio files were loaded or checked. Counts are manifest rows, not unique audio files.",
            "Filtered copies remove only transcripts above filter_above; training inputs are unchanged.",
        ],
        totals=summarize(total),
        manifests=manifests,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="audm/configs/audflash_adapter_1node.yaml")
    parser.add_argument("--model", required=True, help="Local AUD export/checkpoint tokenizer")
    parser.add_argument("--output", required=True, help="New audit directory; never overwritten")
    parser.add_argument("--language", default="ml")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--filter-above", type=int, help="Write new manifest copies excluding longer transcripts"
    )
    report = audit(**vars(parser.parse_args()))
    print(json.dumps({"totals": report["totals"], "manifests": report["manifests"]}, indent=2))


if __name__ == "__main__":
    main()
