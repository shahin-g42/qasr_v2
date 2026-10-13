"""Read a production training configuration into an isolated lab workspace."""

from __future__ import annotations

import hashlib
import heapq
import json
import sys
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

from .checkpoints import use_qasr_source
from .experiments import DEFAULTS, create_plan
from .io import digest, environment, file_hash, fresh_directory, read_json, write_json
from .panel import canonical_sample, identity_keys, save_panel, validate_samples


def _load_config(path, qasr_source=None):
    source = use_qasr_source(qasr_source)
    from qasr.config import TrainConfig

    config = TrainConfig.from_yaml(path)
    config.validate()
    if not config.eval_manifest_specs:
        raise ValueError(
            "An explicit eval_manifest is required; the lab will not invent a test split"
        )
    return config, source


def _sampling_settings(config, sampling, candidates_per_manifest):
    if sampling not in ("records", "full"):
        raise ValueError("sampling must be records or full")
    if type(candidates_per_manifest) is not int or candidates_per_manifest < 1:
        raise ValueError("candidates_per_manifest must be a positive integer")
    manifests = len(
        config.train_manifest_specs + config.eval_manifest_specs + config.non_speech_manifest_specs
    )
    return {
        "mode": sampling,
        "candidates_per_manifest": candidates_per_manifest if sampling == "records" else None,
        "maximum_records_parsed": manifests * candidates_per_manifest
        if sampling == "records"
        else None,
        "source_read": "one full raw read per uncached manifest"
        if sampling == "records"
        else "full parse of every manifest",
        "selection": "uniform record candidates, then per-manifest quotas within each language"
        if sampling == "records"
        else "global per-language hash ranking",
    }


def inspect_training_config(
    path,
    *,
    qasr_source=None,
    checkpoint=None,
    musan=False,
    sampling="records",
    candidates_per_manifest=2000,
):
    config, source = _load_config(path, qasr_source)
    from .musan import configured_musan_directory

    musan_directory = configured_musan_directory(config) if musan else None
    return {
        "config": str(Path(path).resolve()),
        "qasr_source": source,
        "checkpoint": checkpoint or config.model_name_or_path,
        "train_manifests": len(config.train_manifest_specs),
        "eval_manifests": len(config.eval_manifest_specs),
        "languages": sorted({language for _, language in config.train_manifest_specs}),
        "non_speech_manifests": len(config.non_speech_manifest_specs),
        "musan_directory": musan_directory,
        "musan_partition": "noise_only" if musan else None,
        "duration_range_seconds": [config.min_duration_seconds, config.max_duration_seconds],
        "max_target_tokens": config.max_target_length,
        "production_world_size": config.expected_world_size,
        "test_panel": None,
        "evaluation_role": "Existing configured validation data; not a new held-out test set.",
        "sampling": _sampling_settings(config, sampling, candidates_per_manifest),
    }


def _identity_index(samples):
    result = defaultdict(set)
    for sample in samples:
        for key in identity_keys(sample, "/"):
            result[key].add(sample["id"])
    return result


def _collect(
    specs,
    config,
    tokenizer,
    limit,
    *,
    empty_target_ok=False,
    protected=None,
    sampling="full",
    candidates_per_manifest=2000,
    sample_cache=None,
):
    from qasr.data import _UnusableTranscript, parse_record

    heaps, selected = defaultdict(list), {}
    sources, overlaps = [], []
    audio_root = Path(config.audio_root).expanduser().resolve() if config.audio_root else None
    quotas = {}
    if sampling == "records":
        by_language = defaultdict(list)
        for filename, language in specs:
            by_language[language].append(str(Path(filename).expanduser().resolve()))
        for language, filenames in by_language.items():
            for index, filename in enumerate(sorted(filenames)):
                quotas[(filename, language)] = limit // len(filenames) + (
                    index < limit % len(filenames)
                )
        specs = sorted(specs, key=lambda spec: (spec[1], str(Path(spec[0]).resolve())))
    seen_specs = set()
    for filename, language in specs:
        path = Path(filename).expanduser().resolve()
        if (path, language) in seen_specs:
            raise ValueError(f"Manifest configured twice for {language}: {path}")
        seen_specs.add((path, language))
        sample_metadata = None
        quota = quotas[(str(path), language)] if sampling == "records" else limit
        if sampling == "records":
            from .manifest_sampling import sample_record_lines

            records, sample_metadata = sample_record_lines(
                path,
                limit=candidates_per_manifest,
                seed=config.data_seed,
                cache_directory=sample_cache,
            )
            print(
                f"Parsing {len(records):,} sampled {language} records; retain up to {quota}: {path}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(f"Scanning all {language} records: {path}", file=sys.stderr, flush=True)
        sha = hashlib.sha256()
        counts = Counter()
        with path.open("rb") if sampling == "full" else nullcontext() as handle:
            lines = (
                enumerate(handle, 1)
                if sampling == "full"
                else ((row["line_number"], row["record_json"]) for row in records)
            )
            for line_number, line in lines:
                if sampling == "full":
                    sha.update(line)
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                if counts["input_rows"] % 100000 == 0:
                    print(f"  {counts['input_rows']:,} rows scanned", file=sys.stderr, flush=True)
                try:
                    raw = json.loads(line)
                    parsed = parse_record(
                        raw,
                        manifest_path=path,
                        line_number=line_number,
                        audio_root=audio_root,
                        language=language,
                        empty_target_ok=empty_target_ok,
                    )
                except _UnusableTranscript:
                    counts["skipped_unusable_text"] += 1
                    continue
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                # Use the production parser's language assignment and /vast path remapping.
                audio = str(Path(parsed["audio_path"]).resolve())
                probe = {**raw, "audio_filepath": audio}
                if protected:
                    for key in identity_keys(probe, "/"):
                        for sample_id in protected.get(key, ()):
                            if len(overlaps) < 30:
                                overlaps.append(
                                    {
                                        "sample_id": sample_id,
                                        "kind": key[0],
                                        "manifest": str(path),
                                        "line": line_number,
                                    }
                                )
                duration = parsed["duration"]
                if duration is None:
                    import soundfile as sf

                    duration = sf.info(audio).duration
                    counts["durations_read_from_audio"] += 1
                if not config.min_duration_seconds <= duration <= config.max_duration_seconds:
                    counts["skipped_duration"] += 1
                    continue
                if empty_target_ok and parsed["text"]:
                    raise ValueError(
                        f"{path}:{line_number}: non_speech_manifest contains speech text"
                    )
                sample_id = digest(audio)[:24]
                rank = int(digest([config.data_seed, sample_id]), 16)
                heap = heaps[(str(path), language) if sampling == "records" else language]
                if sample_id in selected:
                    old = selected[sample_id]
                    if (
                        old["language"] != language
                        or old["text"] != parsed["text"]
                        or old["duration"] != duration
                    ):
                        raise ValueError(f"Conflicting annotations for selected audio: {audio}")
                    counts["duplicate_selected_audio"] += 1
                    continue
                if quota == 0 or (len(heap) == quota and -rank <= heap[0][0]):
                    continue
                # Tokenize only candidates that can enter the bounded selection.
                token_ids = tokenizer.encode(parsed["text"], add_special_tokens=False)
                if len(token_ids) > config.max_target_length:
                    counts["candidate_transcripts_rejected"] += 1
                    continue
                sample = canonical_sample(
                    {
                        **probe,
                        "id": sample_id,
                        "text": parsed["text"],
                        "duration": duration,
                        "language": language,
                        "stratum": "non_speech" if empty_target_ok else "speech",
                    },
                    base="/",
                )
                sample.update(manifest_filepath=str(path), manifest_line=line_number)
                entry = (-rank, sample_id, sample)
                if len(heap) == quota:
                    removed = heapq.heapreplace(heap, entry)[2]
                    del selected[removed["id"]]
                else:
                    heapq.heappush(heap, entry)
                selected[sample_id] = sample
        if sampling == "records":
            kept = len(heaps[(str(path), language)])
            print(
                f"  Selected {kept}/{quota} records from sampled candidates",
                file=sys.stderr,
                flush=True,
            )
        sources.append(
            {
                "path": str(path),
                "language": language,
                "sha256": sample_metadata["sha256"] if sample_metadata else sha.hexdigest(),
                "counts": dict(counts),
                "sampling": sample_metadata or {"method": "full", "all_records_considered": True},
                "selected_records": len(heaps[(str(path), language)])
                if sampling == "records"
                else None,
                "sample_quota": quota if sampling == "records" else None,
            }
        )
    samples = sorted(selected.values(), key=lambda sample: sample["id"])
    validate_samples(samples)
    expected_languages = {language for _, language in specs}
    missing = expected_languages - {sample["language"] for sample in samples}
    if missing:
        raise ValueError(f"No eligible samples for configured languages: {sorted(missing)}")
    return samples, sources, overlaps


def prepare_from_training_config(
    path,
    output,
    *,
    qasr_source=None,
    checkpoint=None,
    train_per_language=500,
    eval_per_language=50,
    non_speech_per_language=50,
    seeds=None,
    max_steps=200,
    hash_audio=False,
    musan=False,
    sampling="records",
    candidates_per_manifest=2000,
    sample_cache=None,
):
    for value in (train_per_language, eval_per_language, non_speech_per_language, max_steps):
        if type(value) is not int or value < 1:
            raise ValueError("Sample limits and max_steps must be positive integers")
    config, source = _load_config(path, qasr_source)
    sampling_settings = _sampling_settings(config, sampling, candidates_per_manifest)
    musan_directory = None
    if musan:
        from .musan import configured_musan_directory, noise_directory, prepare_musan_noise

        if config.non_speech_manifest_specs:
            raise ValueError("Use either --musan or configured non_speech_manifest, not both")
        musan_directory = configured_musan_directory(config)
        noise_directory(musan_directory)
    seeds = [config.seed, config.seed + 1] if seeds is None else seeds
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("Seeds must be distinct nonnegative integers")
    checkpoint = str(Path(checkpoint or config.model_name_or_path).expanduser().resolve())
    from qasr import QASRProcessor
    from qasr.config import validate_local_checkpoint

    validate_local_checkpoint(checkpoint)
    if read_json(Path(checkpoint) / "config.json").get("model_type") != "qasr":
        raise ValueError("Select a vanilla QASR starting checkpoint")
    all_specs = (
        config.train_manifest_specs + config.eval_manifest_specs + config.non_speech_manifest_specs
    )
    for filename, _ in all_specs:
        if not Path(filename).expanduser().is_file():
            raise FileNotFoundError(filename)
    out = fresh_directory(output)
    sample_cache = (
        str(Path(sample_cache).expanduser().resolve())
        if sample_cache
        else str(out.parent / "manifest-sample-cache")
    )
    collect_options = {
        "sampling": sampling,
        "candidates_per_manifest": candidates_per_manifest,
        "sample_cache": sample_cache,
    }
    write_json(
        out / "source-config.json",
        {
            "path": str(Path(path).resolve()),
            "sha256": file_hash(path),
            "values": config.to_dict(),
            "checkpoint_selected": checkpoint,
            "musan_directory_selected": musan_directory,
            "sampling": sampling_settings,
            "sample_cache": sample_cache if sampling == "records" else None,
            "environment": environment(),
        },
    )
    try:
        print(f"Loading processor only: {checkpoint}", file=sys.stderr, flush=True)
        processor = QASRProcessor.from_pretrained(checkpoint, trust_remote_code=False)
        validation, eval_sources, _ = _collect(
            config.eval_manifest_specs,
            config,
            processor.tokenizer,
            eval_per_language,
            **collect_options,
        )
        # Hash validation before the training scan so supplied training hashes can match copies.
        if hash_audio:
            for sample in validation:
                actual = file_hash(sample["audio_filepath"])
                if sample.get("audio_sha256") and sample["audio_sha256"] != actual:
                    raise ValueError(f"Audio checksum mismatch: {sample['id']}")
                sample["audio_sha256"] = actual
        protected = _identity_index(validation)
        musan_provenance, musan_panels = None, None
        if musan:
            musan_panels, originals, musan_provenance = prepare_musan_noise(
                musan_directory,
                out / "musan-audio",
                languages={language for _, language in config.train_manifest_specs},
                per_language=non_speech_per_language,
                seed=config.data_seed,
                min_seconds=config.min_duration_seconds,
                max_seconds=config.max_duration_seconds,
                protected=protected,
            )
            # Compare parsed speech candidates with original noise paths/hashes,
            # not just derived crop filenames. Full mode checks every speech row.
            protected = _identity_index(
                [
                    *validation,
                    *originals["train"],
                    *originals["validation"],
                    *musan_panels["train"],
                    *musan_panels["validation"],
                ]
            )
        train, train_sources, overlaps = _collect(
            config.train_manifest_specs,
            config,
            processor.tokenizer,
            train_per_language,
            protected=protected,
            **collect_options,
        )
        silence, silence_sources = [], []
        if musan_panels:
            silence = musan_panels["train"]
            silence_sources = [{"musan_provenance": str(out / "musan-audio/provenance.json")}]
        if config.non_speech_manifest_specs:
            silence, silence_sources, silence_overlaps = _collect(
                config.non_speech_manifest_specs,
                config,
                processor.tokenizer,
                non_speech_per_language,
                empty_target_ok=True,
                protected=protected,
                **collect_options,
            )
            overlaps.extend(silence_overlaps)
            validate_samples([*train, *silence])
        if overlaps:
            raise ValueError(f"Configured training/validation overlap: {overlaps[:30]}")
        for sample in [*train, *validation, *silence]:
            if not Path(sample["audio_filepath"]).is_file():
                raise FileNotFoundError(sample["audio_filepath"])
        if hash_audio:
            for sample in [*train, *silence]:
                actual = file_hash(sample["audio_filepath"])
                if sample.get("audio_sha256") and sample["audio_sha256"] != actual:
                    raise ValueError(f"Audio checksum mismatch: {sample['id']}")
                sample["audio_sha256"] = actual
        # Hashes computed after collection can also reveal copied sampled audio.
        evaluation_keys = set(_identity_index(validation))
        if musan_panels:
            evaluation_keys.update(_identity_index(musan_panels["validation"]))
        for sample in [*train, *silence]:
            if evaluation_keys.intersection(identity_keys(sample, "/")):
                raise ValueError(f"Selected training/validation audio overlap: {sample['id']}")
        full_overlap_check = all(
            source.get("sampling", {}).get("all_records_considered", False)
            for source in train_sources + (silence_sources if not musan else [])
        )
        provenance = {
            "training_config": str(Path(path).resolve()),
            "config_sha256": file_hash(path),
            "selection_seed": config.data_seed,
            "language_assignment": "configuration manifest groups",
            "data_parser": "qasr.data.parse_record",
            "selected_audio_hashes_computed": hash_audio,
            "sampling": sampling_settings,
        }
        save_panel(
            train,
            out / "train-panel",
            {
                **provenance,
                "sources": train_sources,
                "per_language_limit": train_per_language,
            },
        )
        save_panel(
            validation,
            out / "validation-panel",
            {
                **provenance,
                "sources": eval_sources,
                "per_language_limit": eval_per_language,
                "role": "existing_configured_validation",
                "warning": "Not a new test set; prior checkpoint exposure is not established.",
                "leakage": {
                    "status": "checked" if full_overlap_check else "sampled_candidates_only",
                    "examples": [],
                    "scope": "Selected validation vs all parsed configured training/non-speech rows; paths, supplied hashes and corpus-qualified identities."
                    if full_overlap_check
                    else "Selected validation vs sampled training/non-speech candidates only; the unsampled corpus is not checked.",
                    "excluded_manifests": train_sources + silence_sources,
                },
            },
        )
        silence_panel = None
        if silence:
            silence_panel = out / "non-speech-panel"
            save_panel(
                silence,
                silence_panel,
                {
                    **provenance,
                    "sources": silence_sources,
                    "per_language_limit": non_speech_per_language,
                    "musan": musan_provenance,
                },
            )
        if musan_panels:
            save_panel(
                musan_panels["validation"],
                out / "non-speech-validation-panel",
                {**provenance, "musan": musan_provenance},
            )
        base = {
            **DEFAULTS,
            "id": "control",
            "checkpoint": checkpoint,
            "qasr_source": str(Path(source).parent.parent),
            "train_panel": str(out / "train-panel"),
            "eval_panel": str(out / "validation-panel"),
            "device": "cuda:0",
            "precision": "bfloat16" if config.bf16 else "float32",
            "min_audio_seconds": config.min_duration_seconds,
            "max_audio_seconds": config.max_duration_seconds,
            "max_target_tokens": config.max_target_length,
            "encoder_lr": config.learning_rate,
            "projector_lr": config.learning_rate,
            "decoder_lr": config.learning_rate * config.llm_lr_factor,
            "weight_decay": config.weight_decay,
            "max_grad_norm": config.max_grad_norm,
            "gradient_checkpointing": config.gradient_checkpointing,
            "max_steps": max_steps,
            "warmup_steps": min(20, max_steps - 1),
        }
        plans = []
        for seed in seeds:
            base_path = out / f"base-{seed}.json"
            write_json(base_path, {**base, "seed": seed, "output": str(out / f"base-run-{seed}")})
            plan = create_plan(base_path, out / f"seed-{seed}", non_speech_panel=silence_panel)
            plans.append(plan)
            for entry in plan["experiments"]:
                experiment = read_json(entry["config"])
                write_json(
                    out / f"eval-{entry['id']}-{seed}.json",
                    {
                        "id": f"{entry['id']}-{seed}",
                        "backend": "qasr-lab",
                        "checkpoint": str(Path(experiment["output"]) / "checkpoint"),
                        "qasr_source": base["qasr_source"],
                        "device": "cuda:0",
                        "dtype": base["precision"],
                        "warmup_samples": 3,
                        "max_new_tokens": config.eval_generation_max_new_tokens,
                        "max_audio_seconds": config.max_duration_seconds,
                    },
                )
        for name, samples in (("train", train), ("validation", validation)):
            counts = Counter()
            smoke_samples = []
            for sample in samples:
                if counts[sample["language"]] < 2:
                    smoke_samples.append(sample)
                    counts[sample["language"]] += 1
            save_panel(
                smoke_samples,
                out / f"smoke-{name}-panel",
                {
                    **provenance,
                    "role": "smoke_subset",
                },
            )
        smoke = read_json(out / f"seed-{seeds[0]}" / "ctc.json")
        smoke.update(
            max_steps=2,
            warmup_steps=0,
            output=str(out / "smoke"),
            train_panel=str(out / "smoke-train-panel"),
            eval_panel=str(out / "smoke-validation-panel"),
        )
        write_json(out / "smoke.json", smoke)
        write_json(
            out / "eval-initial.json",
            {
                "id": "initial-qasr",
                "backend": "qasr",
                "checkpoint": checkpoint,
                "qasr_source": base["qasr_source"],
                "device": "cuda:0",
                "dtype": base["precision"],
                "warmup_samples": 3,
                "max_new_tokens": config.eval_generation_max_new_tokens,
                "max_audio_seconds": config.max_duration_seconds,
            },
        )
        result = {
            **inspect_training_config(
                path,
                qasr_source=qasr_source,
                checkpoint=checkpoint,
                musan=musan,
                sampling=sampling,
                candidates_per_manifest=candidates_per_manifest,
            ),
            "status": "prepared",
            "output": str(out),
            "seeds": seeds,
            "sample_cache": sample_cache if sampling == "records" else None,
            "full_training_overlap_check": full_overlap_check,
            "selected_train": len(train),
            "selected_validation": len(validation),
            "selected_non_speech": len(silence),
            "selected_non_speech_validation": len(musan_panels["validation"])
            if musan_panels
            else 0,
            "experiments": sum(len(plan["experiments"]) for plan in plans),
            "pilot_settings": {
                "steps": max_steps,
                "warmup_steps": base["warmup_steps"],
                "batch_size": 1,
                "gradient_accumulation": 8,
                "scheduler": "cosine",
            },
            "not_reproduced": [
                "production distributed topology",
                "full training schedule",
                "weighted language quotas",
                "augmentation",
                "online code switching",
                "label smoothing",
                "production optimizer/resume state",
            ],
        }
        write_json(out / "prepared.json", result)
        return result
    except Exception as exc:
        write_json(out / "failure.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
