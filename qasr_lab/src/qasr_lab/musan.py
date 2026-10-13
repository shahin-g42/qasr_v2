"""Build empty-target diagnostic panels from the MUSAN noise partition."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

from .io import digest, file_hash, fresh_directory, write_json
from .panel import canonical_sample, identity_keys


def configured_musan_directory(config):
    value = ((config.augmentation or {}).get("noise_injection") or {}).get("noise_dir")
    if not value:
        raise ValueError("--musan requires augmentation.noise_injection.noise_dir in the YAML")
    return str(Path(value).expanduser().resolve())


def noise_directory(directory):
    root = Path(directory).expanduser().resolve()
    noise = root if root.name == "noise" else root / "noise"
    if not noise.is_dir():
        raise ValueError(f"Expected the MUSAN noise/ partition at {noise}")
    return noise


def prepare_musan_noise(
    directory, output, *, languages, per_language, seed, min_seconds, max_seconds, protected=None
):
    """Split whole recordings before drawing one short crop from each selected file.

    Hash groups keep byte-identical copies together. Language is a decoder prompt
    assignment, not a claimed language for the noise. No production data is edited.
    """
    import numpy as np
    import soundfile as sf

    noise = noise_directory(directory)
    languages = sorted(set(languages))
    if not languages or type(per_language) is not int or per_language < 1:
        raise ValueError("MUSAN requires languages and a positive per-language cap")
    if not 0 <= min_seconds < max_seconds or not math.isfinite(max_seconds):
        raise ValueError("Invalid MUSAN crop duration limits")
    files = sorted(p for p in noise.rglob("*") if p.suffix.lower() == ".wav")
    if not files:
        raise ValueError(f"No MUSAN noise WAV files found at {noise}")
    print(f"Scanning MUSAN noise: {len(files)} files at {noise}", file=sys.stderr, flush=True)
    groups, counts = {}, Counter()
    for index, path in enumerate(files, 1):
        if not path.resolve().is_relative_to(noise):
            raise ValueError(f"Noise file resolves outside the noise partition: {path}")
        info = sf.info(path)
        if info.samplerate != 16000 or info.channels != 1:
            raise ValueError(f"Expected standard MUSAN 16 kHz mono audio: {path}")
        if info.frames == 0 or info.duration < min_seconds:
            counts["too_short"] += 1
            continue
        sha = file_hash(path)
        if sha in groups:
            groups[sha]["aliases"].append(str(path.resolve()))
            counts["duplicate_files"] += 1
        else:
            groups[sha] = {
                "path": str(path.resolve()),
                "aliases": [str(path.resolve())],
                "sha256": sha,
                "frames": info.frames,
            }
        if index % 100 == 0:
            print(f"  {index}/{len(files)} MUSAN files scanned", file=sys.stderr, flush=True)
    ordered = sorted(groups.values(), key=lambda row: digest([seed, row["sha256"], "split"]))
    if len(ordered) < 2:
        raise ValueError("MUSAN needs at least two distinct eligible recordings for a split")
    # A fixed 20% split is independent of sampling caps and training seeds.
    boundary = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * 0.2)))
    partitions = {"validation": ordered[:boundary], "train": ordered[boundary:]}
    out = fresh_directory(output)
    panels, originals, crops = {}, {}, []
    crop_frames = max(1, int(min(max_seconds, max(10.0, min_seconds)) * 16000))
    for split, rows in partitions.items():
        panels[split], originals[split] = [], []
        for index, row in enumerate(rows[: per_language * len(languages)]):
            probes = [
                {
                    "id": row["sha256"],
                    "audio_filepath": alias,
                    "audio_sha256": row["sha256"],
                    "source": "musan-noise",
                    "recording_id": row["sha256"],
                }
                for alias in row["aliases"]
            ]
            if protected and any(
                key in protected for probe in probes for key in identity_keys(probe, "/")
            ):
                raise ValueError(f"MUSAN source overlaps configured validation: {row['path']}")
            originals[split].extend(probes)
            frames = min(crop_frames, row["frames"])
            start = int(digest([seed, row["sha256"], "crop"]), 16) % (row["frames"] - frames + 1)
            audio, rate = sf.read(row["path"], start=start, frames=frames, dtype="float32")
            if (
                len(audio) != frames
                or not np.isfinite(audio).all()
                or file_hash(row["path"]) != row["sha256"]
            ):
                raise ValueError(f"Invalid or changed MUSAN waveform: {row['path']}")
            target = out / f"{split}-{row['sha256'][:24]}.wav"
            sf.write(target, audio, rate, subtype="FLOAT")
            sample = canonical_sample(
                {
                    "id": f"musan-{row['sha256'][:24]}",
                    "audio_filepath": str(target),
                    "text": "",
                    "language": languages[index % len(languages)],
                    "duration": frames / rate,
                    "source": "musan-noise",
                    "recording_id": row["sha256"],
                    "audio_sha256": file_hash(target),
                    "tags": ["musan", "noise", "empty_target"],
                },
                base="/",
            )
            panels[split].append(sample)
            crops.append(
                {
                    "id": sample["id"],
                    "split": split,
                    "source": row,
                    "start_frame": start,
                    "frames": frames,
                    "prompt_language": sample["language"],
                }
            )
    train_keys = set().union(*(identity_keys(row, "/") for row in panels["train"]))
    for row in panels["validation"]:
        if train_keys.intersection(identity_keys(row, "/")):
            raise ValueError(f"MUSAN crop duplicates training audio: {row['id']}")
    provenance = {
        "directory": str(noise),
        "partition": "noise_only",
        "selection_seed": seed,
        "files_scanned": len(files),
        "unique_eligible_recordings": len(ordered),
        "counts": dict(counts),
        "split": "20% validation by seeded source-file hash rank",
        "recording_inventory_sha256": digest(sorted(groups)),
        "per_language_limit": per_language,
        "maximum_crop_seconds": crop_frames / 16000,
        "language_assignment": "round-robin decoder prompt; noise has no transcript language",
        "evaluation_role": "diagnostic; disjoint from this pilot's noise training recordings",
        "warning": "Source recordings may have been used in production augmentation; not a new held-out test set. Noise labels follow MUSAN, not an automated speech audit.",
        "crops": crops,
    }
    write_json(out / "provenance.json", provenance)
    return panels, originals, provenance
