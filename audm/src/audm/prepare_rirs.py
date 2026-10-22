"""Pinned OpenSLR SLR28 download and mono RIR manifests; no Kaldi dependency."""

import argparse
import hashlib
import io
import json
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import soundfile as sf

from .prepare_non_speech import dump, jsonl, new_directory, sha256

URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
CHECKSUM_URL = "https://www.openslr.org/resources/28/checksum.md5"
ARCHIVE_MD5 = "e6f48e257286e05de56413b4779d8ffb"


def check_archive(path):
    digest = hashlib.md5()  # Publisher's integrity checksum, over HTTPS.
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"SLR28 checksum mismatch: {path}; do not prepare this archive")


def download(root):
    raw = Path(root).resolve() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "rirs_noises.zip"
    partial = raw / "rirs_noises.zip.part"
    if not archive.exists() and partial.exists():
        # A process may have exited after download but before publishing the
        # completed file. Avoid asking curl to resume past EOF in that case.
        try:
            check_archive(partial)
        except ValueError:
            pass
        else:
            partial.rename(archive)
    if not archive.exists():
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "5",
                "--connect-timeout",
                "30",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                URL,
            ],
            check=True,
        )
        check_archive(partial)
        partial.rename(archive)
    else:
        check_archive(archive)
    result = dict(
        url=URL,
        publisher_checksum_url=CHECKSUM_URL,
        md5=ARCHIVE_MD5,
        sha256=sha256(archive),
        bytes=archive.stat().st_size,
    )
    dump(raw / "download.json", result)
    return result


def rir_kind(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe ZIP path: {name}")
    if path.suffix.lower() != ".wav":
        return None
    if len(path.parts) >= 4 and path.parts[:2] == ("RIRS_NOISES", "simulated_rirs"):
        if path.parts[2] in ("smallroom", "mediumroom", "largeroom"):
            return path.parts[2]
    if path.parts[:2] == ("RIRS_NOISES", "real_rirs_isotropic_noises"):
        if "rir" in path.stem.lower() and "noise" not in path.stem.lower():
            return "real"
    return None  # Point-source and isotropic noises are NOT impulse responses.


def prepare(root):
    root = Path(root).resolve()
    archive = root / "raw/rirs_noises.zip"
    check_archive(archive)
    counts = Counter()
    with new_directory(root / "prepared-v1") as (stage, destination):
        with zipfile.ZipFile(archive) as source, (stage / "rir.jsonl").open("w") as manifest:
            names = set()
            for member in source.infolist():
                kind = rir_kind(member.filename)
                if member.filename in names or stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Duplicate or symlink ZIP member: {member.filename}")
                names.add(member.filename)
                if member.is_dir():
                    continue
                basename = PurePosixPath(member.filename).name.lower()
                if basename.startswith(("readme", "license", "copying", "notice")):
                    target = stage / "source_metadata" / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                if kind is None:
                    continue
                wave, rate = sf.read(
                    io.BytesIO(source.read(member)), dtype="float32", always_2d=True
                )
                if rate != 16000 or not len(wave) or not np.isfinite(wave).all():
                    raise ValueError(f"Invalid/unsupported RIR audio: {member.filename}")
                counts[f"source:{kind}"] += 1
                for channel in range(wave.shape[1]):
                    mono = wave[:, channel]
                    if not np.any(mono):
                        counts["excluded:zero_channel"] += 1
                        continue
                    # Retain each microphone separately: averaging can cancel RIR phases.
                    relative = Path("audio") / (member.filename + f".channel-{channel:03d}.wav")
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(target, mono, rate, subtype="FLOAT")
                    row = dict(
                        audio_filepath=str(destination / relative),
                        duration=len(mono) / rate,
                        sampling_rate=rate,
                        channels=1,
                        source="OpenSLR-SLR28",
                        kind=kind,
                        archive_member=member.filename,
                        source_channel=channel,
                        audio_sha256=sha256(target),
                    )
                    manifest.write(json.dumps(row) + "\n")
                    counts[f"prepared:{kind}"] += 1
                if sum(v for k, v in counts.items() if k.startswith("source:")) % 1000 == 0:
                    print(json.dumps(dict(counts)), flush=True)
        if not any(k.startswith("prepared:") for k in counts):
            raise ValueError("Archive contains no usable room impulse responses")
        report = dict(
            source_url=URL,
            publisher_checksum_url=CHECKSUM_URL,
            archive_md5=ARCHIVE_MD5,
            archive_sha256=sha256(archive),
            counts=dict(counts),
            manifest_sha256=sha256(stage / "rir.jsonl"),
            manifest=str(destination / "rir.jsonl"),
            channel_policy="Each nonzero microphone channel is a separate mono RIR",
            selection="All simulated rooms and real RIRs; no candidate count caps; noises excluded",
        )
        dump(stage / "report.json", report)
    return report


def verify(root):
    prepared = Path(root).resolve() / "prepared-v1"
    report = json.loads((prepared / "report.json").read_text())
    manifest = prepared / "rir.jsonl"
    if report["archive_md5"] != ARCHIVE_MD5 or sha256(manifest) != report["manifest_sha256"]:
        raise ValueError("RIR preparation source/manifest changed")
    count = 0
    for row in jsonl(manifest):
        path = Path(row["audio_filepath"])
        if not path.is_relative_to(prepared / "audio") or sha256(path) != row["audio_sha256"]:
            raise ValueError(f"RIR asset changed or outside preparation: {path}")
        count += 1
    if not count:
        raise ValueError("Empty RIR manifest")
    return dict(manifest=str(manifest), verified_rirs=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare", "verify", "all"])
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    if args.command == "all":
        download(args.root)
        result = (
            verify(args.root) if Path(args.root, "prepared-v1").exists() else prepare(args.root)
        )
    else:
        result = {"download": download, "prepare": prepare, "verify": verify}[args.command](
            args.root
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
