"""Configurable cluster release with a persisted publication and integrity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from .audio_release import package_audio
from .core import sha256, write_json
from .release import publish_bundle, validate_bundle


def ensure_annotations(annotations, archive, *, emit=print):
    """Import the frozen annotation payload atomically, without overwriting local work."""
    annotations, archive = Path(annotations), Path(archive)
    if (annotations / "release.json").is_file():
        return validate_bundle(annotations)
    if annotations.exists() and (not annotations.is_dir() or any(annotations.iterdir())):
        raise ValueError(
            f"Annotation directory exists but has no release.json: {annotations}. "
            "Set ARABIC_REF_ANNOTATIONS_BUNDLE to the complete extracted bundle; "
            "the existing directory will not be overwritten."
        )
    if not archive.is_file():
        raise ValueError(
            f"Annotation bundle is missing: {annotations / 'release.json'}. "
            f"Copy arabic-reference-v4.4-cluster-upload.tar.gz from the Mac to {archive}, "
            "or set ARABIC_REF_ANNOTATIONS_ARCHIVE to its cluster location. "
            "The script will validate and unpack it automatically. "
            "If already extracted elsewhere, set ARABIC_REF_ANNOTATIONS_BUNDLE to that directory. "
            "git pull transfers code, not the repaired transcript data."
        )
    with tarfile.open(archive, "r:*") as handle:
        members = handle.getmembers()
        by_name = {m.name: m for m in members}
        if len(by_name) != len(members):
            raise ValueError("Annotation archive contains duplicate members")
        manifests = [m for m in members if PurePosixPath(m.name).name == "release.json"]
        if len(manifests) != 1 or not manifests[0].isfile():
            raise ValueError("Annotation archive must contain exactly one regular release.json")
        manifest_member = manifests[0]
        prefix = PurePosixPath(manifest_member.name).parent
        if prefix.is_absolute() or ".." in prefix.parts:
            raise ValueError("Unsafe annotation archive path")
        manifest_bytes = handle.extractfile(manifest_member).read()
        manifest = json.loads(manifest_bytes)
        wanted = {*manifest["files"], "release.json"}
        for relative in wanted:
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
                raise ValueError("Unsafe annotation release path")
        inside = {
            str(PurePosixPath(m.name).relative_to(prefix))
            for m in members
            if not m.isdir() and PurePosixPath(m.name).is_relative_to(prefix)
        }
        if inside != wanted:
            raise ValueError("Unexpected or missing files in the archived annotation bundle")
        annotations.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".annotation-import-", dir=annotations.parent
        ) as temp:
            stage = Path(temp) / "bundle"
            stage.mkdir()
            for relative in sorted(wanted):
                member = by_name.get(str(prefix / relative))
                if member is None or not member.isfile():
                    raise ValueError("Annotation payload must contain regular files, not links")
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.extractfile(member) as incoming, destination.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            result = validate_bundle(stage)
            if result["repo_type"] != "dataset":
                raise ValueError("Expected an annotation dataset archive")
            stage.rename(annotations)
    emit(
        json.dumps(
            {
                "event": "annotations_imported",
                "archive": str(archive),
                "annotations": str(annotations),
                "bundle_sha256": result["bundle_sha256"],
            }
        )
    )
    return result


def verify_published_bundle(root, repo_id, revision, api):
    """Verify every published file at the exact commit using Hub content metadata."""
    root = Path(root)
    manifest = validate_bundle(root)
    info = api.repo_info(
        repo_id, repo_type=manifest["repo_type"], revision=revision, files_metadata=True
    )
    if not info.private or info.sha != revision:
        raise ValueError("Published repository privacy or revision differs")
    remote = {f.rfilename: f for f in info.siblings}
    checksums = {**manifest["files"], "release.json": sha256(root / "release.json")}
    for name, checksum in checksums.items():
        entry = remote.get(name)
        path = root / name
        if entry is None or entry.size != path.stat().st_size:
            raise ValueError(f"Published file is missing or has the wrong size: {name}")
        lfs = getattr(entry, "lfs", None)
        if lfs:
            actual = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            if actual != checksum:
                raise ValueError(f"Published LFS content hash differs: {name}")
        else:
            # Git's blob ID includes the file length and a fixed header.
            hasher = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    hasher.update(chunk)
            if getattr(entry, "blob_id", None) != hasher.hexdigest():
                raise ValueError(f"Published Git content hash differs: {name}")
    return {
        "status": "passed",
        "revision": revision,
        "private": True,
        "files_verified": len(checksums),
        "method": "Hub LFS SHA256 and Git blob identities",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["check", "package", "publish", "verify", "all"], nargs="?", default="all"
    )
    args = parser.parse_args(argv)
    root = Path(os.environ["ARABIC_REF_ROOT"])
    annotations = Path(os.environ["ARABIC_REF_ANNOTATIONS_BUNDLE"])
    output = Path(os.environ["ARABIC_REF_DATASET_BUNDLE"])
    repo_id = os.environ["HF_DATASET_REPO"]
    release_id = os.environ["ARABIC_REF_RELEASE_ID"]
    receipt = root / f"publication-{release_id}.json"
    config = {
        "annotations": str(annotations),
        "output": str(output),
        "audio_root": os.environ["ARABIC_REF_AUDIO_ROOT"],
        "repo_id": repo_id,
        "title": os.environ["HF_DATASET_TITLE"],
        "release_id": release_id,
        "datasets": os.environ["REF_DATASETS"].split(),
        "workers": int(os.environ["REF_AUDIO_WORKERS"]),
        "shard_mib": int(os.environ["REF_AUDIO_SHARD_MIB"]),
    }
    print(json.dumps({"event": "release_configuration", **config}, ensure_ascii=False, indent=2))
    if args.stage in {"check", "package", "all"}:
        archive = Path(
            os.environ.get(
                "ARABIC_REF_ANNOTATIONS_ARCHIVE",
                root / "arabic-reference-v4.4-cluster-upload.tar.gz",
            )
        )
        source = ensure_annotations(annotations, archive)
        if not Path(config["audio_root"]).is_dir():
            raise ValueError("Prepared audio directory is absent; configure ARABIC_REF_AUDIO_ROOT")
        if not config["datasets"] or set(config["datasets"]) - set(source["datasets"]):
            raise ValueError("REF_DATASETS contains unknown or no dataset names")
        if args.stage == "check":
            print(
                "Annotation bundle and paths checked. Use package for the exhaustive audio audit."
            )
            return
    if args.stage in {"all", "publish", "verify"}:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        api = HfApi()
        identity = api.whoami()
        print("Authenticated account:", identity["name"])
        # A valid login is required before spending time packaging a release for upload.
    if args.stage in {"all", "package"}:
        result = package_audio(**config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.stage in {"all", "publish"}:
        try:
            info = api.repo_info(repo_id, repo_type="dataset")
        except RepositoryNotFoundError:
            expected_head = None
        else:
            if not info.private:
                raise ValueError("Destination is public; this release must remain private")
            expected_head = info.sha
            print("Updating private repository from commit:", expected_head)
        result = publish_bundle(output, repo_id, execute=True, expected_head=expected_head, api=api)
        result["remote_verification"] = {"status": "pending"}
        write_json(receipt, result)
    if args.stage == "verify":
        result = json.loads(receipt.read_text())
        if result["repo_id"] != repo_id:
            raise ValueError("Publication receipt belongs to a different repository")
    if args.stage in {"all", "publish", "verify"}:
        try:
            result["remote_verification"] = verify_published_bundle(
                output, repo_id, result["commit_sha"], api
            )
        except Exception as exc:
            result["remote_verification"] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            write_json(receipt, result)
            raise
        write_json(receipt, result)
        print(json.dumps(result, indent=2))
        print("Publication receipt:", receipt)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, tarfile.TarError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
