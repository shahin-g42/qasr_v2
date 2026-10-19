"""Configurable cluster release with a persisted publication and integrity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .audio_release import package_audio
from .core import sha256, write_json
from .release import publish_bundle, validate_bundle


def verify_published_bundle(root, repo_id, revision, api):
    """Verify every published file at the exact commit using Hub content metadata."""
    root = Path(root)
    manifest = validate_bundle(root)
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision, files_metadata=True)
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
        source = validate_bundle(annotations)
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
    main()
