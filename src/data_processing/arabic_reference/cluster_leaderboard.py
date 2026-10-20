"""Build, publish and verify a private leaderboard tied to a published dataset."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import sys
from pathlib import Path

from .cluster_release import ensure_annotations, verify_published_bundle
from .core import digest, sha256, write_json
from .leaderboard import build_leaderboard
from .leaderboard_inputs import dataset_pin
from .release import publish_bundle, validate_bundle


def require_dependencies(stage):
    """Check the selected interpreter before reading bundles or contacting the Hub."""
    required = {}
    if stage in {"build", "all"}:
        required.update({"pyarrow.parquet": "pyarrow", "rapidfuzz.distance": "rapidfuzz"})
    if stage in {"check", "publish", "verify", "all"}:
        required["huggingface_hub"] = "huggingface-hub"
    missing, errors = [], []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError as exc:
            missing.append(package)
            errors.append(f"{package}: {exc}")
    if missing:
        command = shlex.join([sys.executable, "-m", "pip", "install", *missing])
        raise ValueError(
            f"Leaderboard dependencies unavailable in {sys.executable} "
            f"({'; '.join(errors)}).\nRun:\n  {command}\n"
            "Alternatively, set PYTHON to the Python executable in your prepared environment."
        )


def verify_dataset_binding(repo_id, revision, annotations, api, *, download=None):
    """Fetch only small frozen metadata, never the audio, before publishing scores."""
    from huggingface_hub import hf_hub_download

    dataset_pin(repo_id, revision)
    info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
    if not info.private or info.sha != revision:
        raise ValueError("Dataset must be private and available at the exact pinned revision")
    download = download or hf_hub_download
    paths = {
        name: Path(download(repo_id, name, repo_type="dataset", revision=revision))
        for name in ("release.json", "provenance.json")
    }
    release = json.loads(paths["release.json"].read_text())
    provenance = json.loads(paths["provenance.json"].read_text())
    source = validate_bundle(annotations)
    if release.get("bundle_sha256") != digest(
        {k: v for k, v in release.items() if k != "bundle_sha256"}
    ):
        raise ValueError("Published dataset release manifest digest differs")
    if release.get("files", {}).get("provenance.json") != sha256(paths["provenance.json"]):
        raise ValueError("Published dataset provenance digest differs")
    if (
        release.get("repo_type") != "dataset"
        or release.get("visibility") != "private"
        or not release.get("audio_included")
        or release.get("quality_status") != "passed"
        or release.get("annotations_bundle_sha256") != source["bundle_sha256"]
        or release.get("manifest_digest") != source["manifest_digest"]
        or release.get("datasets") != source["datasets"]
        or release.get("counts") != source["counts"]
        or provenance.get("annotations_bundle_sha256") != source["bundle_sha256"]
    ):
        raise ValueError(
            "Published audio dataset does not match the leaderboard annotations/coverage"
        )
    return {
        "status": "passed",
        "repo_id": repo_id,
        "revision": revision,
        "annotations_bundle_sha256": source["bundle_sha256"],
        "dataset_bundle_sha256": release["bundle_sha256"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["build", "check", "publish", "verify", "all"], nargs="?", default="all"
    )
    args = parser.parse_args(argv)
    require_dependencies(args.stage)
    root = Path(os.environ["ARABIC_REF_ROOT"])
    output = Path(os.environ["ARABIC_REF_SPACE_BUNDLE"])
    annotations = Path(os.environ["ARABIC_REF_ANNOTATIONS_BUNDLE"])
    repo_id = os.environ["HF_SPACE_REPO"]
    revision = os.environ.get("HF_DATASET_REVISION", "")
    dataset_repo = os.environ["HF_DATASET_REPO"]
    release_id = os.environ["ARABIC_REF_BOARD_RELEASE_ID"]
    submissions = os.environ.get("ARABIC_REF_SUBMISSIONS") or None
    latency_root = os.environ.get("ARABIC_REF_LATENCY_ROOT") or None
    receipt = root / f"publication-{release_id}.json"
    dataset_receipt = Path(os.environ["ARABIC_REF_DATASET_RECEIPT"])
    if dataset_receipt.exists():
        published = json.loads(dataset_receipt.read_text())
        if published["repo_id"] != dataset_repo or not published.get("executed"):
            raise ValueError(
                "Dataset publication receipt has a different repository or was not executed"
            )
        if revision and revision != published["commit_sha"]:
            raise ValueError("HF_DATASET_REVISION differs from the publication receipt")
        revision = published["commit_sha"]
    dataset_pin(dataset_repo, revision)
    print(
        json.dumps(
            {
                "event": "leaderboard_configuration",
                "space_repo": repo_id,
                "dataset_repo": dataset_repo,
                "dataset_revision": revision,
                "annotations": str(annotations),
                "output": str(output),
                "title": os.environ["HF_SPACE_TITLE"],
                "release_id": release_id,
            },
            indent=2,
        )
    )
    ensure_annotations(annotations, os.environ["ARABIC_REF_ANNOTATIONS_ARCHIVE"])
    if args.stage in {"check", "publish", "verify", "all"}:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import RepositoryNotFoundError

        api = HfApi()
        print("Authenticated account:", api.whoami()["name"])
        binding = verify_dataset_binding(dataset_repo, revision, annotations, api)
        print(json.dumps({"event": "dataset_binding_verified", **binding}, indent=2))
    if args.stage in {"build", "all"}:
        result = build_leaderboard(
            Path(os.environ["ARABIC_REF_PLAN"]),
            output,
            annotations=annotations,
            title=os.environ["HF_SPACE_TITLE"],
            release_id=release_id,
            dataset_repo=dataset_repo,
            dataset_revision=revision,
            submissions=submissions,
            latency_root=latency_root,
            timing_runs=[(name, os.environ[key]) for name, key in (("qasr", "ARABIC_REF_QASR_RUN"), ("gqasr", "ARABIC_REF_GQASR_RUN")) if os.environ.get(key) and Path(os.environ[key]).is_dir()],
        )
        print(json.dumps(result, indent=2))
    if args.stage in {"publish", "verify", "all"}:
        bundle = validate_bundle(output)
        if (
            bundle.get("repo_type") != "space"
            or bundle.get("dataset_release") != dataset_pin(dataset_repo, revision)
            or bundle.get("annotations_bundle_sha256") != binding["annotations_bundle_sha256"]
            or bundle["release_id"] != release_id
            or bundle.get("submission_manifest_sha256")
            != (sha256(submissions) if submissions else None)
            or bundle.get("latency_profile_file_sha256")
            != (sha256(Path(latency_root) / "profile.json") if latency_root else None)
        ):
            raise ValueError(
                "Space bundle does not match this dataset or release; build a new version"
            )
    if args.stage in {"publish", "all"}:
        try:
            info = api.repo_info(repo_id, repo_type="space")
        except RepositoryNotFoundError:
            expected_head = None
        else:
            if not info.private:
                raise ValueError("Destination Space is public; private publication is required")
            if getattr(info, "sdk", None) != "static":
                raise ValueError(
                    "Existing destination is not a static Space; choose a new Space name"
                )
            expected_head = info.sha
            print("Updating private Space from commit:", expected_head)
        result = publish_bundle(output, repo_id, execute=True, expected_head=expected_head, api=api)
        result["dataset_verification"] = binding
        result["remote_verification"] = {"status": "pending"}
        write_json(receipt, result)
    if args.stage == "verify":
        result = json.loads(receipt.read_text())
        if result["repo_id"] != repo_id or result["bundle_sha256"] != bundle["bundle_sha256"]:
            raise ValueError(
                "Space publication receipt belongs to a different repository or bundle"
            )
    if args.stage in {"publish", "verify", "all"}:
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
        print("Private Space:", f"https://huggingface.co/spaces/{repo_id}")
        print("Publication receipt:", receipt)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None
