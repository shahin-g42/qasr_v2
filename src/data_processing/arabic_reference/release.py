"""Private annotation datasets and immutable, explicitly enumerated release bundles."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from .core import DATASETS, digest, load_plan, sha256, write_json
from .dataset_card import dataset_card
from .prompts import proposal_checks
from .review import validated_reviews
from .runtime import load_record
from .speculative import load_speculative_drafts


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def valid_release_id(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value):
        raise ValueError("Release ID must be a short filename-safe version")
    return value


def finish_bundle(stage, output, metadata):
    """Never overwrite a different release, or upload files not named by this manifest."""
    stage, output = Path(stage), Path(output)
    files = {
        p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()
    }
    manifest = {"schema_version": 1, "visibility": "private", **metadata, "files": files}
    manifest["bundle_sha256"] = digest(manifest)
    write_json(stage / "release.json", manifest)
    if output.exists():
        previous = validate_bundle(output)
        if previous != manifest:
            raise ValueError(
                "Release already exists with different contents; choose a new output directory/version"
            )
    else:
        stage.rename(output)
    return {"output": str(output.resolve()), **manifest}


def validate_bundle(root):
    root = Path(root)
    manifest = json.loads((root / "release.json").read_text())
    if manifest.get("bundle_sha256") != digest(
        {k: v for k, v in manifest.items() if k != "bundle_sha256"}
    ):
        raise ValueError("Release manifest digest mismatch")
    if manifest.get("visibility") != "private" or manifest.get("repo_type") not in {
        "dataset",
        "space",
    }:
        raise ValueError("Only private dataset/Space bundles are supported")
    paths = manifest.get("files")
    if not isinstance(paths, dict) or not paths:
        raise ValueError("Release has no uploadable files")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Release bundles cannot contain symlinks")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(paths) | {"release.json"}:
        raise ValueError(
            "Unexpected or missing release files; only the frozen file list may be uploaded"
        )
    for relative, expected in paths.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError("Unsafe release path")
        if sha256(root / relative) != expected:
            raise ValueError(f"Release file changed: {relative}")
    return manifest


def load_release_proposals(root, plan, rows):
    if root is None:
        return {}, None
    root = Path(root)
    identity = json.loads((root / "run.json").read_text())
    if identity["manifest_digest"] != plan["manifest_digest"]:
        raise ValueError("Proposals belong to a different plan")
    records = {}
    run_hash = digest(identity)
    for row in rows:
        record = load_record(root, row, run_hash, inherited=identity.get("inherited_records", {}))
        if record is not None:
            if record.get("verified_text") is not None or record.get("ideal_text") is not None:
                raise ValueError("A corrector record cannot supply final reference columns")
            if record["status"] not in {"proposed", "unresolved", "failed"}:
                raise ValueError("Unknown proposal status")
            records[row["id"]] = record
    return records, identity


def public_corrector(identity):
    server = identity.get("server", {})
    reported = server.get("server_model", {})
    result = {
        "served_model": server.get("served_model"),
        "reported_checkpoint": reported.get("root"),
        "revision": identity.get("checkpoint_revision"),
        "run_sha256": digest(identity),
        "prompt_version": identity.get("prompt_version"),
        "prompt_sha256": identity.get("prompt_sha256"),
        "generation": identity.get("generation"),
    }
    if result["reported_checkpoint"] and str(result["reported_checkpoint"]).startswith("/"):
        result["reported_checkpoint"] = "local checkpoint; see private run provenance"
    if identity.get("processing_mode"):
        result["processing_mode"] = identity["processing_mode"]
        result["llm_requests"] = identity["llm_requests"]
    if identity.get("recovery_version"):
        result.update(
            {
                k: identity[k]
                for k in ("recovery_version", "recovery_prompt_sha256", "recovery_temperatures")
            }
        )
        result["source_run_sha256"] = identity["repair_source"]["run_sha256"]
    return result


def public_source(row):
    """Only stable source locators; never copy the prepared cache's absolute path."""
    source = row.get("source") or {}
    permitted = (
        "repository",
        "revision",
        "remote_path",
        "kind",
        "remote_id",
        "row_group",
        "row_index",
        "offset_seconds",
        "duration_seconds",
    )
    result = {k: source[k] for k in permitted if k in source}
    for name in ("remote_path",):
        value = result.get(name)
        if value and (Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("Source locator must be repository-relative")
    return result


def export_dataset(
    plan_root,
    output,
    *,
    proposals=None,
    speculative_drafts=None,
    reviewed=None,
    datasets=None,
    release_id="v4-draft-001",
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    valid_release_id(release_id)
    plan, rows = load_plan(plan_root)
    selected = list(DATASETS if datasets is None else datasets)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(DATASETS):
        raise ValueError("Choose distinct known dataset names")
    selected = [name for name in DATASETS if name in selected]
    records, identity = load_release_proposals(proposals, plan, rows)
    speculative, speculation = load_speculative_drafts(speculative_drafts, rows, records, identity)
    reviewed_rows = {r["id"]: r for r in validated_reviews(rows, reviewed)[0]} if reviewed else {}
    output_rows = []
    for row in rows:
        if row["dataset"] not in selected:
            continue
        record = records.get(row["id"], {})
        proposal = (
            record.get("proposal") if record.get("status") in {"proposed", "unresolved"} else None
        )
        draft = speculative.get(row["id"], {})
        if draft:
            proposal = draft["proposal"]
        final = reviewed_rows.get(row["id"], row)
        candidates = {c["name"]: c["text"] for c in row["candidates"]}
        checks = (
            proposal_checks(proposal, row, (draft or record).get("validation_adjustments", []))
            if proposal
            else None
        )
        output_rows.append(
            {
                "id": row["id"],
                "dataset": row["dataset"],
                "split": "test",
                "text": row["text"],
                "asr_text": candidates.get("qasr"),
                "gqasr_text": candidates.get("gqasr"),
                "candidate_texts_json": json_text(candidates),
                "verified_text_draft": proposal.get("proposed_verified_text") if proposal else None,
                "ideal_text_draft": proposal.get("proposed_ideal_text") if proposal else None,
                "verified_text": final["verified_text"],
                "ideal_text": final["ideal_text"],
                "proposal_status": draft.get("proposal_status", record.get("status", "pending")),
                "speculative": bool(draft),
                "speculation_confidence": draft.get("confidence"),
                "speculative_spans_json": json_text(draft["speculative_spans"]) if draft else None,
                "speculative_reason": draft.get("judgment"),
                "speculative_draft_sha256": draft.get("speculative_draft_sha256"),
                "correction_method": (draft or record).get(
                    "correction_method", "llm_contextual" if proposal else None
                ),
                "review_status": final["review_status"],
                "audio_reviewed": final["review_status"] == "verified",
                "duration_seconds": row["actual_duration"],
                "audio_sha256": row["audio_sha256"],
                "source_locator_json": json_text(public_source(row)),
                "dialect": row["metadata"].get("dialect"),
                "cohorts": row["cohorts"],
                "flags": row["flags"],
                "changes_json": json_text(proposal.get("changes", [])) if proposal else None,
                "checks_json": json_text(checks) if checks is not None else None,
                "uncertainties_json": json_text(proposal.get("uncertainties", []))
                if proposal
                else None,
                "issue_notes_json": json_text(proposal.get("issue_notes", {}))
                if proposal
                else None,
                "binding": row["binding"],
                "policy_sha256": row["policy_sha256"],
                "proposal_run_sha256": record.get("run_sha256"),
                "proposal_record_sha256": record.get("record_sha256"),
                "reviewed_record_sha256": final.get("reviewed_sha256"),
                "release_id": release_id,
                "purpose": "evaluation_only",
            }
        )
    if not output_rows:
        raise ValueError("No rows selected for release")
    string_fields = set(output_rows[0]) - {
        "duration_seconds",
        "audio_reviewed",
        "speculative",
        "cohorts",
        "flags",
    }
    schema = pa.schema(
        [
            (
                name,
                pa.string()
                if name in string_fields
                else pa.float64()
                if name == "duration_seconds"
                else pa.bool_()
                if name in {"audio_reviewed", "speculative"}
                else pa.list_(pa.string()),
            )
            for name in output_rows[0]
        ]
    )
    counts = {
        ds: {
            "rows": sum(r["dataset"] == ds for r in output_rows),
            "proposals": dict(
                Counter(r["proposal_status"] for r in output_rows if r["dataset"] == ds)
            ),
            "reviews": dict(Counter(r["review_status"] for r in output_rows if r["dataset"] == ds)),
        }
        for ds in selected
    }
    model = public_corrector(identity) if identity else None
    origin_counts = Counter(
        r["proposal_run_sha256"] for r in output_rows if r["proposal_run_sha256"]
    )
    corrector_runs = []
    origin = identity
    while origin:
        public = public_corrector(origin)
        public["records_in_release"] = origin_counts[public["run_sha256"]]
        corrector_runs.append(public)
        origin = origin.get("repair_source", {}).get("identity")
    provenance = {
        "release_id": release_id,
        "manifest_digest": plan["manifest_digest"],
        "policy_sha256": plan["policy_sha256"],
        "selection_scope": plan.get("selection_scope", "pilot_plus_diagnostics"),
        "input_counts": plan["input_counts"],
        "selected_datasets": selected,
        "counts": counts,
        "audio_included": False,
        "speculative_drafts": {
            **speculation,
            "records_in_release": sum(r["speculative"] for r in output_rows),
        }
        if speculation
        else None,
        "prediction_sources": [
            {k: s[k] for k in ("name", "predictions_sha256")} for s in plan["sources"]
        ],
        "corrector": model,
        "corrector_runs": corrector_runs,
        "reviewed_manifest_sha256": sha256(reviewed) if reviewed else None,
        "limitations": [
            "Candidate-assisted text-only drafts are not audio verification.",
            "This evaluation set must not be used for training.",
            "No audio bytes are redistributed; use the pinned source locator and existing preparation pipeline.",
            "Single-review attestations are not an independent benchmark audit.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reference-release-", dir=output.parent) as temporary:
        stage = Path(temporary) / "bundle"
        stage.mkdir()
        for ds in selected:
            path = stage / "data" / ds / "test.parquet"
            path.parent.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist([r for r in output_rows if r["dataset"] == ds], schema=schema),
                path,
                compression="zstd",
            )
        write_json(stage / "provenance.json", provenance)
        write_json(stage / "policy.json", plan["policy"])
        (stage / "README.md").write_text(dataset_card(selected, provenance))
        return finish_bundle(
            stage,
            output,
            {
                "repo_type": "dataset",
                "release_id": release_id,
                "manifest_digest": plan["manifest_digest"],
                "counts": counts,
                "datasets": selected,
            },
        )


def publish_bundle(root, repo_id, *, execute=False, expected_head=None, api=None):
    """Private-only publication. Dry-run is offline; an explicit execution performs upload."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise ValueError("Use an explicit namespace/repository Hugging Face ID")
    root = Path(root)
    manifest = validate_bundle(root)
    if manifest.get("audio_included"):
        quality = json.loads((root / "quality_report.json").read_text())
        if (
            manifest.get("quality_status") != "passed"
            or quality.get("status") != "passed"
            or quality.get("release_id") != manifest["release_id"]
            or quality.get("decoded_audio_rows")
            != sum(c["rows"] for c in manifest["counts"].values())
            or not quality.get("all_annotation_columns_preserved")
        ):
            raise ValueError("Audio release has not passed the complete written-shard audit")
        configured_repo = manifest.get("package_config", {}).get("dataset_repo")
        if configured_repo and configured_repo != repo_id:
            raise ValueError("Publish destination differs from the configured dataset card")
    files = sorted([*manifest["files"], "release.json"])
    result = {
        "repo_id": repo_id,
        "repo_type": manifest["repo_type"],
        "private": True,
        "bundle_sha256": manifest["bundle_sha256"],
        "files": files,
        "executed": False,
    }
    if not execute:
        return result
    api = api or HfApi()
    try:
        info = api.repo_info(repo_id, repo_type=manifest["repo_type"])
    except RepositoryNotFoundError:
        if expected_head:
            raise ValueError("Expected an existing repository, but it was not accessible") from None
        api.create_repo(
            repo_id,
            repo_type=manifest["repo_type"],
            private=True,
            exist_ok=False,
            **({"space_sdk": "static"} if manifest["repo_type"] == "space" else {}),
        )
        info = api.repo_info(repo_id, repo_type=manifest["repo_type"])
    else:
        if not info.private:
            raise ValueError(
                "Destination is public; this pipeline only uploads to a private repository"
            )
        if not expected_head or info.sha != expected_head:
            raise ValueError(
                f"Existing repository requires --expected-head {info.sha}; review the target before updating"
            )
    if not info.private:
        raise ValueError("Private destination could not be verified")
    # Snapshot verified bytes before upload. Large audio shards stay on disk, not in RAM.
    import hashlib

    with tempfile.TemporaryDirectory(prefix=".reference-upload-", dir=root.parent) as temporary:
        payloads = {}
        for name in files:
            path = root / name
            if path.stat().st_size <= 8 * 1024**2:
                data = path.read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                payloads[name] = data
            else:
                snapshot = Path(temporary) / name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                with path.open("rb") as incoming, snapshot.open("wb") as outgoing:
                    for chunk in iter(lambda: incoming.read(1024**2), b""):
                        hasher.update(chunk)
                        outgoing.write(chunk)
                checksum = hasher.hexdigest()
                payloads[name] = str(snapshot)
            if name != "release.json" and checksum != manifest["files"][name]:
                raise ValueError("Release changed before upload")
        if json.loads(payloads["release.json"]) != manifest:
            raise ValueError("Release manifest changed before upload")
        commit = api.create_commit(
            repo_id,
            repo_type=manifest["repo_type"],
            parent_commit=info.sha,
            operations=[
                CommitOperationAdd(path_in_repo=name, path_or_fileobj=payloads[name])
                for name in files
            ],
            commit_message=f"Publish {manifest['release_id']} ({manifest['bundle_sha256'][:12]})",
        )
    return {**result, "executed": True, "commit_sha": commit.oid, "commit_url": commit.commit_url}
