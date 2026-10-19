"""Private annotation datasets and immutable, explicitly enumerated release bundles."""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

from .core import DATASETS, digest, load_plan, sha256, write_json
from .prompts import proposal_checks
from .review import validated_reviews
from .runtime import load_record


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
    for row in rows:
        record = load_record(root, row, digest(identity))
        if record is not None:
            if record.get("verified_text") is not None or record.get("ideal_text") is not None:
                raise ValueError("A corrector record cannot supply final reference columns")
            if record["status"] not in {"proposed", "unresolved", "failed"}:
                raise ValueError("Unknown proposal status")
            records[row["id"]] = record
    return records, identity


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
    plan_root, output, *, proposals=None, reviewed=None, datasets=None, release_id="v4-draft-001"
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
    reviewed_rows = {r["id"]: r for r in validated_reviews(rows, reviewed)[0]} if reviewed else {}
    output_rows = []
    for row in rows:
        if row["dataset"] not in selected:
            continue
        record = records.get(row["id"], {})
        proposal = (
            record.get("proposal") if record.get("status") in {"proposed", "unresolved"} else None
        )
        final = reviewed_rows.get(row["id"], row)
        candidates = {c["name"]: c["text"] for c in row["candidates"]}
        checks = (
            proposal_checks(proposal, row, record.get("validation_adjustments", []))
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
                "proposal_status": record.get("status", "pending"),
                "correction_method": record.get(
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
    string_fields = set(output_rows[0]) - {"duration_seconds", "audio_reviewed", "cohorts", "flags"}
    schema = pa.schema(
        [
            (
                name,
                pa.string()
                if name in string_fields
                else pa.float64()
                if name == "duration_seconds"
                else pa.bool_()
                if name == "audio_reviewed"
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
    model = None
    if identity:
        server = identity.get("server", {})
        reported = server.get("server_model", {})
        model = {
            "served_model": server.get("served_model"),
            "reported_checkpoint": reported.get("root"),
            "revision": identity.get("checkpoint_revision"),
            "run_sha256": digest(identity),
            "prompt_version": identity.get("prompt_version"),
            "prompt_sha256": identity.get("prompt_sha256"),
            "generation": identity.get("generation"),
        }
        if model["reported_checkpoint"] and str(model["reported_checkpoint"]).startswith("/"):
            model["reported_checkpoint"] = "local checkpoint; see private run provenance"
    provenance = {
        "release_id": release_id,
        "manifest_digest": plan["manifest_digest"],
        "policy_sha256": plan["policy_sha256"],
        "selection_scope": plan.get("selection_scope", "pilot_plus_diagnostics"),
        "input_counts": plan["input_counts"],
        "selected_datasets": selected,
        "counts": counts,
        "audio_included": False,
        "prediction_sources": [
            {k: s[k] for k in ("name", "predictions_sha256")} for s in plan["sources"]
        ],
        "corrector": model,
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


def dataset_card(datasets, provenance):
    configs = []
    for ds in datasets:
        configs.append(
            f"  - config_name: {ds}\n    data_files:\n      - split: test\n        path: data/{ds}/test.parquet"
        )
    all_paths = "\n".join(f"          - data/{ds}/test.parquet" for ds in datasets)
    header = "---\nlanguage:\n  - ar\ntask_categories:\n  - automatic-speech-recognition\ntags:\n  - arabic\n  - evaluation-only\n  - annotation-drafts\nconfigs:\n"
    header += (
        f"  - config_name: all\n    default: true\n    data_files:\n      - split: test\n        path:\n{all_paths}\n"
        + "\n".join(configs)
        + "\n---\n"
    )
    counts = provenance["counts"]
    table = "\n".join(
        f"| {ds} | {c['rows']} | {c['proposals'].get('proposed', 0)} | {c['reviews'].get('verified', 0)} |"
        for ds, c in counts.items()
    )
    sources = "\n".join(
        f"- `{s['name']}` prediction SHA256: `{s['predictions_sha256']}`"
        for s in provenance["prediction_sources"]
    )
    return (
        header
        + f"""# Arabic ASR reference annotations — {provenance["release_id"]}

Private research dataset. Scope: **{provenance["selection_scope"]}**. This release contains annotation overlays and source locators, without audio bytes. Original source terms and attribution remain applicable. See each row's pinned `source_locator_json` and `provenance.json`; this package does not grant a new blanket license to the underlying sources.

| Dataset | Rows | LLM drafts | Audio-reviewed |
|---|---:|---:|---:|
{table}

`text` is the immutable original reference. `asr_text` is the saved QASR prediction; `gqasr_text` is the GQASR prediction. `candidate_texts_json` retains every named candidate. `verified_text_draft` and `ideal_text_draft` are unverified suggestions. `correction_method` distinguishes `llm_after_initial_cleanup` from `llm_contextual` editing. Every completed row in the full correction route receives LLM contextual review, including initially agreeing rows. The exact supplied blessing صلى الله عليه وسلم is rendered as ﷺ under the explicit editorial convention; a missing blessing is never inserted. Final `verified_text` and `ideal_text` remain null until an attested listening review is imported. Failed, pending and unresolved rows remain visible. Never substitute a draft or original reference for a missing final value.

The corrector sees anonymous candidate labels in varying order. Its fluency, agreement and edit explanations are not proof of speech. `checks_json` records independently computed text differences; model labels do not establish whether edits preserve meaning. `changes_json`, `uncertainties_json` and `issue_notes_json` retain the unverified editorial evidence. Raw server replies, local cache paths, reviewer identities and host logs are excluded from this distribution.

The `test` split is evaluation-only. Do not feed it into training. Pilot and diagnostic membership is in `cohorts`; diagnostics are outcome-selected and must not be reported as an unbiased full-suite score. Duplicate source rows remain identifiable by `audio_sha256`.

```python
from datasets import load_dataset
# Authenticate with your own permitted Hugging Face account; pin the returned commit SHA.
ds = load_dataset("shahink/asr_bench", "all", split="test", revision="<release commit>")
```

The exact plan is bound by `{provenance["manifest_digest"]}`. The annotation policy is included in `policy.json`. The corrector identity and review coverage are in `provenance.json`. Single-review attestations are not independent audit. Reference development has used QASR/GQASR outputs, so results need that disclosure and a fresh holdout for general model-quality claims.

{sources}

The release is immutable locally. Publication uploads only the files enumerated in `release.json` to a private repository; use the Hub commit SHA as the distributed version.
"""
    )


def publish_bundle(root, repo_id, *, execute=False, expected_head=None, api=None):
    """Private-only publication. Dry-run is offline; an explicit execution performs upload."""
    from huggingface_hub import CommitOperationAdd, HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*", repo_id):
        raise ValueError("Use an explicit namespace/repository Hugging Face ID")
    root = Path(root)
    manifest = validate_bundle(root)
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
    # Read verified bytes now: later file mutations cannot change what this commit uploads.
    payloads = {name: (root / name).read_bytes() for name in files}
    import hashlib

    if any(
        hashlib.sha256(payloads[name]).hexdigest() != checksum
        for name, checksum in manifest["files"].items()
    ):
        raise ValueError("Release changed before upload")
    if json.loads(payloads["release.json"]) != manifest:
        raise ValueError("Release manifest changed before upload")
    commit = api.create_commit(
        repo_id,
        repo_type=manifest["repo_type"],
        parent_commit=info.sha,
        operations=[
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=payloads[name]) for name in files
        ],
        commit_message=f"Publish {manifest['release_id']} ({manifest['bundle_sha256'][:12]})",
    )
    return {**result, "executed": True, "commit_sha": commit.oid, "commit_url": commit.commit_url}
