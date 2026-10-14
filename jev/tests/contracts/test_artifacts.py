"""Real-filesystem integrity, no-clobber concurrency, and failure-injection tests."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

import jev_contracts.artifacts as artifacts
from jev_contracts import (
    ArtifactError,
    ValidationError,
    artifact_path,
    atomic_write_json,
    canonical_json,
    safe_relative_path,
    sha256_file,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def root(tmp_path):
    # macOS may expose temporary storage through /var -> /private/var.
    return tmp_path.resolve()


def test_canonical_json_is_stable_and_preserves_unicode():
    value = {"z": [True, None, 1.25], "a": "لَيْسَ ١٢٣ café"}
    result = canonical_json(value)
    assert result == '{"a":"لَيْسَ ١٢٣ café","z":[true,null,1.25]}'
    assert canonical_json(dict(reversed(list(value.items())))) == result
    assert json.loads(result) == value
    assert not result.endswith("\n")


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -float("inf"), {1: "x"}, (1,), {1}, object(), b"x", "\ud800"],
)
def test_non_json_canonical_values_fail(value):
    with pytest.raises(ValidationError):
        canonical_json(value)


def test_canonical_cycles_and_depth_are_bounded():
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValidationError, match="cyclic or excessively nested"):
        canonical_json(cyclic)
    nested = "end"
    for _ in range(66):
        nested = [nested]
    with pytest.raises(ValidationError, match="nested"):
        canonical_json(nested)


@pytest.mark.parametrize(
    "relative",
    [
        "", " ", "/absolute", "../escape", "a/../b", ".", "a/./b", "a//b", "a/", "C:/escape",
        "C:\\escape", "\\\\server\\file", "a\\b", "https://host/a", "a\x00b", "a\nb", None, 1,
    ],
)
def test_unsafe_relative_paths(relative):
    with pytest.raises(ValidationError):
        safe_relative_path(relative)


def test_safe_path_is_not_rewritten(root):
    (root / "nested").mkdir()
    relative = "nested/بيانات.json"
    assert safe_relative_path(relative) == relative
    assert artifact_path(root, relative) == root / relative
    with pytest.raises(FileNotFoundError):
        artifact_path(root, "missing/child")
    with pytest.raises(ArtifactError, match="traversal"):
        artifact_path(root / "..", "other")


def test_atomic_json_is_canonical_private_and_immutable(root):
    target = root / "record.json"
    value = {"ب": [1, None], "a": "لَيْسَ"}
    assert atomic_write_json(target, value) is None
    original = target.read_bytes()
    assert original == (canonical_json(value) + "\n").encode("utf-8")
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    for attempted in (value, {"different": True}):
        with pytest.raises(FileExistsError):
            atomic_write_json(target, attempted)
        assert target.read_bytes() == original
    assert sorted(path.name for path in root.iterdir()) == ["record.json"]


def test_invalid_json_does_not_publish(root):
    with pytest.raises(ValidationError):
        atomic_write_json(root / "record.json", {"value": float("nan")})
    assert list(root.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        atomic_write_json(root / "missing" / "record.json", {})
    assert list(root.iterdir()) == []


def test_sha256_is_streamed_and_requires_regular_files(root):
    path = root / "large.bin"
    content = b"abc\x00" * (1024 * 1024)
    path.write_bytes(content)
    assert sha256_file(path) == hashlib.sha256(content).hexdigest()
    (root / "empty").touch()
    assert sha256_file(root / "empty") == hashlib.sha256(b"").hexdigest()
    with pytest.raises(ArtifactError, match="regular file"):
        sha256_file(root)
    fifo = root / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ArtifactError, match="regular file"):
        sha256_file(fifo)


def test_manifest_roundtrip_sorts_files_and_copies_metadata(root):
    (root / "shards").mkdir()
    (root / "shards" / "ar.jsonl").write_bytes("نفي\n".encode())
    (root / "empty").touch()
    metadata = {"approval": {"status": "pending"}, "version": "v1"}
    manifest = write_manifest(root, iter(["shards/ar.jsonl", "empty"]), metadata)
    assert manifest["schema_version"] == 1
    assert list(manifest["files"]) == ["empty", "shards/ar.jsonl"]
    assert manifest["files"]["shards/ar.jsonl"] == {
        "sha256": sha256_file(root / "shards/ar.jsonl"), "bytes": len("نفي\n".encode())
    }
    assert verify_manifest(root) == manifest
    metadata["approval"]["status"] = "modified"
    assert manifest["metadata"]["approval"]["status"] == "pending"
    original = (root / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(root, ["empty"], {})
    assert (root / "manifest.json").read_bytes() == original
    (root / ".orphan.tmp").write_text("ignored incomplete output")
    assert verify_manifest(root) == manifest


def test_empty_manifests_are_generic_not_release_approval(root):
    manifest = write_manifest(root, [], {})
    assert manifest == {"schema_version": 1, "files": {}, "metadata": {}}
    assert verify_manifest(root) == manifest


@pytest.mark.parametrize(
    "files",
    ["file", {"file": {}}, ["../escape"], ["/absolute"], ["manifest.json"], ["a", "a"], None],
)
def test_manifest_rejects_unsafe_duplicate_or_self_paths(root, files):
    with pytest.raises(ValidationError):
        write_manifest(root, files, {})
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("metadata", [None, [], {"bad": float("inf")}, {1: "key"}])
def test_manifest_rejects_invalid_metadata(root, metadata):
    with pytest.raises(ValidationError):
        write_manifest(root, [], metadata)
    assert list(root.iterdir()) == []


def test_missing_files_prevent_publication_and_verification(root):
    with pytest.raises(FileNotFoundError):
        write_manifest(root, ["absent"], {})
    assert not (root / "manifest.json").exists()
    with pytest.raises(FileNotFoundError):
        verify_manifest(root)
    (root / "data").write_bytes(b"abc")
    write_manifest(root, ["data"], {})
    (root / "data").unlink()
    with pytest.raises(FileNotFoundError):
        verify_manifest(root)


@pytest.mark.parametrize("replacement", [b"xyz", b"longer", b""])
def test_corrupt_committed_files_fail_hash_or_length(root, replacement):
    (root / "data").write_bytes(b"abc")
    write_manifest(root, ["data"], {})
    (root / "data").write_bytes(replacement)
    with pytest.raises(ArtifactError, match="integrity mismatch"):
        verify_manifest(root)


@pytest.mark.parametrize("position", ["root", "parent", "file", "dangling"])
def test_symlinks_rejected_by_all_path_operations(root, position):
    real = root / "real"
    real.mkdir()
    (real / "data").write_text("original")
    linked = root / "linked"
    if position in ("root", "parent"):
        linked.symlink_to(real, target_is_directory=True)
        base = linked if position == "root" else root
        relative = "data" if position == "root" else "linked/data"
    else:
        linked.symlink_to(real / ("data" if position == "file" else "absent"))
        base, relative = root, "linked"
    operations = [
        lambda: artifact_path(base, relative),
        lambda: sha256_file(base / relative),
        lambda: atomic_write_json(base / relative, {}),
        lambda: write_manifest(base, [relative], {}),
    ]
    for operation in operations:
        with pytest.raises(ArtifactError, match="symlink"):
            operation()
    assert (real / "data").read_text() == "original"


def test_manifest_and_listed_file_symlinks_rejected_during_verification(root):
    data = root / "data"
    data.write_bytes(b"abc")
    write_manifest(root, ["data"], {})
    saved = root / "saved"
    data.rename(saved)
    data.symlink_to(saved)
    with pytest.raises(ArtifactError, match="symlink"):
        verify_manifest(root)
    data.unlink()
    saved.rename(data)
    (root / "manifest.json").rename(root / "real_manifest")
    (root / "manifest.json").symlink_to(root / "real_manifest")
    with pytest.raises(ArtifactError, match="symlink"):
        verify_manifest(root)


@pytest.mark.parametrize("raw", [
    '{"schema_version":1,"schema_version":1,"files":{},"metadata":{}}',
    '{"schema_version":true,"files":{},"metadata":{}}',
    '{"schema_version":1.0,"files":{},"metadata":{}}',
    '{"schema_version":2,"files":{},"metadata":{}}',
    '{"schema_version":1,"files":{},"metadata":{},"extra":0}',
    '{"schema_version":1,"files":[],"metadata":{}}',
    '{"schema_version":1,"files":{},"metadata":NaN}',
    '{"schema_version":1,"files":{},"metadata":{"a":1e999}}',
    '{"schema_version":1,"files":{},"metadata":{"a":0,"a":1}}',
    '{"schema_version":1,"files":{},"metadata":null}',
    '[]', '{"unfinished":',
])
def test_malformed_manifests_fail(root, raw):
    (root / "manifest.json").write_text(raw)
    with pytest.raises(ArtifactError):
        verify_manifest(root)


@pytest.mark.parametrize("relative,descriptor", [
    ("../data", {"sha256": "a" * 64, "bytes": 0}),
    ("/data", {"sha256": "a" * 64, "bytes": 0}),
    ("manifest.json", {"sha256": "a" * 64, "bytes": 0}),
    ("data", {"sha256": "A" * 64, "bytes": 0}),
    ("data", {"sha256": "a" * 64, "bytes": True}),
    ("data", {"sha256": "a" * 64, "bytes": 0.0}),
    ("data", {"sha256": "a" * 64, "bytes": -1}),
    ("data", {"sha256": "a" * 64}),
    ("data", {"sha256": "a" * 64, "bytes": 0, "extra": 1}),
])
def test_invalid_manifest_descriptors(root, relative, descriptor):
    value = {"schema_version": 1, "files": {relative: descriptor}, "metadata": {}}
    (root / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ArtifactError):
        verify_manifest(root)


@pytest.mark.parametrize("operation", ["replace", "link"])
def test_prepublication_failures_leave_no_success_marker(root, monkeypatch, operation):
    def fail(*args, **kwargs):
        raise OSError(errno.EIO, "injected publication failure")

    monkeypatch.setattr(os, operation, fail)
    with pytest.raises(OSError, match="injected"):
        atomic_write_json(root / "data.json", {"a": 1})
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("failure_call", [1, 2, 3, 4])
def test_fsync_failures_preserve_publication_boundary(root, monkeypatch, failure_call):
    original = os.fsync
    calls = 0

    def fail_once(fd):
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise OSError(errno.EIO, "injected fsync failure")
        return original(fd)

    monkeypatch.setattr(os, "fsync", fail_once)
    target = root / "data.json"
    with pytest.raises(OSError, match="injected fsync"):
        atomic_write_json(target, {"a": 1})
    assert target.exists() == (failure_call >= 3)
    if target.exists():
        assert json.loads(target.read_text()) == {"a": 1}
        with pytest.raises(FileExistsError):
            atomic_write_json(target, {"a": 2})
    assert not list(root.glob(".jev-publish-*"))


def test_partial_write_failure_cleans_staging(root, monkeypatch):
    original = os.fdopen

    class FailingWriter:
        def __init__(self, fd, mode):
            self.stream = original(fd, mode)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def write(self, data):
            self.stream.write(data[:2])
            raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(os, "fdopen", FailingWriter)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_json(root / "data.json", {"a": 1})
    assert list(root.iterdir()) == []


def test_manifest_fsync_failure_does_not_commit(root, monkeypatch):
    (root / "data").write_bytes(b"abc")

    def fail(fd):
        raise OSError(errno.EIO, "injected artifact fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="artifact fsync"):
        write_manifest(root, ["data"], {})
    assert not (root / "manifest.json").exists()
    assert (root / "data").read_bytes() == b"abc"


def test_concurrent_writers_have_exactly_one_winner(root):
    target = root / "data.json"

    def publish(i):
        try:
            atomic_write_json(target, {"winner": i})
            return i
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(publish, range(32)))
    winners = [value for value in outcomes if value is not None]
    assert len(winners) == 1
    assert json.loads(target.read_text()) == {"winner": winners[0]}
    assert not list(root.glob(".jev-publish-*"))


def test_external_racing_writer_is_never_overwritten(root, monkeypatch):
    original = os.link
    target = root / "data.json"

    def race(*args, **kwargs):
        target.write_text('{"accepted":true}\n')
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"replace": True})
    assert target.read_text() == '{"accepted":true}\n'
    assert not list(root.glob(".jev-publish-*"))


def test_observed_file_mutation_during_hashing_fails(root, monkeypatch):
    target = root / "data"
    target.write_bytes(b"abc")
    original = os.read
    changed = False

    def mutate(fd, count):
        nonlocal changed
        chunk = original(fd, count)
        if not changed:
            changed = True
            target.write_bytes(b"xyz")
        return chunk

    monkeypatch.setattr(os, "read", mutate)
    with pytest.raises(ArtifactError, match="changed during"):
        sha256_file(target)


def test_nested_directory_entries_are_fsynced_before_manifest_publication(root, monkeypatch):
    nested = root / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "data").write_bytes(b"abc")
    synced = set()
    original = os.fsync
    publish = artifacts._atomic_write_at

    def record(fd):
        synced.add(os.fstat(fd).st_ino)
        return original(fd)

    def inspect(parent, name, value):
        assert {path.stat().st_ino for path in (root, root / "a", nested)} <= synced
        assert (nested / "data").stat().st_ino in synced
        return publish(parent, name, value)

    monkeypatch.setattr(os, "fsync", record)
    monkeypatch.setattr(artifacts, "_atomic_write_at", inspect)
    write_manifest(root, ["a/b/data"], {})
    verify_manifest(root)


def test_manifest_mutation_during_read_fails(root, monkeypatch):
    write_manifest(root, [], {})
    original = artifacts.json.load

    def mutate(*args, **kwargs):
        value = original(*args, **kwargs)
        (root / "manifest.json").write_text('{"changed":true}')
        return value

    monkeypatch.setattr(artifacts.json, "load", mutate)
    with pytest.raises(ArtifactError, match="changed during verification"):
        verify_manifest(root)


def test_verification_rejects_symlinked_parent_even_with_matching_bytes(root):
    (root / "nested").mkdir()
    (root / "nested" / "data").write_bytes(b"abc")
    write_manifest(root, ["nested/data"], {})
    (root / "nested").rename(root / "renamed")
    (root / "nested").symlink_to(root / "renamed", target_is_directory=True)
    with pytest.raises(ArtifactError, match="symlink"):
        verify_manifest(root)


def test_racing_symlink_cannot_redirect_publication(root, monkeypatch):
    original = os.link
    target = root / "data.json"
    protected = root / "protected.json"
    protected.write_text('{"original":true}')

    def race(*args, **kwargs):
        target.symlink_to(protected)
        return original(*args, **kwargs)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        atomic_write_json(target, {"replace": True})
    assert protected.read_text() == '{"original":true}'
    assert not list(root.glob(".jev-publish-*"))
