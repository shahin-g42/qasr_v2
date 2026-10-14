"""Generic immutable artifact publication and verification, using only stdlib.

The filesystem contract is POSIX: no-follow directory descriptors, fsync, atomic
rename and hard links must work on the configured storage. Unsupported operations
fail closed. This is integrity/durability plumbing, not approval or a signature:
a caller must separately trust the manifest and enforce its release policies.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._validation import ValidationError, json_copy, object_fields, text, version

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "manifest.json"
Pathish = str | os.PathLike[str]


class ArtifactError(ValidationError):
    """Unsafe artifact path, invalid manifest, or failed integrity verification.

    Ordinary IO errors retain their OSError subtype (e.g. FileNotFoundError,
    FileExistsError, PermissionError). An fsync failure after publication raises
    while leaving a complete visible artifact; callers must verify it, never
    silently overwrite it or assume a raised error means nothing was published.
    """


def canonical_json(value: Any) -> str:
    """Return stable compact UTF-8-compatible JSON, sorted keys, without a newline.

    Unicode is preserved (ensure_ascii=False); NaN/Inf, non-string keys, tuples,
    objects, cycles, and excessively deep structures fail with ValidationError.
    This is Python JSON canonicalization, not a claim of RFC 8785 compliance.
    """
    validated = json_copy(value)
    try:
        return json.dumps(
            validated, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"JSON serialization failed: {exc}") from exc


def safe_relative_path(value: str) -> str:
    """Validate and return an unchanged portable relative POSIX artifact path.

    Reject absolute/drive/UNC paths, dot components, repeated separators,
    backslashes, colons, and control characters. Never sanitize by dropping
    components: ambiguous or escaping paths must fail rather than alias a file.
    Opaque example IDs must not be treated as paths without this validation.
    """
    try:
        text(value, "artifact path")
    except ValidationError as exc:
        raise ArtifactError(str(exc)) from exc
    if "\\" in value or ":" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ArtifactError(
            f"artifact path {value!r}: backslashes, colons or controls are forbidden"
        )
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ArtifactError(
            f"artifact path {value!r}: expected relative path without dot components"
        )
    return value


def _absolute(path: Pathish) -> Path:
    result = Path(path).absolute()
    if ".." in result.parts:
        raise ArtifactError(f"{path}: parent traversal is forbidden")
    return result


def _directory_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ArtifactError("artifact storage requires POSIX O_NOFOLLOW and O_DIRECTORY")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


def _open_directory_at(parent: int, name: str) -> int:
    try:
        return os.open(name, _directory_flags(), dir_fd=parent)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise ArtifactError(f"{name}: symlink or non-directory in artifact path") from exc
        raise


@contextmanager
def _directory(path: Pathish) -> Iterator[int]:
    absolute = _absolute(path)
    fd = os.open(absolute.anchor, _directory_flags())
    try:
        for part in absolute.parts[1:]:
            child = _open_directory_at(fd, part)
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _relative_parent(
    root_fd: int, relative: str, *, durable: bool = False
) -> Iterator[tuple[int, str]]:
    parts = safe_relative_path(relative).split("/")
    fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            if durable:
                os.fsync(fd)
            child = _open_directory_at(fd, part)
            os.close(fd)
            fd = child
        yield fd, parts[-1]
    finally:
        os.close(fd)


def _reject_symlink(parent: int, name: str) -> os.stat_result | None:
    try:
        info = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if stat.S_ISLNK(info.st_mode):
        raise ArtifactError(f"{name}: symlinks are forbidden")
    return info


def artifact_path(root: Pathish, relative: str) -> Path:
    """Return an absolute path after rejecting traversal and symlink components.

    Root and intermediate directories must already exist. The final file may be
    absent. This convenience check cannot secure subsequent caller IO against a
    race; the hash/manifest/writer functions instead keep no-follow descriptors.
    """
    relative = safe_relative_path(relative)
    absolute = _absolute(root)
    with _directory(absolute) as root_fd, _relative_parent(root_fd, relative) as (parent, name):
        _reject_symlink(parent, name)
    return absolute / relative


def _open_regular(parent: int, name: str) -> int:
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ArtifactError(f"{name}: symlinks are forbidden") from exc
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ArtifactError(f"{name}: expected a regular file")
    return fd


def _identity(info: os.stat_result) -> tuple:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _file_info(parent: int, name: str, *, durable: bool = False) -> dict:
    fd = _open_regular(parent, name)
    try:
        before = os.fstat(fd)
        digest = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        if durable:
            os.fsync(fd)
        after = os.fstat(fd)
        current = _reject_symlink(parent, name)
        if (
            current is None
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(current)
            or size != after.st_size
        ):
            raise ArtifactError(f"{name}: file changed during checksum computation")
        return {"sha256": digest.hexdigest(), "bytes": size}
    finally:
        os.close(fd)


def sha256_file(path: Pathish) -> str:
    """Stream SHA-256 of a regular file, rejecting symlinks and observed mutation."""
    absolute = _absolute(path)
    with _directory(absolute.parent) as parent:
        return _file_info(parent, absolute.name)["sha256"]


def _atomic_write_at(parent: int, name: str, value: Any) -> None:
    data = (canonical_json(value) + "\n").encode("utf-8")
    if _reject_symlink(parent, name) is not None:
        raise FileExistsError(errno.EEXIST, "immutable artifact already exists", name)
    staging = f".jev-publish-{uuid.uuid4().hex}"
    os.mkdir(staging, mode=0o700, dir_fd=parent)
    stage_fd = None
    try:
        stage_fd = _open_directory_at(parent, staging)
        fd = os.open(
            "payload.tmp", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600, dir_fd=stage_fd,
        )
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace("payload.tmp", "payload.ready", src_dir_fd=stage_fd, dst_dir_fd=stage_fd)
        os.fsync(stage_fd)
        # A public os.replace would clobber a racing writer. A hard link is an
        # atomic create-if-absent publication of the fully fsynced, renamed inode.
        os.link(
            "payload.ready", name, src_dir_fd=stage_fd, dst_dir_fd=parent, follow_symlinks=False
        )
        os.fsync(parent)
    finally:
        if stage_fd is not None:
            try:
                for temporary in ("payload.tmp", "payload.ready"):
                    try:
                        os.unlink(temporary, dir_fd=stage_fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(stage_fd)
        os.rmdir(staging, dir_fd=parent)
        os.fsync(parent)


def atomic_write_json(path: Pathish, value: Any) -> None:
    """Publish canonical JSON plus newline, exclusively; never replace an artifact.

    Parents must exist and no component may be a symlink. Publication uses a
    private exclusive temporary file, flush/fsync, os.replace inside staging,
    then an atomic no-clobber hard link and directory fsync. Concurrent writers
    cannot overwrite each other. Even an identical existing file raises
    FileExistsError: an idempotent caller must verify and explicitly reuse it.
    Cleanup never removes the final artifact. A failure after linking may leave
    a valid visible file with uncertain durability; see ArtifactError's docstring.
    """
    absolute = _absolute(path)
    name = safe_relative_path(absolute.name)
    with _directory(absolute.parent) as parent:
        _atomic_write_at(parent, name, value)


def _manifest_paths(files: Iterable[str]) -> list[str]:
    if isinstance(files, (str, bytes, dict)):
        raise ArtifactError(
            "files: expected an iterable of relative file paths, not a string/object"
        )
    try:
        iterator = iter(files)
    except TypeError as exc:
        raise ArtifactError("files: expected an iterable of relative file paths") from exc
    paths: set[str] = set()
    for value in iterator:
        relative = safe_relative_path(value)
        if relative == MANIFEST_FILENAME:
            raise ArtifactError("files: manifest.json cannot checksum itself")
        if relative in paths:
            raise ArtifactError(f"files: duplicate path {relative!r}")
        paths.add(relative)
    return sorted(paths)


def write_manifest(root: Pathish, files: Iterable[str], metadata: dict) -> dict:
    """Fsync/hash existing files, then exclusively publish root/manifest.json.

    Shape: {schema_version: 1, files: {relative: {sha256: str, bytes: int}},
    metadata: dict}. Metadata is arbitrary strict finite JSON, not an approval
    schema. Empty file sets are permitted for generic metadata-only artifacts;
    consumers must enforce required members. Parents/files must already exist.
    Callers own staging and must stop mutating files before publication. New
    releases need new roots; existing manifests are never replaced, even if equal.
    """
    validated_metadata = json_copy(metadata, "metadata")
    if type(validated_metadata) is not dict:
        raise ArtifactError("metadata: expected an object")
    paths = _manifest_paths(files)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION, "files": {}, "metadata": validated_metadata
    }
    with _directory(root) as root_fd:
        if _reject_symlink(root_fd, MANIFEST_FILENAME) is not None:
            raise FileExistsError(
                errno.EEXIST, "immutable manifest already exists", MANIFEST_FILENAME
            )
        for relative in paths:
            with _relative_parent(root_fd, relative, durable=True) as (parent, name):
                manifest["files"][relative] = _file_info(parent, name, durable=True)
                os.fsync(parent)
        _atomic_write_at(root_fd, MANIFEST_FILENAME, manifest)
    return manifest


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactError(f"manifest: duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ArtifactError(f"manifest: non-finite JSON constant {value}")


def _read_manifest(root_fd: int) -> dict:
    fd = _open_regular(root_fd, MANIFEST_FILENAME)
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        before = os.fstat(fd)
        try:
            value = json.load(
                stream, object_pairs_hook=_unique_object, parse_constant=_invalid_constant
            )
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise ArtifactError(f"manifest: invalid JSON: {exc}") from exc
        after = os.fstat(fd)
        current = _reject_symlink(root_fd, MANIFEST_FILENAME)
        if (
            current is None
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(current)
        ):
            raise ArtifactError("manifest.json: file changed during verification")
    try:
        value = json_copy(value, "manifest")
        object_fields(value, "manifest", {"schema_version", "files", "metadata"})
        version(value["schema_version"], MANIFEST_SCHEMA_VERSION, "manifest.schema_version")
        if type(value["files"]) is not dict or type(value["metadata"]) is not dict:
            raise ArtifactError("manifest.files and manifest.metadata must be objects")
        for relative, info in value["files"].items():
            safe_relative_path(relative)
            if relative == MANIFEST_FILENAME:
                raise ArtifactError("manifest.json cannot checksum itself")
            object_fields(info, f"manifest.files.{relative}", {"sha256", "bytes"})
            digest = text(info["sha256"], f"manifest.files.{relative}.sha256")
            if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                raise ArtifactError(
                    f"manifest.files.{relative}.sha256: expected lowercase SHA-256 hex"
                )
            if type(info["bytes"]) is not int or info["bytes"] < 0:
                raise ArtifactError(
                    f"manifest.files.{relative}.bytes: expected a nonnegative integer"
                )
    except ValidationError as exc:
        raise ArtifactError(str(exc)) from exc
    return value


def verify_manifest(root: Pathish) -> dict:
    """Validate manifest v1 and every listed file's bytes/hash; return the manifest.

    Missing files fail. Root, manifest, listed files, and their ancestors cannot
    be symlinks. Extra unlisted files (including orphan staging output) are ignored.
    This verifies only integrity, not signatures, approval, complete datasets, or
    future immutability. Consumers must not modify the verified tree afterward.
    """
    with _directory(root) as root_fd:
        manifest = _read_manifest(root_fd)
        for relative, expected in manifest["files"].items():
            with _relative_parent(root_fd, relative) as (parent, name):
                actual = _file_info(parent, name)
            if actual != expected:
                raise ArtifactError(f"{relative}: integrity mismatch (SHA-256 or byte length)")
    return manifest
