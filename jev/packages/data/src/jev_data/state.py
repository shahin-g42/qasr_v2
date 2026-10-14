"""One local SQLite writer per shard, immutable publication and bounded recovery."""

from __future__ import annotations

import fcntl
import os
import sqlite3
import time
from pathlib import Path

from jev_contracts import atomic_write_json, verify_manifest, write_manifest

from .io import digest, integer, read_json, read_verified_json, require, text

TRANSITIONS = {
    "planned": {"authored", "quarantined", "excluded"},
    "authored": {"verified", "quarantined", "excluded"},
    "verified": {"adjudicated", "accepted", "quarantined", "excluded"},
    "adjudicated": {"accepted", "quarantined", "excluded"},
    "quarantined": set(), "accepted": set(), "excluded": set(),
}


class ShardState:
    def __init__(self, root: str | Path, owner: str, *, max_attempts: int = 3,
                 local_storage: bool = False):
        require(local_storage is True, "shard SQLite requires explicitly validated local storage")
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        from jev_contracts import artifact_path

        artifact_path(self.root, "owner.lock")
        self.owner = text(owner)
        self.max_attempts = integer(max_attempts, 1, 3)
        self._fd = os.open(self.root / "owner.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._fd)
            raise ValueError("shard already owned by another writer") from None
        self.connection = None
        try:
            artifact_path(self.root, "state.sqlite3")
            self.connection = sqlite3.connect(self.root / "state.sqlite3", timeout=1)
            self.connection.row_factory = sqlite3.Row
            self.connection.executescript("""
                PRAGMA journal_mode=DELETE;
                PRAGMA synchronous=FULL;
                CREATE TABLE IF NOT EXISTS ownership (
                    id INTEGER PRIMARY KEY, owner TEXT, beat REAL);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, state TEXT NOT NULL,
                    attempt INTEGER NOT NULL, reason TEXT, artifact TEXT, artifact_hash TEXT);
                CREATE TABLE IF NOT EXISTS history (
                    sequence INTEGER PRIMARY KEY, job_id TEXT, state TEXT, attempt INTEGER,
                    reason TEXT, owner TEXT, time REAL);
            """)
            self.heartbeat()
            self.reconcile()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def heartbeat(self) -> None:
        require(self.connection is not None, "shard owner is closed")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO ownership VALUES (1, ?, ?)",
                                    (self.owner, time.time()))

    def _history(self, job_id: str, state: str, attempt: int, reason: str | None) -> None:
        self.connection.execute("INSERT INTO history VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                                (job_id, state, attempt, reason, self.owner, time.time()))

    def get(self, job_id: str) -> dict:
        self.heartbeat()
        result = self.connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        require(result is not None, "unknown shard job")
        return dict(result)

    def plan(self, job_id: str, request: dict) -> dict:
        job_id, request_hash = text(job_id), digest(request)
        self.heartbeat()
        with self.connection:
            existing = self.connection.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if existing:
                require(existing["request_hash"] == request_hash, "job request identity changed")
                return dict(existing)
            self.connection.execute(
                "INSERT INTO jobs VALUES (?, ?, 'planned', 1, NULL, NULL, NULL)",
                (job_id, request_hash))
            self._history(job_id, "planned", 1, None)
        return self.get(job_id)

    def transition(self, job_id: str, state: str, *, reason: str | None = None) -> dict:
        job = self.get(job_id)
        require(state != "accepted", "acceptance must publish an immutable artifact first")
        require(state in TRANSITIONS[job["state"]], "invalid or immutable job transition")
        if state in {"quarantined", "excluded"}:
            text(reason)
        with self.connection:
            self.connection.execute("UPDATE jobs SET state=?, reason=? WHERE id=?",
                                    (state, reason, job_id))
            self._history(job_id, state, job["attempt"], reason)
        return self.get(job_id)

    def retry(self, job_id: str) -> dict:
        job = self.get(job_id)
        require(job["state"] == "quarantined", "only quarantined jobs can be retried")
        require(job["attempt"] < self.max_attempts, "content attempt budget exhausted")
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET state='planned', attempt=attempt+1, reason=NULL WHERE id=?",
                (job_id,))
            self._history(job_id, "planned", job["attempt"] + 1, "explicit retry")
        return self.get(job_id)

    def _artifact_root(self, job: dict) -> Path:
        return self.root / ("artifact-" + digest([job["id"], job["request_hash"], job["attempt"]]))

    def publish(self, job_id: str, artifact: dict) -> dict:
        job = self.get(job_id)
        if job["state"] == "accepted":
            self.reconcile()
            manifest = verify_manifest(self._artifact_root(job))
            require(manifest["metadata"]["artifact_hash"] == digest(artifact),
                    "accepted artifact is immutable")
            require(read_verified_json(self._artifact_root(job), "artifact.json") == artifact,
                    "accepted artifact content changed")
            return manifest
        require(job["state"] in {"verified", "adjudicated"}, "job is not ready for acceptance")
        root = self._artifact_root(job)
        root.mkdir(exist_ok=True)
        metadata = {"kind": "accepted_job", "job_id": job_id, "request_hash": job["request_hash"],
                    "attempt": job["attempt"], "artifact_hash": digest(artifact)}
        if (root / "manifest.json").exists():
            manifest = verify_manifest(root)
            require(manifest["metadata"] == metadata, "committed artifact identity mismatch")
            require(read_verified_json(root, "artifact.json") == artifact,
                    "committed artifact content mismatch")
        else:
            path = root / "artifact.json"
            if path.exists():
                require(read_json(path) == artifact,
                        "orphan artifact differs; quarantine before retry")
            else:
                atomic_write_json(path, artifact)
            manifest = write_manifest(root, ["artifact.json"], metadata)
        self._accept(job, manifest)
        return manifest

    def _accept(self, job: dict, manifest: dict) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE jobs SET state='accepted', artifact=?, artifact_hash=? WHERE id=?",
                (self._artifact_root(job).name, manifest["metadata"]["artifact_hash"], job["id"]))
            self._history(job["id"], "accepted", job["attempt"], "manifest published")

    def reconcile(self) -> None:
        for raw in self.connection.execute("SELECT * FROM jobs").fetchall():
            job = dict(raw)
            root = self._artifact_root(job)
            if (root / "manifest.json").exists() or job["state"] == "accepted":
                manifest = verify_manifest(root)
                require(set(manifest["files"]) == {"artifact.json"}, "incomplete committed job")
                meta = manifest["metadata"]
                require(meta.get("kind") == "accepted_job" and meta.get("job_id") == job["id"]
                        and meta.get("request_hash") == job["request_hash"]
                        and meta.get("attempt") == job["attempt"], "job commit mismatch")
                require(meta["artifact_hash"] == digest(read_verified_json(root, "artifact.json")),
                        "job artifact content identity mismatch")
                if job["state"] == "accepted":
                    require(job["artifact_hash"] == meta["artifact_hash"]
                            and job["artifact"] == root.name,
                            "accepted artifact differs from durable ledger")
                require(job["state"] in {"verified", "adjudicated", "accepted"},
                        "commit exists for invalid job state")
                if job["state"] != "accepted":
                    self._accept(job, manifest)

    def counts(self) -> dict:
        self.reconcile()
        counts = {key: 0 for key in TRANSITIONS}
        for state, count in self.connection.execute(
                "SELECT state, COUNT(*) FROM jobs GROUP BY state"):
            counts[state] = count
        return {"schema_version": 1, "counts": counts, "planned_total": sum(counts.values()),
                "pending": sum(counts[key] for key in
                               ("planned", "authored", "verified", "adjudicated")),
                "content_attempts": self.connection.execute(
                    "SELECT COALESCE(SUM(attempt),0) FROM jobs").fetchone()[0],
                "history": [dict(item) for item in self.connection.execute(
                    "SELECT * FROM history ORDER BY sequence")]}
