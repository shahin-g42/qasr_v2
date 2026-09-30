"""Durable per-clip run state: recovery authority for one exhaustive slice.

This module owns the crash-consistency story for stage 2 of the exhaustive
build. One :class:`SliceState` is the single durable record for one
(language, pool-part) slice: which input rows it owns, what has been tried,
what succeeded, what is quarantined, and what has been published to disk.

The design principle is that **the commit descriptor is the recovery
authority**. Durable manifests/sidecars/quarantine segments are written
first (fsync, rename, fsync directory), and only then is a checksummed
descriptor published (fsync) that names them. The SQLite index is a
rebuildable cache of descriptor contents: it may lag, never lead. On
restart, descriptors are replayed; orphan files without a descriptor are
ignored (they will be rewritten identically).

Input handling follows the same rule: :meth:`SliceState.enqueue` advances
the durable input cursor in the same transaction that stores the bounded
work window, so a crash between reading input and recording ownership never
loses or duplicates a row.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import socket
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .canonical import (
    Meta,
    Sample,
    write_manifest,
    write_sidecar,
)

__all__ = [
    "RunOwner",
    "SliceState",
    "atomic_json",
    "atomic_jsonl",
    "file_digest",
    "item_id",
    "preflight_storage",
    "read_jsonl",
    "recover_owner",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS window (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    item TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cursor (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id TEXT NOT NULL, stage TEXT NOT NULL, generation INTEGER NOT NULL DEFAULT 0,
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (id, stage, generation)
);
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT NOT NULL, stage TEXT NOT NULL, generation INTEGER NOT NULL DEFAULT 0,
    data TEXT NOT NULL,
    PRIMARY KEY (id, stage, generation)
);
CREATE TABLE IF NOT EXISTS results (
    id TEXT NOT NULL PRIMARY KEY,
    generation INTEGER NOT NULL DEFAULT 0,
    result TEXT NOT NULL,
    published INTEGER NOT NULL DEFAULT 0,
    descriptor TEXT
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    payload TEXT NOT NULL
);
"""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path: str | Path) -> str:
    """SHA-256 of a file's bytes."""
    return _sha256_file(Path(path))


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: str | Path, payload: Any) -> dict:
    """Write ``payload`` atomically; returns the file's digest metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def atomic_jsonl(path: str | Path, rows: Any) -> dict:
    """Write an iterable of rows as JSONL atomically; returns digest metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)
    return {"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size, "rows": n}


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream one JSON object per line (tolerates blank lines)."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def item_id(audio_filepath: str) -> str:
    """Stable item identity: sha256 of the exact audio path."""
    return hashlib.sha256(audio_filepath.encode("utf-8")).hexdigest()


def preflight_storage(root: str | Path) -> dict:
    """Verify rename/fsync/locking work in ``root``; fail closed.

    This validates the local filesystem mechanics the commit protocol needs.
    It does NOT claim network-storage-wide crash guarantees; the one-writer
    ownership rule remains the safety argument on shared storage.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".preflight"
    meta = atomic_json(probe, {"check": "preflight"})
    if meta["sha256"] != _sha256_file(probe):
        raise RuntimeError(f"storage digest check failed under {root}")
    db = root / ".preflight.sqlite3"
    conn = sqlite3.connect(str(db), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("CREATE TABLE IF NOT EXISTS t (x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.execute("COMMIT")
        # flock on a plain fd for the same file: sqlite3.Connection exposes no
        # fileno, and advisory-lock support is part of the durability preflight.
        lock_fd = os.open(str(db), os.O_RDONLY)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)
    finally:
        conn.close()
        db.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)
    return {"ok": True, "root": str(root)}


class RunOwner:
    """Exclusive owner of a resource directory for the lifetime of the object.

    Ownership is an atomically-created directory containing host/pid/run
    metadata. There is NO automatic takeover of a stale owner: recovery is an
    explicit, confirmed operation (:func:`recover_owner`), because a heartbeat
    expiring is not proof the previous writer stopped touching the files.
    """

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self._held = False

    def acquire(self) -> RunOwner:
        try:
            self.path.mkdir(parents=True)  # exclusive: fails if it exists
        except FileExistsError as exc:
            raise RuntimeError(
                f"resource {self.path} is already owned; explicit recovery is "
                "required (recover_owner) after confirming the previous owner stopped"
            ) from exc
        meta = {"host": socket.gethostname(), "pid": os.getpid(), "run_id": self.run_id}
        atomic_json(self.path / "owner.json", meta)
        self._held = True
        return self

    def release(self) -> None:
        if not self._held:
            return
        (self.path / "owner.json").unlink(missing_ok=True)
        # rmdir may fail while the directory is not empty: leave the evidence
        # behind rather than mask the cause.
        with contextlib.suppress(OSError):
            self.path.rmdir()
        self._held = False

    def __enter__(self) -> RunOwner:
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def recover_owner(path: str | Path, *, confirmed_dead: bool) -> None:
    """Remove an owner directory after explicit confirmation.

    ``confirmed_dead=True`` asserts the caller verified the previous owner is
    not running (same host + pid check is performed when the recorded host
    matches this machine, as a belt-and-braces guard).
    """
    path = Path(path)
    meta_file = path / "owner.json"
    if meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("host") == socket.gethostname():
            try:
                os.kill(int(meta.get("pid", -1)), 0)
            except OSError:
                pass  # pid gone: safe
            else:
                raise RuntimeError(
                    f"owner pid {meta.get('pid')} is still alive on this host; "
                    "refusing recovery"
                )
    if not confirmed_dead:
        raise RuntimeError("recovery requires confirmed_dead=True")
    meta_file.unlink(missing_ok=True)
    with contextlib.suppress(OSError):
        path.rmdir()


class SliceState:
    """Durable state for one (lang, pool-part) slice of the exhaustive build.

    Owns: the bounded input window + durable cursor, per-item per-stage
    attempt counts and snapshots, terminal results, and the published-output
    index. The SQLite file uses rollback-journal DELETE + ``synchronous=FULL``
    with explicit bounded transactions (shared-filesystem WAL is not assumed
    safe merely because there is one writer).
    """

    def __init__(self, state_dir: str | Path, output_dir: str | Path, *,
                 run_id: str, lang: str, part: int, fingerprint: str) -> None:
        self.state_dir = Path(state_dir)
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.lang = lang
        self.part = part
        self.fingerprint = fingerprint
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / f"slice_{lang}_p{part:04d}.sqlite3"
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript(_SCHEMA)
        self._ensure_run()
        self._replayed = False
        # Test hook: crash injection points ("after_files", "after_descriptor").
        self._checkpoint = lambda label: None

    # -- lifecycle -----------------------------------------------------------
    def _ensure_run(self) -> None:
        row = self._conn.execute("SELECT value FROM cursor WHERE name='init'").fetchone()
        if row is None:
            self._tx(lambda c: c.execute(
                "INSERT INTO cursor(name, value) VALUES ('init', 1)"))
            # Run identity lives beside the database as a small fsynced file;
            # opening an existing slice state under a different run/fingerprint
            # is a hard error (deleted-run claims must never be reused).
            with open(self.db_path.with_suffix(".meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"run_id": self.run_id, "lang": self.lang, "part": self.part,
                           "fingerprint": self.fingerprint}, fh, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
        else:
            meta_file = self.db_path.with_suffix(".meta.json")
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("run_id") != self.run_id or meta.get("fingerprint") != self.fingerprint:
                    raise RuntimeError(
                        f"slice state {self.db_path} belongs to run "
                        f"{meta.get('run_id')!r}, not {self.run_id!r}")

    def _tx(self, fn) -> None:
        """One bounded explicit transaction; rollback-journal, FULL sync."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            fn(self._conn)
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SliceState:
        self.replay_committed()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- durable input window --------------------------------------------------
    def enqueue(self, items: list[dict], *, cursor: int) -> None:
        """Record a bounded work window and advance the input cursor atomically."""
        if not items:
            self._tx(lambda c: c.execute(
                "INSERT INTO cursor(name, value) VALUES ('input', ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value", (cursor,)))
            return
        def _write(c):
            for it in items:
                c.execute(
                    "INSERT INTO window(item, generation) VALUES (?, 0)",
                    (json.dumps(it, ensure_ascii=False, sort_keys=True),))
            c.execute(
                "INSERT INTO cursor(name, value) VALUES ('input', ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value", (cursor,))
        self._tx(_write)

    @property
    def input_cursor(self) -> int:
        row = self._conn.execute(
            "SELECT value FROM cursor WHERE name='input'").fetchone()
        return int(row[0]) if row else 0

    # -- attempts / snapshots ---------------------------------------------------
    def attempts_for(self, item_id_: str, *, generation: int = 0) -> dict[str, int]:
        return {r[0]: r[1] for r in self._conn.execute(
            "SELECT stage, count FROM attempts WHERE id=? AND generation=?",
            (item_id_, generation))}

    def snapshots_for(self, item_id_: str, *, generation: int = 0) -> dict[str, Any]:
        return {r[0]: json.loads(r[1]) for r in self._conn.execute(
            "SELECT stage, data FROM snapshots WHERE id=? AND generation=?",
            (item_id_, generation))}

    # -- scheduler events ---------------------------------------------------------
    def event(self, ev: dict) -> None:
        """Record one scheduler event (main/owner thread only).

        Types: ``attempt`` (id, stage, generation, attempt), ``stage_result``
        (id, stage, generation, data), ``terminal`` (id, generation, result).
        Unknown types raise, so a producer/consumer schema drift fails loudly.
        """
        kind = ev.get("type")
        if kind == "attempt":
            self._bump_attempt(ev["id"], ev["stage"], ev.get("generation", 0),
                               int(ev.get("attempt", 1)))
        elif kind == "stage_result":
            self._put_snapshot(ev["id"], ev["stage"], ev.get("generation", 0), ev.get("data"))
        elif kind == "terminal":
            self.record_result(ev["result"])
        else:
            raise ValueError(f"unknown event type: {kind!r}")

    def _bump_attempt(self, id_: str, stage: str, generation: int, attempt: int) -> None:
        self._tx(lambda c: c.execute(
            "INSERT INTO attempts(id, stage, generation, count) VALUES (?,?,?,?) "
            "ON CONFLICT(id, stage, generation) DO UPDATE SET count=excluded.count",
            (id_, stage, generation, max(1, attempt))))

    def _put_snapshot(self, id_: str, stage: str, generation: int, data: Any) -> None:
        blob = json.dumps(data if data is not None else {}, ensure_ascii=False, sort_keys=True)
        self._tx(lambda c: c.execute(
            "INSERT INTO snapshots(id, stage, generation, data) VALUES (?,?,?,?) "
            "ON CONFLICT(id, stage, generation) DO UPDATE SET data=excluded.data",
            (id_, stage, generation, blob)))

    # -- pending work ------------------------------------------------------------
    def pending(self) -> Iterator[dict]:
        """Yield unfinished items annotated with durable attempts/snapshots.

        Items that already have a terminal result are skipped (they are
        awaiting publication, not more work).
        """
        rows = self._conn.execute(
            "SELECT seq, item, generation FROM window ORDER BY seq").fetchall()
        done = {r[0] for r in self._conn.execute(
            "SELECT id FROM results WHERE published=1 AND "
            "json_extract(result, '$.status')='accepted'")}
        # A published QUARANTINE is terminal only for its own generation: a
        # requeued item (generation bumped by start_retry_generation) is
        # pending work again, while an unresolved same-generation result is
        # awaiting publication, not more work.
        terminal = {r[0]: int(r[1] or 0) for r in self._conn.execute(
            "SELECT id, generation FROM results WHERE published=1")}
        awaiting = {r[0]: int(r[1] or 0) for r in self._conn.execute(
            "SELECT id, generation FROM results WHERE published=0")}
        for seq, blob, generation in rows:
            it = json.loads(blob)
            id_ = it.get("id") or item_id(it["audio_filepath"])
            if id_ in done or terminal.get(id_, -1) >= generation \
                    or awaiting.get(id_, -1) >= generation:
                continue
            it["_id"] = id_
            it["_generation"] = generation
            it["_attempts"] = self.attempts_for(id_, generation=generation)
            it["_snapshots"] = self.snapshots_for(id_, generation=generation)
            it["_seq"] = seq
            yield it

    # -- terminal results -----------------------------------------------------------
    def record_result(self, result: dict) -> None:
        """Persist one terminal outcome durably (not yet published).

        Accepted results are immutable once published. A published quarantine
        can be superseded only by a later-generation ACCEPTED result (that is
        how a retry generation records resolution); it can never be demoted.
        """
        id_ = result["id"]
        self._tx(lambda c: c.execute(
            "INSERT INTO results(id, generation, result, published) VALUES (?,?,?,0) "
            "ON CONFLICT(id) DO UPDATE SET result=excluded.result, "
            "generation=excluded.generation, published=0 "
            "WHERE results.published=0 "
            "OR json_extract(results.result, '$.status')='quarantined'",
            (id_, int(result.get("generation", 0)),
             json.dumps(result, ensure_ascii=False, sort_keys=True))))

    def _unpublished(self, limit: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, result FROM results WHERE published=0 ORDER BY id LIMIT ?",
            (limit,)).fetchall()
        return [json.loads(blob) for _, blob in rows]

    # -- publication -----------------------------------------------------------------
    def publish_ready(self, *, max_items: int = 256) -> dict | None:
        """Publish up to ``max_items`` terminal results as one committed segment.

        Order of operations (each fsynced before the next step):
        1. manifest + sidecar + quarantine segment files (if any rows)
        2. commit descriptor naming them, with checksums (authoritative)
        3. SQLite marks those results published, referencing the descriptor

        Returns the descriptor dict, or None when nothing is unpublished.
        """
        results = self._unpublished(max_items)
        if not results:
            return None
        seg_dir = self.output_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        n = self._next_segment()
        base = f"{self.lang}_p{self.part:04d}_seg{n:05d}"

        accepted, quarantined = [], []
        for r in results:
            (accepted if r.get("status") == "accepted" else quarantined).append(r)
        files: dict[str, dict] = {}

        if accepted:
            samples, metas = [], []
            for r in accepted:
                it = r["item"]
                samples.append(Sample(audio_filepath=it["audio_filepath"],
                                      duration=float(it.get("duration") or 0.0),
                                      text=r.get("text") or it.get("normalized_text") or "",
                                      lang=it.get("lang") or self.lang))
                metas.append(Meta(
                    audio_filepath=it["audio_filepath"],
                    dataset=it.get("source") or "internal",
                    quality=(lambda v: float(v) if v is not None else None)(it.get("quality")),
                    accent=it.get("accent"),
                    source_text=it.get("text"),
                    normalized_text=it.get("normalized_text"),
                ))
                # provenance/attempt detail that the four-key manifest cannot hold:
                metas[-1].metrics["attempts"] = float(sum(r.get("attempts", {}).values()))
                metas[-1].metrics["quarantine_provenance_count"] = float(
                    it.get("provenance_count") or 0)
                metas[-1].metrics["gen"] = float(r.get("generation", 0))
                metas[-1].stages["llm"] = "ok"
                metas[-1].stages["review"] = "ok"
                metas[-1].final_text = r.get("text")
            man = seg_dir / f"{base}.manifest.jsonl"
            side = seg_dir / f"{base}.sidecar.jsonl"
            write_manifest(man, samples)
            write_sidecar(side, metas)
            files["manifest"] = {"path": str(man), "sha256": _sha256_file(man),
                                 "bytes": man.stat().st_size, "rows": len(accepted)}
            files["sidecar"] = {"path": str(side), "sha256": _sha256_file(side),
                                "bytes": side.stat().st_size, "rows": len(accepted)}
        if quarantined:
            qpath = seg_dir / f"{base}.quarantine.jsonl"
            qmeta = atomic_jsonl(qpath, quarantined)
            files["quarantine"] = qmeta

        self._checkpoint("after_files")
        descriptor = {
            "run_id": self.run_id, "lang": self.lang, "part": self.part,
            "segment": n, "base": base, "files": files,
            "accepted": len(accepted), "quarantined": len(quarantined),
            "ids": [r["id"] for r in results],
            "fingerprint": self.fingerprint,
        }
        dpath = seg_dir / f"{base}.commit.json"
        dmeta = atomic_json(dpath, descriptor)
        self._checkpoint("after_descriptor")

        def _mark(c):
            for id_ in descriptor["ids"]:
                c.execute(
                    "UPDATE results SET published=1, descriptor=? WHERE id=? AND published=0",
                    (dmeta["path"], id_))
        self._tx(_mark)
        self._checkpoint("after_index")
        return descriptor

    def _next_segment(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(DISTINCT descriptor) FROM results WHERE published=1").fetchone()
        return int(row[0])

    def committed_descriptors(self) -> Iterator[dict]:
        """Stream all published commit descriptors (dedup by path)."""
        seen: set[str] = set()
        for (dpath,) in self._conn.execute(
                "SELECT DISTINCT descriptor FROM results WHERE published=1 AND descriptor IS NOT NULL"):
            if dpath in seen:
                continue
            seen.add(dpath)
            with open(dpath, encoding="utf-8") as fh:
                yield json.load(fh)

    # -- recovery -----------------------------------------------------------------------
    def replay_committed(self) -> int:
        """Verify published descriptors and rebuild the index after any crash.

        Orphan files without a descriptor are ignored (rewritten later).
        Returns the number of descriptors replayed. Idempotent.
        """
        seg_dir = self.output_dir / "segments"
        if not seg_dir.is_dir():
            return 0
        n = 0
        for dpath in sorted(seg_dir.glob("*.commit.json")):
            desc = json.loads(dpath.read_text(encoding="utf-8"))
            if desc.get("run_id") != self.run_id:
                continue
            # Verify descriptor integrity: named files must exist with the
            # recorded checksums.
            for role, meta in desc.get("files", {}).items():
                p = Path(meta["path"])
                if not p.exists() or _sha256_file(p) != meta["sha256"]:
                    raise RuntimeError(
                        f"published segment file missing/corrupt: {role} {p}")
            rows: list[dict] = []
            for role in ("manifest", "sidecar", "quarantine"):
                meta = desc.get("files", {}).get(role)
                if not meta:
                    continue
                rows.extend(read_jsonl(meta["path"]))
            by_id = {}
            for r in rows:
                rid = r.get("id")
                if rid:
                    by_id[rid] = r
            def _mark(c, desc=desc, dpath=str(dpath), by_id=by_id):
                for id_ in desc["ids"]:
                    result = by_id.get(id_)
                    c.execute(
                        "INSERT INTO results(id, generation, result, published, descriptor) "
                        "VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                        "published=1, descriptor=excluded.descriptor "
                        "WHERE results.published=0",
                        (id_, int(result.get("generation", 0)) if result else 0,
                         json.dumps(result, ensure_ascii=False, sort_keys=True)
                         if result else "{}", 1, dpath))
            self._tx(_mark)
            n += 1
        self._replayed = True
        return n

    # -- retry generations ----------------------------------------------------------------
    def start_retry_generation(self, *, reasons: list[str] | None = None) -> int:
        """Create a new retry generation over unresolved quarantine items.

        Returns the new generation number. Attempt budgets reset only for the
        selected items; accepted results are immutable. Quarantine resolution
        is represented by a later commit of the item with status accepted;
        historical failures remain in earlier segments for audit.
        """
        gen_row = self._conn.execute(
            "SELECT COALESCE(MAX(generation),0) FROM results").fetchone()
        new_gen = int(gen_row[0]) + 1
        def _select(c):
            rows = c.execute(
                "SELECT id, result FROM results WHERE published=1").fetchall()
            picked = []
            for id_, blob in rows:
                r = json.loads(blob)
                if r.get("status") != "quarantined":
                    continue
                if reasons and r.get("reason") not in reasons:
                    continue
                picked.append((id_, r))
            for id_, r in picked:
                c.execute(
                    "INSERT INTO window(item, generation) VALUES (?, ?)",
                    (json.dumps(r["item"], ensure_ascii=False, sort_keys=True), new_gen))
                c.execute("DELETE FROM attempts WHERE id=? AND generation=?",
                          (id_, new_gen))
            c.execute(
                "INSERT INTO cursor(name, value) VALUES ('retry_gen', ?) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value", (new_gen,))
        self._tx(_select)
        return new_gen

    # -- stats ------------------------------------------------------------------------------
    def stats(self) -> dict:
        def one(sql: str):
            return self._conn.execute(sql).fetchone()[0]

        return {
            "lang": self.lang, "part": self.part, "run_id": self.run_id,
            "input_cursor": self.input_cursor,
            "window_open": one("SELECT COUNT(*) FROM window w WHERE NOT EXISTS "
                               "(SELECT 1 FROM results r WHERE r.id = json_extract(w.item, '$.id'))"),
            "results_unpublished": one("SELECT COUNT(*) FROM results WHERE published=0"),
            "results_published": one("SELECT COUNT(*) FROM results WHERE published=1"),
            "accepted_published": one(
                "SELECT COUNT(*) FROM results WHERE published=1 AND "
                "json_extract(result,'$.status')='accepted'"),
            "quarantine_published": one(
                "SELECT COUNT(*) FROM results WHERE published=1 AND "
                "json_extract(result,'$.status')='quarantined'"),
            "replayed": self._replayed,
        }
