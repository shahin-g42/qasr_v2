"""Tests for the durable slice state: atomicity, publication, recovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data_processing.run_state import (
    RunOwner,
    SliceState,
    atomic_json,
    atomic_jsonl,
    file_digest,
    item_id,
    preflight_storage,
    read_jsonl,
    recover_owner,
)


def _item(path: str, text: str = "hello") -> dict:
    return {
        "id": item_id(path),
        "audio_filepath": path,
        "lang": "ar",
        "normalized_text": text,
        "duration": 1.5,
        "source": "internal_v76_ar_inworld_full",
        "accent": "Gulf",
    }


def _accepted(item: dict, text: str = "corrected") -> dict:
    return {
        "id": item["id"],
        "status": "accepted",
        "text": text,
        "item": item,
        "attempts": {"correct": 1, "review": 1},
        "generation": 0,
    }


def _quarantined(item: dict, reason: str = "parse") -> dict:
    return {
        "id": item["id"],
        "status": "quarantined",
        "reason": reason,
        "item": item,
        "attempts": {"correct": 5},
        "generation": 0,
    }


class TestAtomicFiles(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def test_atomic_json_roundtrip_and_digest(self) -> None:
        meta = atomic_json(self.tmp / "x" / "y.json", {"b": 1, "a": "ü"})
        self.assertEqual(json.loads((self.tmp / "x" / "y.json").read_text()), {"a": "ü", "b": 1})
        self.assertEqual(meta["sha256"], file_digest(self.tmp / "x" / "y.json"))
        self.assertFalse(list(self.tmp.glob("**/*.tmp")))

    def test_atomic_jsonl_and_read_jsonl_skip_blanks(self) -> None:
        meta = atomic_jsonl(self.tmp / "f.jsonl", [{"a": 1}, {"b": "نص"}])
        self.assertEqual(meta["rows"], 2)
        self.assertEqual(list(read_jsonl(self.tmp / "f.jsonl")), [{"a": 1}, {"b": "نص"}])


class TestPreflight(unittest.TestCase):
    def test_preflight_ok_and_leaves_no_probe_files(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.assertEqual(preflight_storage(tmp), {"ok": True, "root": str(tmp)})
        self.assertEqual(list(tmp.iterdir()), [])


class TestRunOwner(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "own"

    def test_exclusive_acquire_and_release(self) -> None:
        with RunOwner(self.path, "r1"):
            self.assertTrue(self.path.is_dir())
            with self.assertRaisesRegex(RuntimeError, "already owned"):
                RunOwner(self.path, "r2").acquire()
        RunOwner(self.path, "r2").acquire().release()
        self.assertFalse(self.path.exists())

    def test_recovery_requires_confirmation_and_liveness(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "confirmed_dead"):
            recover_owner(self.path, confirmed_dead=False)
        RunOwner(self.path, "r1").acquire()  # left held by "crashed" owner
        with self.assertRaisesRegex(RuntimeError, "still alive"):
            recover_owner(self.path, confirmed_dead=True)  # our own pid recorded
        # Simulate a dead pid on this host: rewrite owner metadata.
        meta = json.loads((self.path / "owner.json").read_text())
        meta["pid"] = 2**22  # assumed unallocated
        (self.path / "owner.json").write_text(json.dumps(meta))
        recover_owner(self.path, confirmed_dead=True)
        self.assertFalse(self.path.exists())


class TestSliceState(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.fp = "fp123"

    def _state(self, run_id: str = "r1") -> SliceState:
        return SliceState(self.tmp / "state", self.tmp / "out", run_id=run_id,
                          lang="ar", part=3, fingerprint=self.fp)

    def test_run_fingerprint_mismatch_is_rejected(self) -> None:
        with self._state() as st:
            st.enqueue([_item("a.wav")], cursor=1)
        with self.assertRaisesRegex(RuntimeError, "belongs to run"):
            self._state(run_id="other").__enter__()

    def test_enqueue_cursor_and_pending(self) -> None:
        item = _item("a.wav")
        with self._state() as st:
            st.enqueue([item], cursor=10)
            self.assertEqual(st.input_cursor, 10)
            pend = list(st.pending())
            self.assertEqual(len(pend), 1)
            self.assertEqual(pend[0]["_id"], item["id"])
            st.record_result(_accepted(item))
            self.assertEqual(list(st.pending()), [])  # terminal -> no longer pending

    def test_attempts_accumulate_per_stage_and_generation(self) -> None:
        item = _item("a.wav")
        with self._state() as st:
            st.event({"type": "attempt", "id": item["id"], "stage": "correct", "attempt": 1})
            st.event({"type": "attempt", "id": item["id"], "stage": "correct", "attempt": 2})
            st.event({"type": "attempt", "id": item["id"], "stage": "review", "attempt": 1})
            self.assertEqual(st.attempts_for(item["id"]), {"correct": 2, "review": 1})
            st.event({"type": "stage_result", "id": item["id"], "stage": "correct",
                      "data": {"ok": True}})
            self.assertEqual(st.snapshots_for(item["id"]), {"correct": {"ok": True}})

    def test_unknown_event_type_raises(self) -> None:
        with self._state() as st, self.assertRaises(ValueError):
            st.event({"type": "nope"})

    def test_publish_segments_and_descriptor_index(self) -> None:
        items = [_item(f"{i}.wav") for i in range(3)]
        with self._state() as st:
            st.enqueue(items, cursor=3)
            for it in items:
                st.record_result(_accepted(it))
            desc = st.publish_ready(max_items=2)
            self.assertEqual(desc["accepted"], 2)
            self.assertEqual(st.stats()["results_unpublished"], 1)
            desc2 = st.publish_ready()
            self.assertEqual(desc2["accepted"], 1)
            self.assertEqual(st.stats()["results_published"], 3)
            self.assertEqual(st.stats()["accepted_published"], 3)
            # published files are intact and match their checksums
            for d in (desc, desc2):
                man = Path(d["files"]["manifest"]["path"])
                self.assertEqual(file_digest(man), d["files"]["manifest"]["sha256"])
                self.assertEqual(len(list(read_jsonl(man))), d["accepted"])

    def test_publish_mixed_accepted_and_quarantine(self) -> None:
        a, q = _item("a.wav"), _item("b.wav")
        with self._state() as st:
            st.enqueue([a, q], cursor=2)
            st.record_result(_accepted(a))
            st.record_result(_quarantined(q))
            desc = st.publish_ready()
            self.assertEqual((desc["accepted"], desc["quarantined"]), (1, 1))
            qrows = list(read_jsonl(Path(desc["files"]["quarantine"]["path"])))
            self.assertEqual(qrows[0]["status"], "quarantined")

    def test_crash_after_files_before_descriptor_leaves_orphans_ignored(self) -> None:
        item = _item("a.wav")
        st = self._state()
        st._checkpoint = lambda label: (_ for _ in ()).throw(
            RuntimeError("crash")) if label == "after_files" else None
        with st:
            st.enqueue([item], cursor=1)
            st.record_result(_accepted(item))
            with self.assertRaises(RuntimeError):
                st.publish_ready()
        # Restart: the orphan segment has no descriptor -> ignored, republished.
        with self._state() as st2:
            self.assertEqual(st2.replay_committed(), 0)
            self.assertEqual(st2.stats()["results_published"], 0)
            desc = st2.publish_ready()
            self.assertEqual(desc["accepted"], 1)

    def test_crash_after_descriptor_before_index_recovers_on_replay(self) -> None:
        item = _item("a.wav")
        st = self._state()
        st._checkpoint = lambda label: (_ for _ in ()).throw(
            RuntimeError("crash")) if label == "after_descriptor" else None
        with st:
            st.enqueue([item], cursor=1)
            st.record_result(_accepted(item))
            with self.assertRaises(RuntimeError):
                st.publish_ready()
        with self._state() as st2:
            self.assertEqual(st2.replay_committed(), 1)
            self.assertEqual(st2.stats()["results_published"], 1)
            self.assertEqual(st2.publish_ready(), None)  # nothing left unpublished

    def test_corrupt_published_file_fails_replay_loudly(self) -> None:
        item = _item("a.wav")
        with self._state() as st:
            st.enqueue([item], cursor=1)
            st.record_result(_accepted(item))
            st.publish_ready()
        man = next((self.tmp / "out" / "segments").glob("*.manifest.jsonl"))
        man.write_text("corrupted\n")
        # __enter__ replays committed descriptors, so the corrupt file must
        # fail the constructor-with, loudly, before any work resumes.
        with self.assertRaisesRegex(RuntimeError, "missing/corrupt"), self._state():
            pass

    def test_retry_generation_requeues_quarantine_only(self) -> None:
        a, q = _item("a.wav"), _item("b.wav")
        with self._state() as st:
            st.enqueue([a, q], cursor=2)
            st.record_result(_accepted(a))
            st.record_result(_quarantined(q, reason="parse"))
            st.publish_ready()
            gen = st.start_retry_generation(reasons=["parse"])
            self.assertEqual(gen, 1)
            pend = [p for p in st.pending() if p["_generation"] == gen]
            self.assertEqual([p["id"] for p in pend], [q["id"]])
            self.assertEqual(pend[0]["_attempts"], {})  # budget reset for the new generation
            # resolve it in the new generation
            resolved = _accepted(q)
            resolved["generation"] = gen
            st.record_result(resolved)
            st.publish_ready()
            self.assertEqual(st.stats()["accepted_published"], 2)

    def test_record_result_immutable_once_published(self) -> None:
        item = _item("a.wav")
        with self._state() as st:
            st.enqueue([item], cursor=1)
            st.record_result(_accepted(item, text="first"))
            st.publish_ready()
            st.record_result(_accepted(item, text="second"))  # must not overwrite
            self.assertEqual(st.publish_ready(), None)
            rows = list(st.committed_descriptors())
            self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
