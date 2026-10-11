"""Parallel verification must find corruption without changing committed data."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_processing.asr_clean.__main__ import main
from data_processing.asr_clean.worker import RunPaths, verify


def _rows(*audio_paths) -> bytes:
    return b"".join(
        (json.dumps({"audio_filepath": audio, "duration": 1.0, "text": "علّم",
                     "org_text": "علم", "asr_text": "علم"}, ensure_ascii=False) + "\n").encode()
        for audio in audio_paths
    )


class VerifyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = str(Path(self.tmp.name) / "run")
        self.paths = RunPaths(self.root)
        self.chunks: list[dict] = []
        self._save_plan()

    def _save_plan(self) -> None:
        (self.paths.state / "plan.json").write_text(json.dumps({"chunks": self.chunks}))

    def _chunk(self, data: bytes, *, written: int = 1, out_bytes: int | None = None,
               started: bool = True, status: str = "done") -> dict:
        part = len(self.chunks)
        chunk = {"id": f"en__data__train_en_x__{part:05d}", "lang": "en",
                 "stem": "data/train_en_x", "part": part}
        self.chunks.append(chunk)
        self._save_plan()
        path = self.paths.part(chunk)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if started:
            (self.paths.progress / f"{chunk['id']}.json").write_text(json.dumps({
                "status": status, "out_bytes": len(data) if out_bytes is None else out_bytes,
                "counts": {"written": written},
            }))
        return chunk

    def _snapshot(self) -> dict[str, bytes]:
        return {str(p.relative_to(self.root)): p.read_bytes()
                for p in Path(self.root).rglob("*") if p.is_file()}

    def test_parallel_matches_serial_for_valid_and_corrupt_chunks(self) -> None:
        good = _rows("/a.wav", "/b.wav")
        # An active chunk's uncommitted/torn tail must never be parsed.
        self._chunk(good + b'{"torn', written=2, out_bytes=len(good), status="running")
        self._chunk(b"", written=0)
        expected = {}
        cases = [
            (_rows("/duplicate.wav", "/duplicate.wav"), 2, "audio written twice: /duplicate.wav"),
            (b'{"broken"\n', 1, "unparseable line 1"),
            (b'\xff\n', 1, "unparseable line 1"),
            (b'null\n', 1, "line 1 is not an object"),
            (b'{"text": "only one column"}\n', 1, "line 1 has columns ['text']"),
            (_rows(["/not-a-string.wav"]), 1, "line 1 has non-string audio_filepath"),
            (_rows("/a.wav"), 2, "1 records on disk, progress counted 2"),
        ]
        for data, written, reason in cases:
            chunk = self._chunk(data, written=written)
            expected[chunk["id"]] = reason
        chunk = self._chunk(good, written=2, out_bytes=len(good) + 10)
        expected[chunk["id"]] = f"part file {len(good)} bytes < committed {len(good) + 10}"
        chunk = self._chunk(good, written=2)
        self.paths.part(chunk).unlink()
        expected[chunk["id"]] = f"part file -1 bytes < committed {len(good)}"
        self._chunk(b"not committed", started=False)
        before = self._snapshot()

        serial = verify(self.root, jobs=1)
        parallel = verify(self.root, jobs=2)

        self.assertEqual(serial, {"checked": 11, "bad": expected, "reset": False})
        self.assertEqual(parallel, serial)
        self.assertEqual(list(parallel["bad"]), sorted(expected))
        self.assertEqual(self._snapshot(), before)

    def test_duplicates_are_checked_within_each_chunk(self) -> None:
        # The frozen plan owns global dedup; verify has always checked within a part.
        self._chunk(_rows("/same.wav"))
        self._chunk(_rows("/same.wav"))
        self.assertEqual(verify(self.root, jobs=2), {"checked": 2, "bad": {}, "reset": False})

    def test_parallel_reset_removes_only_bad_chunk_artifacts(self) -> None:
        good = self._chunk(_rows("/good.wav"))
        bad = self._chunk(_rows("/duplicate.wav", "/duplicate.wav"), written=2)
        unstarted = self._chunk(b"not committed", started=False)
        for chunk in (good, bad, unstarted):
            for path in (self.paths.rejects(chunk), self.paths.review(chunk), self.paths.claims / chunk["id"]):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("keep unless this chunk is bad")
        before = self._snapshot()
        removed = (self.paths.progress / f"{bad['id']}.json", self.paths.part(bad),
                   self.paths.rejects(bad), self.paths.review(bad), self.paths.claims / bad["id"])
        expected = {k: v for k, v in before.items()
                    if k not in {str(p.relative_to(self.root)) for p in removed}}

        result = verify(self.root, reset=True, jobs=2)

        self.assertEqual(result["checked"], 2)
        self.assertEqual(list(result["bad"]), [bad["id"]])
        self.assertTrue(result["reset"])
        self.assertEqual(self._snapshot(), expected)
        self.assertEqual(verify(self.root, jobs=2)["bad"], {})

    def test_failed_scan_does_not_reset_already_found_bad_chunks(self) -> None:
        self._chunk(_rows("/bad.wav", "/bad.wav"), written=2)
        self._chunk(_rows("/good.wav"))
        before = self._snapshot()
        with (patch("data_processing.asr_clean.worker._verify_part",
                    side_effect=["audio written twice: /bad.wav", RuntimeError("scan interrupted")]),
              self.assertRaisesRegex(RuntimeError, "scan interrupted")):
            verify(self.root, reset=True, jobs=1)
        self.assertEqual(self._snapshot(), before)

    def test_empty_run_and_invalid_jobs(self) -> None:
        self.assertEqual(verify(self.root, jobs=2), {"checked": 0, "bad": {}, "reset": False})
        for jobs in (0, -1):
            with self.subTest(jobs=jobs), self.assertRaisesRegex(ValueError, "jobs must be at least 1"):
                verify(self.root, jobs=jobs)

    def test_default_jobs_are_capped_at_committed_chunks(self) -> None:
        self._chunk(_rows("/a.wav"))
        self._chunk(_rows("/b.wav"))
        err = io.StringIO()
        cpu_method = "sched_getaffinity" if hasattr(os, "sched_getaffinity") else "cpu_count"
        cpus = set(range(64)) if cpu_method == "sched_getaffinity" else 64
        with (patch(f"data_processing.asr_clean.worker.os.{cpu_method}", return_value=cpus),
              contextlib.redirect_stderr(err)):
            result = verify(self.root, progress=True)
        self.assertEqual(result["bad"], {})
        self.assertIn("2 committed chunks with 2 processes", err.getvalue())
        self.assertIn("2/2 chunks checked (100.0%), 0 bad", err.getvalue())

    def test_cli_summary_progress_and_exit_status(self) -> None:
        self._chunk(_rows("/bad.wav", "/bad.wav"), written=2)
        for reset in (False, True):
            out, err = io.StringIO(), io.StringIO()
            args = ["verify", "--run-root", self.root, "--jobs", "1"]
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = main(args + (["--reset"] if reset else []))
            self.assertEqual(code, 0 if reset else 1)
            self.assertIn("checked 1 chunks: 1 bad", out.getvalue())
            self.assertIn("1/1 chunks checked (100.0%), 1 bad", err.getvalue())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(["verify", "--run-root", self.root, "--jobs", "0"])
        self.assertEqual(exc.exception.code, 2)

    def test_shell_forwards_jobs_and_explicit_option_overrides_environment(self) -> None:
        self._chunk(_rows("/a.wav"))
        self._chunk(_rows("/b.wav"))
        repo = Path(__file__).resolve().parents[1]
        env = {**os.environ, "RUN_ID": "test", "ROOT": self.root, "JOBS": "2", "PYTHON": sys.executable}
        for extra, jobs in (([], 2), (["--jobs", "1"], 1)):
            with self.subTest(jobs=jobs):
                result = subprocess.run(["bash", "scripts/corpus/run_asr_clean.sh", "verify", *extra],
                                        cwd=repo, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("checked 2 chunks: 0 bad", result.stdout)
                self.assertIn(f"2 committed chunks with {jobs} processes", result.stderr)


if __name__ == "__main__":
    unittest.main()
