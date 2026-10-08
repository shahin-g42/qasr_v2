"""scripts/asr_load_test.py against a fake vLLM server: sampling, load, scoring."""

from __future__ import annotations

import importlib.util
import json
import random
import struct
import sys
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("asr_load_test", SCRIPTS / "asr_load_test.py")
alt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alt)


class _FakeASR(BaseHTTPRequestHandler):
    """Echoes the transcript stored next to the clip, in the model's output format."""

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        url = body["messages"][1]["content"][0]["audio_url"]["url"]
        wav = Path(url[len("file://"):])
        text = wav.with_suffix(".txt").read_text(encoding="utf-8")
        payload = {
            "choices": [{"message": {"content": f"language {body['messages'][0]['content']}<asr_text>{text}"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": len(text)},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


def _wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\0\0" * int(16_000 * seconds))


class NormalizeTest(unittest.TestCase):
    def test_arabic_marks_do_not_split_words(self) -> None:
        self.assertEqual(alt.normalize("مَرْحَبًا بِكُم، أهلاً", "ar"), "مرحبا بكم اهلا")

    def test_indic_vowel_signs_survive(self) -> None:
        self.assertEqual(alt.normalize("नमस्ते दुनिया!", "hi"), "नमस्ते दुनिया")

    def test_digits_and_case_fold(self) -> None:
        self.assertEqual(alt.normalize("Room ١٢, OK.", "en"), "room 12 ok")


class SamplingTest(unittest.TestCase):
    def test_byte_seek_path_returns_whole_distinct_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.jsonl"
            path.write_text("".join(json.dumps({"i": i, "pad": "x" * (i % 50)}) + "\n" for i in range(5000)))
            lines = alt.sample_lines(str(path), 100, random.Random(0), small_bytes=0)
            self.assertEqual(len(lines), 100)
            ids = [json.loads(line)["i"] for line in lines]  # every line parses: no torn lines
            self.assertEqual(len(set(ids)), 100)

    def test_flac_streaminfo_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.flac"
            rate, total = 16_000, 16_000 * 7 + 400
            info = bytearray(34)
            info[10:13] = bytes([(rate >> 12) & 0xFF, (rate >> 4) & 0xFF, ((rate & 0xF) << 4)])
            info[13] = (total >> 32) & 0x0F
            info[14:18] = struct.pack(">I", total & 0xFFFFFFFF)
            path.write_bytes(b"fLaC" + b"\0\0\0\x22" + bytes(info))
            self.assertAlmostEqual(alt.audio_duration(str(path)), total / rate)

    def test_q3asr_envelope_is_stripped(self) -> None:
        row = alt.parse_row(json.dumps({"audio_filepath": "/x.wav", "text": "language Arabic<asr_text>نص",
                                        "duration": 2.0}), "ar", "m.jsonl")
        self.assertEqual(row["ref"], "نص")


class EndToEndTest(unittest.TestCase):
    def test_sweep_against_fake_server(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeASR)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                rows = []
                for i in range(20):
                    wav = Path(tmp) / f"c{i}.wav"
                    _wav(wav, 1.0 + i % 3)
                    text = f"hello world number {i}"
                    # every 4th clip: model output differs from the reference by one word
                    wav.with_suffix(".txt").write_text(text if i % 4 else f"hello word number {i}")
                    rows.append({"audio_filepath": str(wav), "text": text})  # no duration: header read
                manifest = Path(tmp) / "eval_en.jsonl"
                manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
                out = Path(tmp) / "out"
                argv = sys.argv
                sys.argv = ["asr_load_test.py", "--url", url, "--manifest", f"en={manifest}",
                            "--per-lang", "20", "--concurrency", "2,8", "--warmup", "4",
                            "--min-level-seconds", "0",
                            "--project-hours", "1000", "--out", str(out)]
                try:
                    self.assertEqual(alt.main(), 0)
                finally:
                    sys.argv = argv
                report = json.loads((out / "report.json").read_text())
        finally:
            server.shutdown()

        self.assertEqual([lv["concurrency"] for lv in report["levels"]], [2, 8])
        for lv in report["levels"]:
            self.assertEqual(lv["error_rate"], 0)
            self.assertEqual(lv["undated_clips"], 0)
        acc = report["accuracy"]["en"]
        self.assertEqual(acc["n"], 20)  # unique clips, not requests
        self.assertEqual(acc["lid_agree"], 1.0)
        # 5 of 20 clips have 1 wrong word out of 4 -> WER 5/80
        self.assertAlmostEqual(acc["wer"], round(5 / 80, 4))
        self.assertEqual(acc["issues"], {})  # the fake server is deterministic
        self.assertIn("projection", report)


class IssueDetectorTest(unittest.TestCase):
    """Cases taken from the first real load test (2026-10-08)."""

    def _rec(self, lang: str, ref: str, hyp: str, **kw) -> dict:
        return {"lang": lang, "ref": ref, "hyp": hyp, "finish": "stop", **kw}

    def test_malayalam_stripped_marks(self) -> None:
        rec = self._rec("ml", "ജനം, ജയ്ഹിന്ദ് തുടങ്ങിയ ചാനലുകളിലും സ്ഥിതിയും തഥൈവയാണ്.",
                        "ജന, ജയഹനദ തടങങയ ചനലകളല സഥത തഥവയണ.")
        self.assertIn("stripped_marks", alt.issues(rec))
        ok = self._rec("ml", "എന്നാൽ കാലക്രമേണ അദ്ദേഹത്തിന്റെ മഹത്ത്വം", "എന്നാൽ കാലക്രമേണ അദ്ദേഹത്തിന്റെ മഹത്വം")
        self.assertEqual(alt.issues(ok), [])

    def test_chinese_loops_and_romanization(self) -> None:
        loop = self._rec("zh", "江小源很郁闷的，在楼下等了一整天。", "将小圆在哪儿？在哪儿？在哪儿？")
        self.assertIn("loop", alt.issues(loop))
        roman = self._rec("zh", "混滔天明了，皇上又跟进如好，这就完了万岁。",
                          "Punpao tian, ming ah. Wang sheng yu geng jin yu hao.")
        self.assertIn("wrong_script", alt.issues(roman))

    def test_natural_repetition_in_reference_is_not_a_loop(self) -> None:
        rec = self._rec("en", "no no no no no no", "no no no no no no")
        self.assertNotIn("loop", alt.issues(rec))

    def test_unstable_clip_flagged_across_requests(self) -> None:
        base = {"lang": "zh", "source": "s", "path": "/a.wav", "ref": "如果我爸妈在家", "ok": True,
                "finish": "stop", "lid": "chinese"}
        records = [{**base, "hyp": "如果我爸妈在家"}, {**base, "hyp": "如果我爸妈不在家"}]
        self.assertEqual(alt.score(records)["zh"]["issues"], {"unstable": 1})


class ConfigParseTest(unittest.TestCase):
    def test_fallback_parser_matches_pyyaml_on_every_v76_config(self) -> None:
        import builtins

        import yaml

        real_import = builtins.__import__

        def no_yaml(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        configs = sorted((SCRIPTS.parent / "configs" / "v7.6").glob("*.yaml"))
        self.assertTrue(configs)
        for cfg in configs:
            data = yaml.safe_load(cfg.read_text()) or {}
            for key in ("train_manifest", "eval_manifest"):
                builtins.__import__ = no_yaml
                try:
                    parsed = alt.config_manifests(str(cfg), key)
                finally:
                    builtins.__import__ = real_import
                expected = {k: list(v) for k, v in (data.get(key) or {}).items()}
                self.assertEqual(parsed, expected, f"{cfg.name}:{key}")


if __name__ == "__main__":
    unittest.main()
