"""data_processing.asr_clean end to end against fake ASR and LLM servers.

Covers what a 176M-record run must survive: duplicate clips across eval and
train, an ASR server answering 503, an LLM that omits items from its JSON or
answers in the wrong script, two workers racing for chunks, and a crash in
the middle of a chunk (resume must neither lose nor duplicate a record).
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
import unittest
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from data_processing.asr_clean.diacritics import check, density, strip_marks
from data_processing.asr_clean.plan import (
    build_plan,
    config_sources,
    data_dir_sources,
    load_dup_mask,
)
from data_processing.asr_clean.prompts import guard, parse_response, system_prompt
from data_processing.asr_clean.worker import OUTPUT_KEYS, WorkerConfig, assemble, run_worker, status

ITEM_RE = re.compile(r"(\d+)\.\n   ORIGINAL: <<<(.*?)>>>\n   ASR:      <<<(.*?)>>>", re.DOTALL)


class _Fakes:
    """One HTTP server playing both the ASR (/v1 with audio) and the LLM (/v1 with text)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.asr_calls = 0
        self.asr_503_left = 3
        self.flaky_seen = False

        fakes = self

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, obj: dict) -> None:
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                self._send(200, {})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                content = body["messages"][-1]["content"]
                if isinstance(content, list):  # ASR request
                    with fakes.lock:
                        fakes.asr_calls += 1
                        if fakes.asr_503_left > 0:
                            fakes.asr_503_left -= 1
                            return self._send(503, {"error": "warming up"})
                    path = Path(content[0]["audio_url"]["url"][len("file://"):])
                    text = path.with_suffix(".asr").read_text(encoding="utf-8")
                    return self._send(200, {"choices": [{"message": {
                        "content": f"language {body['messages'][0]['content']}<asr_text>{text}"},
                        "finish_reason": "stop"}]})
                if "ONLY job is diacritics" in body["messages"][0]["content"]:
                    out = []
                    for i, text in re.findall(r"(\d+)\. <<<(.*?)>>>", content):
                        if "BADDIAC" in text:  # "diacritizes" by changing a letter: must be refused
                            diac = text.replace("جملة", "جملت")
                        else:  # a fatha after the first letter of every Arabic word
                            diac = re.sub(r"(?<![\u0621-\u064a])([\u0621-\u064a])", "\\1\u064e", text)
                        out.append({"i": int(i), "t": diac})
                    return self._send(200, {"choices": [{"message": {"content": json.dumps(out, ensure_ascii=False)},
                                                         "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 10}})
                # LLM request: follow ORIGINAL, add a full stop; misbehave on markers.
                if body["chat_template_kwargs"]["enable_thinking"] and "LONGTHINK" in content:
                    # Reasoning that never reaches the answer (27B node, 8k context).
                    return self._send(200, {"choices": [{"message": {"content": "<think>hmm " * 50},
                                                         "finish_reason": "length"}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 6000}})
                out = []
                for i, org, asr in ITEM_RE.findall(content):
                    i = int(i)
                    if "FLAKY" in org:
                        with fakes.lock:
                            first = not fakes.flaky_seen
                            fakes.flaky_seen = True
                        if first:
                            continue  # omit this item once: must be retried, not dropped
                    if "DROPME" in org:
                        out.append({"i": i, "text": "", "choice": "original", "drop": True})
                    elif "LATIN" in org:
                        out.append({"i": i, "text": "this is english not arabic at all", "choice": "asr"})
                    elif "MUTATE" in org:  # rewrites a word both transcripts agree on
                        out.append({"i": i, "text": org.replace("جملة", "جملتان") + ".", "choice": "merged"})
                    else:
                        out.append({"i": i, "text": org + ".", "choice": "original" if org == asr else "merged"})
                think = "<think>reasoning about the items</think>" if body["chat_template_kwargs"]["enable_thinking"] else ""
                return self._send(200, {"choices": [{"message": {"content": think + json.dumps(out, ensure_ascii=False)},
                                                     "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 10}})

            def log_message(self, *a) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self) -> None:
        self.server.shutdown()


def _wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\0\0" * int(16_000 * seconds))


def _dataset(tmp: Path) -> tuple[Path, Path, list[str]]:
    """eval (10 clips) + train (40 clips, 5 of them the eval clips again)."""
    audio = tmp / "audio"
    audio.mkdir()
    rows_eval, rows_train = [], []
    for i in range(45):
        wav = audio / f"c{i}.wav"
        _wav(wav, 1.0 + (i % 3))
        org = f"جملة رقم {i}"
        if i == 7:
            org = "FLAKY " + org
        if i == 11:
            org = "DROPME " + org
        if i == 13:
            org = "LATIN " + org
        if i == 21:
            org = org + " BADDIAC"
        if i == 19:  # odd: ORIGINAL == ASR, so the format lane; the fake LLM rewrites a word
            org = org + " MUTATE"
        if i == 16:  # disagrees with ASR -> adjudication lane, with thinking
            org = org + " وكلام إضافي طويل بالعربية LONGTHINK"  # Arabic-dominant, or the script guard fires
        asr = org if i % 2 else f"جملة رقمي {i}"  # half agree, half need adjudication
        wav.with_suffix(".asr").write_text(asr, encoding="utf-8")
        row = {"audio_filepath": str(wav), "text": f"language Arabic<asr_text>{org}" if i % 5 == 0 else org}
        if i % 4:
            row["duration"] = 1.0 + (i % 3)  # others: header probe
        (rows_eval if i < 10 else rows_train).append(row)
    rows_train += rows_eval[:5]  # leakage: eval clips also in train
    long_wav = audio / "long.wav"
    _wav(long_wav, 40.0)
    rows_train.append({"audio_filepath": str(long_wav), "text": "طويل جدا", "duration": 40.0})
    rows_train.append({"text": "no audio path"})
    rows_train.append({"audio_filepath": str(audio / "music_only.wav"), "text": "[موسيقى]", "duration": 2.0})
    ev, tr = tmp / "eval_ar_x.jsonl", tmp / "train_ar_x.jsonl"
    ev.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_eval))
    tr.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_train))
    cfg = tmp / "cfg.yaml"
    cfg.write_text(f"eval_manifest:\n  ar:\n    - {ev}\ntrain_manifest:\n  ar:\n    - {tr}\n")
    return cfg, tr, [str(audio / f"c{i}.wav") for i in range(45)]


class PlanTest(unittest.TestCase):
    def test_eval_first_dedup_and_chunking(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            cfg, _, _ = _dataset(tmp)
            sources = config_sources(str(cfg))
            self.assertEqual([s["split"] for s in sources], ["eval", "train"])
            plan = build_plan(str(tmp / "run"), sources, chunk_mb=0, jobs=2)  # 0 MB: one line per chunk
            self.assertEqual(plan["lines"], 10 + 35 + 5 + 3)
            self.assertEqual(plan["duplicates"], 5)
            dup_chunks = [c for c in plan["chunks"] if c["dups"]]
            self.assertTrue(all(c["split"] == "train" for c in dup_chunks))  # eval copies kept
            self.assertTrue(all(load_dup_mask(str(tmp / "run"), c).all() for c in dup_chunks))
            with self.assertRaises(SystemExit):  # a run's plan is frozen
                build_plan(str(tmp / "run"), sources, jobs=1)

    def test_same_file_name_in_two_trees_gets_two_outputs(self) -> None:
        # The real case: training_manifests/v7.6/ar/train_ar_q3asr.jsonl and
        # q3asr_sft_manifests/ar/train_ar_q3asr.jsonl are different data.
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            srcs = []
            for tree in ("training_manifests/v7.6", "q3asr_sft_manifests"):
                d = tmp / tree / "ar"
                d.mkdir(parents=True)
                (d / "train_ar_q3asr.jsonl").write_text(json.dumps({"audio_filepath": f"/{tree}.wav", "text": "x"}) + "\n")
                srcs.append({"lang": "ar", "split": "train", "path": str(d / "train_ar_q3asr.jsonl")})
            plan = build_plan(str(tmp / "run"), srcs, jobs=1)
            self.assertEqual(sorted(c["stem"] for c in plan["chunks"]),
                             ["q3asr_sft_manifests/train_ar_q3asr", "v7.6/train_ar_q3asr"])
            self.assertEqual(len({c["id"] for c in plan["chunks"]}), 2)

    def test_data_dir_sources_from_the_real_file_names(self) -> None:
        # `ls data/` on the cluster, 2026-10-09.
        names = ["eval_ar_inworld", "train_zh_q3asr", "eval_ar_ar_ae", "eval_en_inworld", "eval_en_expresso", "eval_ml_inworld_2", "eval_zh_inworld", "train_hi_q3asr", "train_ml_itn_punct", "train_en_q3asr", "train_en_commentary", "train_ar_q3asr", "train_ar_ar_ae", "train_ar_camel_race", "train_en_inworld", "train_ar_inworld_full", "train_ar_qudratech_batch2", "train_ar_qudratech_phase4", "train_ar_khalid_msa", "train_ar_el_gen_v5", "train_ar_el_ar_v1", "train_ar_se_v1", "train_ar_spotify_v1", "train_ar_dialect_gulf", "train_ar_ar_ae_train", "train_en_expresso", "train_en_anispeech", "train_en_hifi_tts", "eval_ml_inworld", "eval_hi_inworld"]
        with tempfile.TemporaryDirectory() as t:
            data = Path(t) / "data"
            data.mkdir()
            for n in names:
                (data / f"{n}.json").write_text(json.dumps({"audio_filepath": f"/{n}.wav", "text": "x"}) + "\n")
            srcs = data_dir_sources(str(data))
            self.assertEqual(len(srcs), 30)
            evals = [s_ for s_ in srcs if s_["split"] == "eval"]
            self.assertEqual(srcs[:len(evals)], evals)  # eval first: the dedup priority
            langs = {}
            for s_ in srcs:
                langs[s_["lang"]] = langs.get(s_["lang"], 0) + 1
            self.assertEqual(langs, {"ar": 15, "en": 8, "hi": 2, "ml": 3, "zh": 2})
            plan = build_plan(str(Path(t) / "run"), srcs, jobs=1)
            self.assertEqual(len({c["stem"] for c in plan["chunks"]}), 30)
            self.assertTrue(all(c["stem"].startswith("data/") for c in plan["chunks"]))

    def test_json_array_files_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            data = Path(t) / "data"
            data.mkdir()
            (data / "train_ar_x.json").write_text('[{"audio_filepath": "/a.wav", "text": "x"}]')
            with self.assertRaises(SystemExit) as ctx:
                build_plan(str(Path(t) / "run"), data_dir_sources(str(data)), jobs=1)
            self.assertIn("JSON Lines", str(ctx.exception))
            (data / "notes_x.json").write_text("{}\n")
            with self.assertRaises(SystemExit):
                data_dir_sources(str(data))

    def test_limit_takes_the_first_n_records_of_every_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            cfg, _, _ = _dataset(tmp)
            plan = build_plan(str(tmp / "run"), config_sources(str(cfg)), chunk_mb=0, jobs=1, limit=3)
            self.assertEqual(plan["lines"], 6)  # 3 eval + 3 train
            self.assertEqual(plan["duplicates"], 0)
            self.assertAlmostEqual(plan["estimated_full_lines"], 53, delta=8)  # 10 eval + 43 train


class EndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fakes = _Fakes()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmpdir.name)
        cfg, _, self.paths = _dataset(self.tmp)
        self.root = str(self.tmp / "run")
        build_plan(self.root, config_sources(str(cfg)), chunk_mb=1, jobs=2)

    def tearDown(self) -> None:
        self.fakes.close()
        self.tmpdir.cleanup()

    def _cfg(self, wid: str, **kw) -> WorkerConfig:
        return WorkerConfig(run_root=self.root, asr_urls=[self.fakes.url], llm_url=self.fakes.url,
                            worker_id=wid, flush_every=4, asr_concurrency=4, llm_concurrency=3,
                            format_batch=3, adjudicate_batch=2, thinking="auto", **kw)

    def _outputs(self) -> tuple[list[dict], list[dict]]:
        out = [json.loads(line) for p in sorted((Path(self.root) / "ar").rglob("part-*.jsonl"))
               for line in p.read_text(encoding="utf-8").splitlines()]
        rej = [json.loads(line) for p in sorted((Path(self.root) / "_rejects").rglob("*.jsonl"))
               for line in p.read_text(encoding="utf-8").splitlines()]
        return out, rej

    def _check_complete(self, resumed: bool = False) -> None:
        out, rej = self._outputs()
        for r in out:
            self.assertEqual(tuple(r), OUTPUT_KEYS)  # exactly the 5 columns, in order
            self.assertIsInstance(r["duration"], float)
            self.assertNotIn("<asr_text>", r["org_text"])  # q3asr envelope stripped
            plain = strip_marks(r["text"])
            if "MUTATE" in r["org_text"]:  # agree guard: the rewrite is overruled
                self.assertEqual(plain, r["org_text"])
            else:
                self.assertEqual(plain, r["org_text"] + ".")  # diacritics never change a letter
            if "BADDIAC" in r["org_text"]:
                self.assertEqual(r["text"], plain)  # failed check -> kept bare
            else:
                self.assertGreater(density(r["text"]), 0)  # every other Arabic record got marks
        written = [r["audio_filepath"] for r in out]
        self.assertEqual(len(written), len(set(written)))  # exactly once
        self.assertEqual(set(written), set(self.paths) - {self.paths[11], self.paths[13]})
        reasons = sorted(r["reason"] for r in rej)
        self.assertEqual(reasons, ["bad_record", "duration_out_of_range", "guard_wrong_script", "llm_drop",
                                   "non_speech"])
        st = status(self.root)
        self.assertEqual(st["chunks_done"], st["chunks"])
        self.assertEqual(st["written"], 43)
        self.assertEqual(st["processed"], st["to_clean"])
        self.assertEqual(st["lanes"]["format"] + st["lanes"]["adjudicate"], 43)
        self.assertEqual(st["agree_guard"], 1)
        self.assertEqual(st["arabic_diacritics"]["diacritized"], 42)
        self.assertEqual(st["arabic_diacritics"]["failed"], {"letters_changed": 1})
        self.assertGreater(st["arabic_diacritics"]["records_with_marks_pct"], 95)
        tel = st["telemetry"]
        self.assertGreaterEqual(tel["think_fallback_items"], 1)  # LONGTHINK recovered without thinking
        self.assertGreater(tel["llm"]["adjudicate_think"]["hit_max_tokens_pct"], 0)
        self.assertIn("adjudicate", tel["llm"])  # the non-thinking retry
        # Every clip that reached ASR (incl. drop/latin); a crash redoes its in-flight windows.
        (self.assertGreaterEqual if resumed else self.assertEqual)(tel["asr_requests"], 43 + 2)

    def test_two_racing_workers_clean_everything_exactly_once(self) -> None:
        threads = [threading.Thread(target=run_worker, args=(self._cfg(f"h:{i}"),)) for i in range(2)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self._check_complete()
        self.assertTrue(self.fakes.flaky_seen)
        merged = assemble(self.root)
        self.assertEqual(len(merged), 2)
        finals = sorted(p for p in (Path(self.root) / "ar").rglob("*.jsonl") if not p.name.startswith("part-"))
        self.assertEqual([p.name for p in finals], ["eval_ar_x.jsonl", "train_ar_x.jsonl"])
        self.assertEqual(sum(len(p.read_text().splitlines()) for p in finals), 43)

    def test_overrules_are_logged_and_the_guard_can_be_switched_off(self) -> None:
        run_worker(self._cfg("h:0"))
        review = [json.loads(line) for p in (Path(self.root) / "_review").rglob("*.jsonl")
                  for line in p.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(review), 1)
        self.assertIn("جملتان", review[0]["llm_text"])  # the rewrite that was NOT written
        self.assertEqual(review[0]["kept"], "org_text")

    def test_guard_off_writes_the_llm_version(self) -> None:
        run_worker(self._cfg("h:0", agree_max_added=-1))
        out, _ = self._outputs()
        mutated = [r for r in out if "MUTATE" in r["org_text"]]
        self.assertEqual(len(mutated), 1)
        self.assertIn("جملتان", strip_marks(mutated[0]["text"]))

    def test_crash_mid_chunk_then_resume(self) -> None:
        run_worker(self._cfg("h:0"), stop_after_windows=2)  # dies after 2 commits
        part = next((Path(self.root) / "ar").rglob("part-*.jsonl"))
        with open(part, "ab") as fh:  # torn write after the last commit
            fh.write(b'{"audio_filepath": "half a rec')
        run_worker(self._cfg("h:0"))  # same id resumes its own chunk
        self._check_complete(resumed=True)


class DiacriticsTest(unittest.TestCase):
    def test_marks_only_edit_passes(self) -> None:
        self.assertIsNone(check("مدرسة كبيرة، 3 طلاب.", "مَدْرَسَةٌ كَبِيرَةٌ، 3 طُلَّابٍ."))
        self.assertIsNone(check("ما أرفض شي", "مَا أَرْفُض شِي"))  # dialect, partial marks
        self.assertIsNone(check("بِيِعْمِل", "بيعمل"))  # removing marks is a marks-only edit too

    def test_letter_changes_are_refused(self) -> None:
        self.assertEqual(check("بيعمل", "يَعْمَل"), "letters_changed")  # dialect -> MSA
        self.assertEqual(check("مدرسه", "مَدْرَسَة"), "letters_changed")  # ha -> ta marbuta "fix"
        self.assertEqual(check("اسم", "إِسْم"), "letters_changed")  # hamza seat "fix"
        self.assertEqual(check("قال كذا", "قَالَ كَذَا."), "letters_changed")  # punctuation added

    def test_misplaced_or_stacked_marks_are_refused(self) -> None:
        self.assertEqual(check("رقم 3", "رَقَمٌ 3\u064e"), "mark_not_on_letter")
        self.assertEqual(check("في campus", "فِي c\u064eampus"), "mark_not_on_letter")
        self.assertEqual(check("علم", "عَ\u0651\u064e\u064bلم"), "stacked_marks")

    def test_shadda_and_tanween_are_never_dropped(self) -> None:
        self.assertEqual(check("مرّة", "مرة"), "dropped_shadda_or_tanween")
        self.assertEqual(check("شكرًا", "شكرا"), "dropped_shadda_or_tanween")
        self.assertIsNone(check("شكرًا", "شُكْرًا"))  # adding around them is fine
        self.assertIsNone(check("بِيِعْمِل", "بيعمل"))  # plain vowels may still be removed

    def test_original_marks_carried_onto_unchanged_words_only(self) -> None:
        from data_processing.asr_clean.diacritics import transfer_marks

        org = "شكرًا يا أستاذ، المدرّسة كبيرةٌ جدًا"
        text = "شكرا يا استاذ، المدرسة كبيرة جدا."
        out = transfer_marks(org, text)
        self.assertEqual(out, "شكرًا يا استاذ، المدرّسة كبيرةٌ جدًا.")
        self.assertIsNone(check(text, out))  # still a marks-only edit
        self.assertEqual(transfer_marks("قال كذا", "قال كذا."), "قال كذا.")  # nothing to carry

    def test_density(self) -> None:
        self.assertEqual(density("قال"), 0.0)
        self.assertAlmostEqual(density("قَالَ"), 2 / 3)
        self.assertEqual(density("hello 123"), 0.0)

    def test_policies(self) -> None:
        from data_processing.asr_clean.diacritics import system_prompt as dsp

        self.assertIn("CRITICAL", dsp("critical"))
        self.assertIn("FULL", dsp("full"))
        self.assertIn("Never change, add, remove or reorder a letter", dsp("full"))


class ConventionsTest(unittest.TestCase):
    def test_arabic_canon(self) -> None:
        from data_processing.asr_clean.conventions import canonicalize as c

        self.assertEqual(c("قال شكراً، الخصم 50% بس ١٠ ريال", "ar"), "قال شكرًا، الخصم 50٪ بس 10 ريال")
        self.assertEqual(c("كبيـــرة جداً", "ar"), "كبيرة جدًا")

    def test_indic_canon(self) -> None:
        from data_processing.asr_clean.conventions import canonicalize as c

        self.assertEqual(c("यह है | अब १२ बजे", "hi"), "यह है। अब 12 बजे")
        # nta: the Unicode 5.1+ recommended chillu-n form; legacy ZWJ chillu -> atomic
        self.assertEqual(c("ഗവൺമെന്റിന്റെ ന്\u200d", "ml"), "ഗവൺമെൻ്റിൻ്റെ ൻ")

    def test_english_case_and_meridiem(self) -> None:
        from data_processing.asr_clean.conventions import canonicalize as c

        cases = {
            "the meeting is at 8 p.m. and i'm late": "The meeting is at 8 PM and I'm late",
            "we met at 10 am. she left": "We met at 10 AM. She left",
            "we met at 10 a.m. She left": "We met at 10 AM. She left",  # the dotted form's stop survives
            "Call at 3P.M.": "Call at 3 PM.",
            "I am fine. i was there": "I am fine. I was there",  # "am" the verb is untouched
            "Visit iPhone store. the end": "Visit iPhone store. The end",
        }
        for src, want in cases.items():
            self.assertEqual(c(src, "en"), want)

    def test_meta_prefix_from_earlier_llm_pass(self) -> None:
        from data_processing.asr_clean.conventions import strip_non_speech

        self.assertEqual(strip_non_speech("Corrected Transcript: Horizontal business"), "Horizontal business")
        self.assertEqual(strip_non_speech("the transcript: shows it"), "the transcript: shows it")

    def test_non_speech_tags_only(self) -> None:
        from data_processing.asr_clean.conventions import strip_non_speech

        self.assertEqual(strip_non_speech("[موسيقى]"), "")
        self.assertEqual(strip_non_speech("hello [Laughter] there <unk>"), "hello there")
        self.assertEqual(strip_non_speech("the interval [a, b] is closed"), "the interval [a, b] is closed")

    def test_comparison_folds(self) -> None:
        from data_processing.asr_clean.text import added_letters, normalize

        self.assertEqual(normalize("لصالحة على", "ar"), normalize("لصالحه علي", "ar"))  # spelling fixes
        self.assertEqual(added_letters("يعطية لصالحة", "يعطيه لصالحه", "ar"), 0)
        self.assertEqual(normalize("ന്\u200dറ", "ml").count(" "), 0)  # ZWJ never splits a word
        self.assertEqual(normalize("ഗവൺമെന്റ്", "ml"), normalize("ഗവൺമെൻ്റ്", "ml"))  # one nta

    def test_itn_specs_cover_the_hard_cases(self) -> None:
        from data_processing.asr_clean.conventions import ITN

        must = {"ar": ["٪", "ريال", "ملايين", "ألف عافية", "واحد صاحبي", "Hijri", "Quran", "21/10", "100 دراهم",
                       "15 رمضان"],
                "en": ["$50", "21st", "1990", "World War II", "no one", "10,000", "£30,000", "50 rupees",
                       "March 21st", "31 July", "AM", "info@example.com", "USA", "-5"],
                "zh": ["2021年", "10万", "三个计划", "第一", "一心一意"],
                "hi": ["2 लाख", "4:15", "4:45", "एक आदमी", "रुपये"],
                "ml": ["2 ലക്ഷം", "ആയിരം", "ഒരു ദിവസം", "രൂപ"]}
        for lang, needles in must.items():
            for needle in needles:
                self.assertIn(needle, ITN[lang], f"{lang}: {needle}")
        self.assertNotIn("\u0d05\u0d3e", ITN["ml"])  # the malformed "അായിരം" example
        from data_processing.asr_clean.conventions import conventions

        self.assertIn("sentence case", conventions("en"))
        self.assertIn("Mr.", conventions("en"))
        self.assertIn("《》", conventions("zh"))
        self.assertIn("never $ or ¥", conventions("zh"))
        for lang in ("ar", "hi", "ml", "zh"):
            self.assertIn("acronyms capitalized", conventions(lang), lang)


class PromptTest(unittest.TestCase):
    def test_parse_is_strict_and_0_based(self) -> None:
        content = '```json\n[{"i": 0, "text": "a"}, {"i": 0, "text": "dup"}, {"i": 5, "text": "x"}, {"i": 1}]\n```'
        self.assertEqual(parse_response(content, 2), {0: {"text": "a", "choice": "unknown", "drop": False}})
        self.assertEqual(parse_response("not json", 2), {})

    def test_parse_compact_keys(self) -> None:
        content = '[{"i":0,"t":"مرحبا.","s":"a"},{"i":1,"d":1},{"i":2,"t":"x","s":"zz"}]'
        self.assertEqual(parse_response(content, 3), {
            0: {"text": "مرحبا.", "choice": "asr", "drop": False},
            1: {"text": "", "choice": "unknown", "drop": True},
            2: {"text": "x", "choice": "unknown", "drop": False}})

    def test_asr_cap_is_configurable(self) -> None:
        self.assertEqual(WorkerConfig(run_root="x", asr_urls=["u"]).asr_max_tokens, 1024)

    def test_guards(self) -> None:
        self.assertEqual(guard("this is english", "مرحبا بكم جميعا", "مرحبا بكم", "ar"), "wrong_script")
        self.assertEqual(guard("<<<مرحبا>>>", "مرحبا", "مرحبا", "ar"), "prompt_leak")
        self.assertEqual(guard("كلام مختلف تماما عن المدخلات", "مرحبا بكم", "مرحبا بكم", "ar"), "divergent")
        self.assertIsNone(guard("مرحبًا بكم.", "مرحبا بكم", "مرحبا بكو", "ar"))
        # vowel signs lost relative to BOTH inputs (test3, Hindi)
        hi = "उन्होंने बताया, ये बच्चा बरगुना की अदालत के निर्देश से, 22 दिसंबर को हमारे यहाँ लाया गया।"
        self.assertEqual(guard("उन्होंन बताय, ये बच्च बरगुन क अदालत क निर्देश स, 22 दिसंबर क हमार यहाँ लाय गय।",
                               hi, hi, "hi"), "stripped_marks")
        self.assertIsNone(guard(hi, hi, hi, "hi"))
        # ...and relative to the better input when the ASR itself is stripped (load test, IMaSC voices)
        self.assertEqual(guard("ജന ജയഹനദ തടങങയ", "ജനം, ജയ്ഹിന്ദ് തുടങ്ങിയ", "ജന ജയഹനദ തടങങയ", "ml"),
                         "stripped_marks")

    def test_code_switched_chinese_is_not_wrong_script(self) -> None:
        # test1 rejected 72 of these: more Latin letters than Han characters, legitimately.
        zh = "听起来很有趣！你怎么学data analysis的？"
        self.assertIsNone(guard(zh, zh, zh, "zh"))
        self.assertEqual(guard("Punpao tian ming ah wang sheng", "混滔天明了皇上又跟进", "混滔天明了皇上", "zh"),
                         "wrong_script")

    def test_added_letters_allows_formatting_and_itn_not_rewrites(self) -> None:
        from data_processing.asr_clean.text import added_letters

        # formatting, diacritics, ITN: free
        self.assertEqual(added_letters("ست سنوات بعد سنتين", "6 سنوات بعد سنتين.", "ar"), 0)
        self.assertEqual(added_letters("مرحبا بكم", "مَرْحَبًا بِكُم،", "ar"), 0)
        self.assertEqual(added_letters("My uncle is thirty years", "My uncle is 30 years.", "en"), 0)
        self.assertEqual(added_letters("നിങ്ങൾ കണ്ടു പിടിച്ചോ", "നിങ്ങൾ കണ്ടുപിടിച്ചോ?", "ml"), 0)
        # rewrites seen in test1: plural added, verse changed, word replaced
        self.assertGreater(added_letters("ചില പ്ലേറ്റ് കുറച്ച്", "ചില പ്ലേറ്റുകൾ കുറച്ച്", "ml"), 1)
        self.assertGreater(added_letters("فَوَرَبِّ السَّمَاءِ وَالْأَرْضِ", "فَوَرَبِّ السَّماواتِ وَالْأَرْضِ", "ar"), 1)
        self.assertGreater(added_letters("പരമ്പരാഗതമായിട്ടുള്ള വസ്ത്രങ്ങളും", "പരസ്പരം ആഗതമായുള്ള വസ്ത്രങ്ങളും", "ml"), 1)

    def test_format_lane_prompt_freezes_the_words(self) -> None:
        from data_processing.asr_clean.prompts import user_prompt

        items = [{"org_text": "a", "asr_text": "a"}]
        self.assertIn("words are certain", user_prompt(items, "ml", "format"))
        self.assertNotIn("words are certain", user_prompt(items, "ml", "adjudicate"))

    def test_arabic_prompt_carries_corpus_conventions(self) -> None:
        sp = system_prompt("ar")
        for needle in ("Western digits", "ORIGINAL", "ASR", "DIALECT PRESERVATION", "DIACRITICS", '[{"i":0,"t":'):
            self.assertIn(needle, sp)
        self.assertNotIn("{{", sp)  # the output example must be literal JSON, not a format template
        self.assertIn("an Arabic", sp)
        self.assertIn("matra", system_prompt("ml"))


if __name__ == "__main__":
    unittest.main()
