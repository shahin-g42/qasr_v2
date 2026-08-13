"""Tests for scripts/prepare_q3asr_filter.py.

Covers envelope parsing, language normalization, script-heuristic language
identification, the vLLM identification path, and the end-to-end transform
into per-language training manifests.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_q3asr_filter.py"

spec = importlib.util.spec_from_file_location("prepare_q3asr_filter", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules["prepare_q3asr_filter"] = mod  # dataclasses introspects sys.modules
spec.loader.exec_module(mod)

MERGE_PATH = REPO_ROOT / "scripts" / "merge_q3asr_shards.py"
_merge_spec = importlib.util.spec_from_file_location("merge_q3asr_shards", MERGE_PATH)
merge_mod = importlib.util.module_from_spec(_merge_spec)
sys.modules["merge_q3asr_shards"] = merge_mod
_merge_spec.loader.exec_module(merge_mod)


class ShardRecordsTest(unittest.TestCase):
    """Sharding must lose nothing, duplicate nothing, and stay stable."""

    NUM_NODES = 8

    def _corpus(self, n: int = 500) -> list[mod.SourceRecord]:
        return [
            mod.SourceRecord(
                split="train" if i % 3 else "eval",
                audio=f"data/asr/{i}.wav",
                language="en",
                text=f"utterance {i}",
            )
            for i in range(n)
        ]

    def test_partition_is_complete_and_disjoint(self) -> None:
        records = self._corpus()
        seen: list[str] = []
        for rank in range(self.NUM_NODES):
            seen.extend(r.audio for r in mod.shard_records(records, rank, self.NUM_NODES))
        # Every record claimed exactly once: no gaps, no double-processing.
        self.assertEqual(len(seen), len(records))
        self.assertEqual(set(seen), {r.audio for r in records})

    def test_every_node_gets_work(self) -> None:
        records = self._corpus()
        sizes = [
            len(mod.shard_records(records, rank, self.NUM_NODES))
            for rank in range(self.NUM_NODES)
        ]
        self.assertTrue(all(sizes), f"some node idle: {sizes}")

    def test_same_audio_always_lands_on_one_node(self) -> None:
        """Why we hash the audio path instead of round-robining line indices.

        The dedup and train/eval overlap checks only see one shard, so both
        copies of a repeated audio file must be on the same node or those
        checks silently pass on data they never compared.
        """
        shared = "data/asr/shared.wav"
        records = [
            mod.SourceRecord(split="train", audio=shared, language="en", text="a"),
            mod.SourceRecord(split="eval", audio=shared, language="en", text="b"),
            mod.SourceRecord(split="train", audio=shared, language="en", text="c"),
        ]
        owners = [
            rank
            for rank in range(self.NUM_NODES)
            if mod.shard_records(records, rank, self.NUM_NODES)
        ]
        self.assertEqual(len(owners), 1)
        self.assertEqual(len(mod.shard_records(records, owners[0], self.NUM_NODES)), 3)

    def test_partition_is_stable_across_processes(self) -> None:
        """Guards against regressing to the builtin hash().

        PYTHONHASHSEED randomises str hashing per process, so hash() would give
        each node a different partition function — records both duplicated and
        dropped, with nothing in the logs to show it.
        """
        import subprocess

        code = (
            "import importlib.util, sys;"
            f"s=importlib.util.spec_from_file_location('m', r'{SCRIPT_PATH}');"
            "m=importlib.util.module_from_spec(s);sys.modules['m']=m;"
            "s.loader.exec_module(m);"
            "rs=[m.SourceRecord(split='train',audio=f'a/{i}.wav',language='en',"
            "text='t') for i in range(200)];"
            "print(','.join(r.audio for r in m.shard_records(rs,0,8)))"
        )
        outs = set()
        for seed in ("0", "1", "12345"):
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env={"PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT / "src")},
                check=True,
            )
            outs.add(result.stdout.strip())
        self.assertEqual(len(outs), 1, "partition changed with PYTHONHASHSEED")

    def test_single_node_is_identity(self) -> None:
        records = self._corpus(10)
        self.assertIs(mod.shard_records(records, 0, 1), records)


def _write_input(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


class ParseTextEnvelopeTest(unittest.TestCase):
    def test_named_language(self) -> None:
        lang, text = mod.parse_text_envelope(
            "language Arabic<asr_text>التي تعد مصدر دخلهم الوحيد."
        )
        self.assertEqual(lang, "Arabic")
        self.assertEqual(text, "التي تعد مصدر دخلهم الوحيد.")

    def test_none_language(self) -> None:
        lang, text = mod.parse_text_envelope(
            "language None<asr_text>Our little bushel just lost one apple."
        )
        self.assertEqual(lang, "None")
        self.assertEqual(text, "Our little bushel just lost one apple.")

    def test_marker_inside_transcript_kept(self) -> None:
        lang, text = mod.parse_text_envelope(
            "language English<asr_text>say <asr_text> twice"
        )
        self.assertEqual(lang, "English")
        self.assertEqual(text, "say <asr_text> twice")

    def test_missing_envelope(self) -> None:
        self.assertIsNone(mod.parse_text_envelope("just a plain transcript"))

    def test_empty_transcript(self) -> None:
        self.assertIsNone(mod.parse_text_envelope("language Arabic<asr_text>  "))


class NormalizeLanguageValueTest(unittest.TestCase):
    def test_names(self) -> None:
        self.assertEqual(mod.normalize_language_value("Arabic"), "ar")
        self.assertEqual(mod.normalize_language_value("English"), "en")
        self.assertEqual(mod.normalize_language_value("Hindi"), "hi")
        self.assertEqual(mod.normalize_language_value("Malayalam"), "ml")
        self.assertEqual(mod.normalize_language_value("Chinese"), "zh")

    def test_codes(self) -> None:
        self.assertEqual(mod.normalize_language_value("ar"), "ar")
        self.assertEqual(mod.normalize_language_value("EN"), "en")

    def test_none_variants(self) -> None:
        self.assertIsNone(mod.normalize_language_value("None"))
        self.assertIsNone(mod.normalize_language_value("none"))
        self.assertIsNone(mod.normalize_language_value(""))
        self.assertIsNone(mod.normalize_language_value("Klingon"))


class SplitFromFilenameTest(unittest.TestCase):
    def test_splits(self) -> None:
        self.assertEqual(mod.split_from_filename(Path("train_filter.jsonl")), "train")
        self.assertEqual(mod.split_from_filename(Path("eval_filter.jsonl")), "eval")

    def test_unknown_defaults_to_train(self) -> None:
        self.assertEqual(mod.split_from_filename(Path("misc.jsonl")), "train")


class HeuristicLanguageTest(unittest.TestCase):
    def test_scripts(self) -> None:
        self.assertEqual(mod.heuristic_language("مرحبا بالعالم"), "ar")
        self.assertEqual(mod.heuristic_language("नमस्ते दुनिया"), "hi")
        self.assertEqual(mod.heuristic_language("നമസ്കാരം"), "ml")
        self.assertEqual(mod.heuristic_language("你好世界"), "zh")

    def test_latin_needs_llm(self) -> None:
        self.assertIsNone(mod.heuristic_language("Looks like it."))

    def test_codeswitched_dominant_script_wins(self) -> None:
        self.assertEqual(
            mod.heuristic_language("لا، يا عزيزي، أقصد الألم، Pain."), "ar"
        )
        # Chinese carrying heavy Latin is still Chinese (script share ~0.46)
        self.assertEqual(mod.heuristic_language("我们的 meeting 在 3:30 开始"), "zh")

    def test_english_quoting_a_foreign_word_is_not_stolen(self) -> None:
        """A minority script must not decide the language.

        Presence-based detection routed all three of these to the quoted
        script, which would bake a wrong ``language X`` marker into the
        training target. They are English and belong to the LLM pass.
        """
        self.assertIsNone(mod.heuristic_language("The word for pain is ألم in Arabic"))
        self.assertIsNone(mod.heuristic_language("The Hindi word नमस्ते means hello"))
        self.assertIsNone(mod.heuristic_language("She said വണക്കം to greet us"))
        self.assertIsNone(
            mod.heuristic_language("The character 中 means middle in Chinese texts")
        )

    def test_absolute_floor_beats_share(self) -> None:
        """Very short text can pass the share test on 1-2 stray characters."""
        self.assertLess(len("中"), mod.MIN_SCRIPT_CHARS)
        self.assertIsNone(mod.heuristic_language("中"))
        self.assertIsNone(mod.heuristic_language("ok 中"))

    def test_no_alphabetic_content(self) -> None:
        self.assertIsNone(mod.heuristic_language(""))
        self.assertIsNone(mod.heuristic_language("123 -- 45.6 !!"))


class ResolveUnknownLanguagesTest(unittest.TestCase):
    def _record(self, text: str) -> mod.SourceRecord:
        return mod.SourceRecord(split="train", audio="a.wav", language=None, text=text)

    def test_heuristic_short_circuits_llm(self) -> None:
        record = self._record("مرحبا بالعالم")
        with mock.patch.object(mod, "VLLMClient") as client_cls:
            resolved = asyncio.run(
                mod.resolve_unknown_languages([record], config=mock.Mock(), concurrency=4)
            )
        self.assertEqual(record.language, "ar")
        self.assertEqual(resolved, 0)
        client_cls.assert_not_called()

    def test_llm_resolves_latin_text(self) -> None:
        record = self._record("Looks like it.")

        class FakeClient:
            def __init__(self, config: object, **kwargs: object) -> None:
                pass

            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def wait_for_health(self, max_wait_seconds: float = 300) -> None:
                pass

            async def chat_completion_json(
                self, messages: list[dict], **kwargs: object
            ) -> dict:
                return {"language": "en"}

        with mock.patch.object(mod, "VLLMClient", FakeClient):
            resolved = asyncio.run(
                mod.resolve_unknown_languages([record], config=mock.Mock(), concurrency=4)
            )
        self.assertEqual(record.language, "en")
        self.assertEqual(resolved, 1)

    def test_llm_failure_falls_back_then_unknown(self) -> None:
        record = self._record("Looks like it.")

        class BrokenClient:
            def __init__(self, config: object, **kwargs: object) -> None:
                pass

            async def __aenter__(self) -> BrokenClient:
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def wait_for_health(self, max_wait_seconds: float = 300) -> None:
                pass

            async def chat_completion_json(
                self, messages: list[dict], **kwargs: object
            ) -> dict:
                raise RuntimeError("server down")

        with mock.patch.object(mod, "VLLMClient", BrokenClient):
            resolved = asyncio.run(
                mod.resolve_unknown_languages([record], config=mock.Mock(), concurrency=4)
            )
        self.assertIsNone(record.language)
        self.assertEqual(resolved, 0)


class LabelAuditTest(unittest.TestCase):
    """The source labels are not authoritative — see find_label_contradictions."""

    # Real shape of the mislabels found in v4.0/en/train_en_q3asr.jsonl:
    # Mandarin carrying English loanwords, filed as English.
    MISLABELLED = "我会记住这个。谢谢分享你的insights。"

    def _record(self, language: str, text: str) -> mod.SourceRecord:
        return mod.SourceRecord(
            split="train", audio="a.wav", language=language, text=text
        )

    def _client(self, verdict: str | None):
        class Client:
            # open_lid_client passes rate_limit/rate_capacity, so accept kwargs.
            def __init__(self, config: object, **kwargs: object) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc: object) -> bool:
                return False

            async def wait_for_health(self, max_wait_seconds: float = 300) -> None:
                pass

            async def chat_completion_json(self, messages: list[dict], **kw: object):
                return {"language": verdict}

        return Client

    def test_finds_contradiction(self) -> None:
        good = self._record("zh", self.MISLABELLED)
        bad = self._record("en", self.MISLABELLED)
        found = mod.find_label_contradictions([good, bad])
        self.assertEqual(found, [(bad, "zh")])

    def test_agreeing_labels_are_not_questioned(self) -> None:
        records = [
            self._record("ar", "مرحبا بالعالم"),
            self._record("en", "Plain English here."),  # script defers, no clash
        ]
        self.assertEqual(mod.find_label_contradictions(records), [])

    def test_consensus_relabels(self) -> None:
        record = self._record("en", self.MISLABELLED)
        with mock.patch.object(mod, "VLLMClient", self._client("zh")):
            stats = asyncio.run(
                mod.audit_language_labels([record], mock.Mock(), concurrency=4)
            )
        self.assertEqual(record.language, "zh")
        self.assertEqual(stats["relabelled"], 1)
        self.assertIn("relabelled en->zh", record.audit)

    def test_llm_backing_the_label_keeps_it(self) -> None:
        record = self._record("en", self.MISLABELLED)
        with mock.patch.object(mod, "VLLMClient", self._client("en")):
            stats = asyncio.run(
                mod.audit_language_labels([record], mock.Mock(), concurrency=4)
            )
        self.assertEqual(record.language, "en")  # never overruled alone
        self.assertEqual(stats["kept"], 1)

    def test_three_way_disagreement_is_disputed_not_relabelled(self) -> None:
        record = self._record("en", self.MISLABELLED)
        with mock.patch.object(mod, "VLLMClient", self._client("hi")):
            stats = asyncio.run(
                mod.audit_language_labels([record], mock.Mock(), concurrency=4)
            )
        self.assertEqual(record.language, "en")  # conservative: keep the label
        self.assertEqual(stats["disputed"], 1)
        self.assertIn("disputed", record.audit)


class TransformTest(unittest.TestCase):
    def test_end_to_end(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [
                    {
                        "audio": "data/asr/a.wav",
                        "text": "language Arabic<asr_text>وأقف عندها.",
                    },
                    {
                        "audio": "data/asr/b.wav",
                        "text": "language English<asr_text>That is the difference.",
                    },
                    {
                        "audio": "data/asr/c.wav",
                        "text": "language None<asr_text>نحرره من الآخرين.",
                    },
                    {
                        "audio": "data/asr/d.wav",
                        "text": "no envelope here",
                    },
                ],
            )
            eval_in = _write_input(
                tmp_path / "eval_filter.jsonl",
                [
                    {
                        "audio": "data/asr/e.wav",
                        "text": "language Hindi<asr_text>नमस्ते दोस्तों",
                    },
                ],
            )

            out_dir = tmp_path / "out"
            fake_durations = {
                str(tmp_path / "data/asr/a.wav"): 1.0,
                str(tmp_path / "data/asr/b.wav"): 2.0,
                str(tmp_path / "data/asr/c.wav"): 3.0,
                str(tmp_path / "data/asr/e.wav"): 4.0,
            }

            def fake_probe(path: str) -> float | None:
                return fake_durations.get(path)

            from data_processing.config import PipelineConfig

            with mock.patch.object(mod, "probe_duration", fake_probe):
                stats = mod.transform(
                    inputs=[str(train_in), str(eval_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                )

            # Per-language train/eval manifests in the training schema.
            # Filenames embed the language so they stay self-describing when
            # copied into training_manifests/ (train_ar_q3asr.json convention).
            ar_train = out_dir / "ar" / "train_ar_q3asr.jsonl"
            en_train = out_dir / "en" / "train_en_q3asr.jsonl"
            hi_eval = out_dir / "hi" / "eval_hi_q3asr.jsonl"
            self.assertEqual(stats[str(ar_train)], 2)  # explicit + None->ar
            self.assertEqual(stats[str(en_train)], 1)
            self.assertEqual(stats[str(hi_eval)], 1)

            ar_records = [
                json.loads(line) for line in ar_train.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                ar_records[0],
                {
                    "audio_filepath": str(tmp_path / "data/asr/a.wav"),
                    "text": "وأقف عندها.",
                    "duration": 1.0,
                },
            )
            # The "language None" record was resolved to Arabic by heuristics
            self.assertEqual(ar_records[1]["text"], "نحرره من الآخرين.")

            # No rejected piles expected
            self.assertFalse((out_dir / "rejected_no_duration.jsonl").exists())
            self.assertFalse((out_dir / "rejected_unknown_language.jsonl").exists())

    def test_missing_duration_and_unknown_language_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [
                    {
                        "audio": "data/asr/good.wav",
                        "text": "language English<asr_text>Works fine.",
                    },
                    {
                        "audio": "data/asr/broken.wav",
                        "text": "language English<asr_text>Header unreadable.",
                    },
                    {
                        "audio": "data/asr/mystery.wav",
                        "text": "language None<asr_text>batle tobi se naode",
                    },
                ],
            )
            out_dir = tmp_path / "out"

            def fake_probe(path: str) -> float | None:
                return 1.5 if path.endswith("good.wav") else None

            from data_processing.config import PipelineConfig

            class NoopClient:
                def __init__(self, config: object, **kwargs: object) -> None:
                    pass

                async def __aenter__(self) -> NoopClient:
                    return self

                async def __aexit__(self, *exc: object) -> bool:
                    return False

                async def wait_for_health(self, max_wait_seconds: float = 300) -> None:
                    pass

                async def chat_completion_json(
                    self, messages: list[dict], **kwargs: object
                ) -> dict:
                    return {"language": "nonsense"}

            with (
                mock.patch.object(mod, "probe_duration", fake_probe),
                mock.patch.object(mod, "VLLMClient", NoopClient),
            ):
                stats = mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                )

            en_train = out_dir / "en" / "train_en_q3asr.jsonl"
            self.assertEqual(stats[str(en_train)], 1)

            no_duration = out_dir / "rejected_no_duration.jsonl"
            unknown_lang = out_dir / "rejected_unknown_language.jsonl"
            self.assertEqual(stats[str(no_duration)], 1)
            self.assertEqual(stats[str(unknown_lang)], 1)
            rejected = json.loads(unknown_lang.read_text(encoding="utf-8"))
            self.assertEqual(rejected["audio"], "data/asr/mystery.wav")

    def test_out_of_band_durations_are_rejected_not_silently_emitted(self) -> None:
        """Training drops these anyway; the manifest must not claim them.

        The 32s case guards a real bug: the band was briefly taken from the
        QASRConfig dataclass default (30.0) instead of the 35.0 that every
        configs/*.yaml sets, which silently discarded the whole 30-35s band.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [
                    {"audio": "a.wav", "text": "language English<asr_text>Just right."},
                    {"audio": "d.wav", "text": "language English<asr_text>Long but ok."},
                    {"audio": "b.wav", "text": "language English<asr_text>Too long."},
                    {"audio": "c.wav", "text": "language English<asr_text>Too short."},
                ],
            )
            durations = {"a.wav": 5.0, "d.wav": 32.0, "b.wav": 45.0, "c.wav": 0.02}
            out_dir = tmp_path / "out"

            from data_processing.config import PipelineConfig

            with mock.patch.object(
                mod, "probe_duration", lambda p: durations[Path(p).name]
            ):
                stats = mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                )

            kept = out_dir / "en" / "train_en_q3asr.jsonl"
            self.assertEqual(stats[str(kept)], 2)  # 5.0s and 32.0s both train
            out_of_band = out_dir / "rejected_out_of_band_duration.jsonl"
            self.assertEqual(stats[str(out_of_band)], 2)

    def test_duration_band_matches_the_training_configs(self) -> None:
        """The band must track configs/*.yaml, which all set 0.1 / 35.0.

        35.0 is also the feature extractor's max_audio_clip_s, above which
        train.py refuses to start, so it cannot legitimately grow.
        """
        self.assertEqual(mod.TRAIN_MIN_DURATION, 0.1)
        self.assertEqual(mod.TRAIN_MAX_DURATION, 35.0)

    def test_intra_split_duplicates_dropped(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [
                    {"audio": "dup.wav", "text": "language English<asr_text>Once."},
                    {"audio": "dup.wav", "text": "language English<asr_text>Twice."},
                ],
            )
            out_dir = tmp_path / "out"

            from data_processing.config import PipelineConfig

            with mock.patch.object(mod, "probe_duration", lambda p: 2.0):
                stats = mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                )

            kept = out_dir / "en" / "train_en_q3asr.jsonl"
            self.assertEqual(stats[str(kept)], 1)


class _WordTokenizer:
    """Stand-in for the training tokenizer: one token per whitespace word.

    Records the kwargs it was called with, so the tests can assert we measure
    transcripts the way training does rather than merely assuming it.
    """

    is_fast = True

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text, **kwargs):
        self.calls.append(kwargs)
        batch = [text] if isinstance(text, str) else list(text)
        return {"input_ids": [item.split() for item in batch]}


class TranscriptLengthTest(unittest.TestCase):
    """Over-length transcripts are rejected, never truncated.

    Not a cosmetic count: with truncate_long_transcripts false,
    ResilientAudioDataset.__getitem__ marks an over-length index unusable and
    serves a *different* record in its place. Since __len__ is unchanged, the
    slot is filled by a duplicate of a neighbour, so leaving these in the
    manifest both loses the record and silently oversamples another one.
    """

    def test_budget_matches_the_training_configs(self) -> None:
        self.assertEqual(mod.MAX_TARGET_TOKENS, 512)

    def test_counts_use_the_same_rule_as_training(self) -> None:
        """data.py and collator.py both pass add_special_tokens=False,
        truncation=False, and tokenize the bare manifest text. Counting the
        "language X<asr_text>...<eos>" envelope instead would over-count and
        reject records training accepts."""
        tokenizer = _WordTokenizer()
        counts = mod.count_target_tokens(["a b c", "d"], tokenizer)

        self.assertEqual(counts, [3, 1])
        for kwargs in tokenizer.calls:
            self.assertIs(kwargs["add_special_tokens"], False)
            self.assertIs(kwargs["truncation"], False)

    def test_counts_are_batched_but_stay_aligned(self) -> None:
        tokenizer = _WordTokenizer()
        texts = [" ".join(["w"] * (i + 1)) for i in range(25)]

        counts = mod.count_target_tokens(texts, tokenizer, batch_size=4)

        self.assertEqual(counts, list(range(1, 26)))
        self.assertEqual(len(tokenizer.calls), 7)

    def test_over_length_transcripts_go_to_rejected_not_the_manifest(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            long_text = " ".join(["word"] * 600)
            edge_text = " ".join(["word"] * 512)  # exactly at the budget: kept
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [
                    {"audio": "ok.wav", "text": "language English<asr_text>Short one."},
                    {
                        "audio": "edge.wav",
                        "text": f"language English<asr_text>{edge_text}",
                    },
                    {
                        "audio": "long.wav",
                        "text": f"language English<asr_text>{long_text}",
                    },
                ],
            )
            out_dir = tmp_path / "out"

            from data_processing.config import PipelineConfig

            with mock.patch.object(mod, "probe_duration", lambda p: 4.0):
                stats = mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                    tokenizer=_WordTokenizer(),
                )

            kept = out_dir / "en" / "train_en_q3asr.jsonl"
            self.assertEqual(stats[str(kept)], 2)
            too_long = out_dir / "rejected_too_long_transcript.jsonl"
            self.assertEqual(stats[str(too_long)], 1)

            rejected = json.loads(too_long.read_text(encoding="utf-8"))
            self.assertEqual(rejected["audio"], "long.wav")
            # The count travels with the record so the pile is actionable.
            self.assertEqual(rejected["token_count"], 600)
            self.assertEqual(rejected["language"], "en")

    def test_rejected_transcripts_are_not_trimmed(self) -> None:
        """A truncated transcript is a wrong transcript: it would teach the
        model to stop mid-utterance. The pile keeps the full text."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            long_text = " ".join(["word"] * 600)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [{"audio": "long.wav", "text": f"language English<asr_text>{long_text}"}],
            )
            out_dir = tmp_path / "out"

            from data_processing.config import PipelineConfig

            with mock.patch.object(mod, "probe_duration", lambda p: 4.0):
                mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                    tokenizer=_WordTokenizer(),
                )

            rejected = json.loads(
                (out_dir / "rejected_too_long_transcript.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(rejected["text"], long_text)
            self.assertFalse((out_dir / "en").exists())

    def test_without_a_tokenizer_nothing_is_length_filtered(self) -> None:
        """--no-length-check must pass everything through rather than guess:
        a character-count estimate is what made this gap invisible in
        validate_data.py, where 512*4 chars lets ~600-token CJK slip by."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            long_text = " ".join(["word"] * 600)
            train_in = _write_input(
                tmp_path / "train_filter.jsonl",
                [{"audio": "long.wav", "text": f"language English<asr_text>{long_text}"}],
            )
            out_dir = tmp_path / "out"

            from data_processing.config import PipelineConfig

            with mock.patch.object(mod, "probe_duration", lambda p: 4.0):
                stats = mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                    tokenizer=None,
                )

            self.assertEqual(stats[str(out_dir / "en" / "train_en_q3asr.jsonl")], 1)
            self.assertNotIn(
                str(out_dir / "rejected_too_long_transcript.jsonl"), stats
            )

    def test_cli_refuses_to_run_unmeasured(self) -> None:
        """Omitting --tokenizer must fail rather than quietly skip the check."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            train_in = _write_input(
                Path(tmp) / "train_filter.jsonl",
                [{"audio": "a.wav", "text": "language English<asr_text>Hi."}],
            )
            argv = [
                "--inputs", str(train_in),
                "--output-dir", str(Path(tmp) / "out"),
                "--config", "configs/data_processing/multilingual_cleaning.yaml",
            ]

            with mock.patch.object(mod, "transform") as transform:
                self.assertEqual(mod.main(argv), 2)
            transform.assert_not_called()


class ShardedTransformTest(unittest.TestCase):
    """Distributing across nodes must not change the result."""

    ROWS: ClassVar[list[dict[str, str]]] = [
        {
            "audio": f"data/asr/{i}.wav",
            "text": (
                "language Arabic<asr_text>التي تعد مصدر دخلهم."
                if i % 3 == 0
                else "language English<asr_text>Our little bushel."
                if i % 3 == 1
                else "language Hindi<asr_text>नमस्ते दुनिया कैसे हो."
            ),
        }
        for i in range(60)
    ]

    def _run(self, tmp_path: Path, tag: str, num_nodes: int) -> dict[str, list[str]]:
        """Run the transform (sharded or not) and return {relpath: sorted lines}."""
        from data_processing.config import PipelineConfig

        train_in = _write_input(tmp_path / f"train_filter_{tag}.jsonl", self.ROWS)
        out_dir = tmp_path / f"out_{tag}"
        with mock.patch.object(mod, "probe_duration", lambda p: 3.0):
            for rank in range(num_nodes):
                mod.transform(
                    inputs=[str(train_in)],
                    base_dir=str(tmp_path),
                    output_dir=str(out_dir),
                    config=PipelineConfig(),
                    audit_labels=False,
                    node_rank=rank,
                    num_nodes=num_nodes,
                )
        if num_nodes > 1:
            rc = merge_mod.main(["--output-dir", str(out_dir), "--num-nodes", str(num_nodes)])
            self.assertEqual(rc, 0)
        return {
            str(p.relative_to(out_dir)): sorted(
                p.read_text(encoding="utf-8").splitlines()
            )
            for p in sorted(out_dir.rglob("*.jsonl"))
            if ".rank" not in p.name
        }

    def test_eight_shards_plus_merge_equals_single_node(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            single = self._run(tmp_path, "single", 1)
            sharded = self._run(tmp_path, "eight", 8)

        self.assertEqual(sorted(single), sorted(sharded))
        for name in single:
            self.assertEqual(single[name], sharded[name], f"{name} differs")
        # Sanity: the fixture really did exercise all three languages.
        self.assertEqual(len(single), 3)

    def test_merge_refuses_without_completion_markers(self) -> None:
        """A still-running node must not yield a manifest that looks finished."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            out_dir = root / "en"
            out_dir.mkdir(parents=True)
            for rank in (0, 1):
                (out_dir / f"train_en_q3asr.rank{rank}of4.jsonl").write_text(
                    '{"audio_filepath": "a.wav", "duration": 1.0, "text": "x"}\n',
                    encoding="utf-8",
                )
            # Only 2 of 4 nodes reported completion.
            for rank in (0, 1):
                (root / f"_shard_done.rank{rank}of4.json").write_text(
                    "{}", encoding="utf-8"
                )

            rc = merge_mod.main(["--output-dir", str(root), "--num-nodes", "4"])
            self.assertEqual(rc, 1)
            self.assertFalse((out_dir / "train_en_q3asr.jsonl").exists())

            rc = merge_mod.main(
                ["--output-dir", str(root), "--num-nodes", "4", "--allow-missing"]
            )
            self.assertEqual(rc, 0)
            merged = out_dir / "train_en_q3asr.jsonl"
            self.assertEqual(len(merged.read_text(encoding="utf-8").splitlines()), 2)

    def test_a_rank_with_no_records_for_a_language_is_not_a_failure(self) -> None:
        """The bug this design replaced.

        Completion used to be inferred from per-language shard presence, so a
        node that legitimately had no records for one language looked like a
        crashed node. Small languages make that the common case, which would
        have trained everyone to pass --allow-missing reflexively.
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "out"
            (root / "ml").mkdir(parents=True)
            # 4 nodes all finished, but only rank 2 saw any Malayalam.
            (root / "ml" / "train_ml_q3asr.rank2of4.jsonl").write_text(
                '{"audio_filepath": "m.wav", "duration": 1.0, "text": "y"}\n',
                encoding="utf-8",
            )
            for rank in range(4):
                (root / f"_shard_done.rank{rank}of4.json").write_text(
                    "{}", encoding="utf-8"
                )

            rc = merge_mod.main(["--output-dir", str(root), "--num-nodes", "4"])
            self.assertEqual(rc, 0)
            merged = root / "ml" / "train_ml_q3asr.jsonl"
            self.assertEqual(len(merged.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
