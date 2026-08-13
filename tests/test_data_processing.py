"""Tests for the data_processing package."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from data_processing.arabic_utils import (
    arabic_ratio,
    classify_dialect,
    contains_arabic,
    has_diacritics,
    normalize_whitespace,
    preprocess_text,
    remove_tatweel,
    strip_control_chars,
)
from data_processing.config import PipelineConfig
from data_processing.generic_prompts import (
    build_generic_cleaner_messages,
    build_generic_validator_messages,
)
from data_processing.itn import (
    check_digit_consistency,
    normalize_digits_to_western,
    verify_itn,
)
from data_processing.manifest_io import (
    CleanedRecord,
    collect_processed_keys,
    merge_shards,
    read_manifest,
    shard_manifest,
    write_shard_atomic,
)
from data_processing.prompts import (
    build_batch_cleaner_messages,
    build_cleaner_messages,
    build_validator_messages,
)
from data_processing.text_utils import (
    contains_expected_script,
    devanagari_to_western,
    expected_script_ratio,
    preprocess_text_generic,
)


class TestArabicUtils(unittest.TestCase):
    """Tests for arabic_utils module."""

    def test_remove_tatweel(self) -> None:
        text = "مرحـــبا"
        result = remove_tatweel(text)
        self.assertEqual(result, "مرحبا")

    def test_strip_control_chars(self) -> None:
        text = "مرحبا\x00\x01\x02 بالعالم"
        result = strip_control_chars(text)
        self.assertEqual(result, "مرحبا بالعالم")

    def test_normalize_whitespace(self) -> None:
        text = "مرحبا    بالعالم\t\t test"
        result = normalize_whitespace(text)
        self.assertEqual(result, "مرحبا بالعالم test")

    def test_preprocess_text(self) -> None:
        text = "مرحـــبا\x00  [noise]   بالعالم"
        result = preprocess_text(text)
        self.assertEqual(result, "مرحبا بالعالم")

    def test_contains_arabic(self) -> None:
        self.assertTrue(contains_arabic("مرحبا"))
        self.assertTrue(contains_arabic("Hello مرحبا"))
        self.assertFalse(contains_arabic("Hello World"))
        self.assertFalse(contains_arabic("12345"))

    def test_arabic_ratio(self) -> None:
        self.assertAlmostEqual(arabic_ratio("مرحبا"), 1.0)
        self.assertAlmostEqual(arabic_ratio("Hello"), 0.0)
        ratio = arabic_ratio("مرحبا Hello")
        self.assertGreater(ratio, 0.4)
        self.assertLess(ratio, 0.7)

    def test_has_diacritics(self) -> None:
        self.assertTrue(has_diacritics("مُحَمَّد"))  # With diacritics
        self.assertFalse(has_diacritics("محمد"))  # Without diacritics

    def test_classify_dialect_gulf(self) -> None:
        text = "وش كيفك الحين"
        dialect = classify_dialect(text)
        self.assertEqual(dialect, "gulf")

    def test_classify_dialect_msa(self) -> None:
        text = "هذا كتاب مفيد جداً"
        dialect = classify_dialect(text)
        self.assertEqual(dialect, "msa")


class TestITN(unittest.TestCase):
    """Tests for ITN verification module."""

    def test_check_digit_consistency_western_only(self) -> None:
        consistent, issue = check_digit_consistency("لدي 5 كتب و 10 أقلام")
        self.assertTrue(consistent)
        self.assertIsNone(issue)

    def test_check_digit_consistency_mixed(self) -> None:
        consistent, issue = check_digit_consistency("لدي ٥ كتب و 10 أقلام")
        self.assertFalse(consistent)
        self.assertIsNotNone(issue)

    def test_normalize_digits_to_western(self) -> None:
        text = "لدي ٥ كتب"
        result = normalize_digits_to_western(text)
        self.assertEqual(result, "لدي 5 كتب")

    def test_verify_itn_valid(self) -> None:
        result = verify_itn("لدي 5 كتب")
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.issues), 0)

    def test_verify_itn_mixed_digits(self) -> None:
        result = verify_itn("لدي ٥ كتب و 10 أقلام")
        self.assertFalse(result.is_valid)
        self.assertTrue(result.has_mixed_digits)


class TestManifestIO(unittest.TestCase):
    """Tests for manifest I/O module."""

    def test_read_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test.jsonl"
            records = [
                {"audio_filepath": "a.wav", "text": "مرحبا", "duration": 1.0},
                {"audio_filepath": "b.wav", "text": "أهلا", "duration": 2.0},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
                encoding="utf-8",
            )

            result = list(read_manifest(manifest_path))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0].text, "مرحبا")
            self.assertEqual(result[1].audio_filepath, "b.wav")

    def test_shard_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "test.jsonl"
            records = [
                {"audio_filepath": f"{i}.wav", "text": f"text{i}", "duration": 1.0}
                for i in range(10)
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(r) for r in records),
                encoding="utf-8",
            )

            shard_dir = Path(tmpdir) / "shards"
            shard_paths = shard_manifest(manifest_path, 3, shard_dir)

            self.assertEqual(len(shard_paths), 3)
            total_records = 0
            for shard_path in shard_paths:
                total_records += len(list(read_manifest(shard_path)))
            self.assertEqual(total_records, 10)

    def test_write_shard_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"
            records = [
                CleanedRecord(
                    audio_filepath="a.wav",
                    text="cleaned text",
                    duration=1.0,
                    original_text="original",
                    dialect="msa",
                    confidence=0.9,
                    changes=["clean", "punct"],
                )
            ]
            write_shard_atomic(output_path, records)

            self.assertTrue(output_path.is_file())
            content = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(content["text"], "cleaned text")

    def test_merge_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            shard_paths = []
            for i in range(3):
                shard_path = Path(tmpdir) / f"shard_{i}.jsonl"
                records = [
                    {"audio_filepath": f"{i}_{j}.wav", "text": f"text{i}{j}"}
                    for j in range(5)
                ]
                shard_path.write_text(
                    "\n".join(json.dumps(r) for r in records),
                    encoding="utf-8",
                )
                shard_paths.append(shard_path)

            output_path = Path(tmpdir) / "merged.jsonl"
            total = merge_shards(shard_paths, output_path)

            self.assertEqual(total, 15)
            lines = output_path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 15)


class TestPrompts(unittest.TestCase):
    """Tests for prompt templates."""

    def test_build_cleaner_messages(self) -> None:
        messages = build_cleaner_messages("مرحبا بالعالم")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("مرحبا بالعالم", messages[1]["content"])

    def test_build_batch_cleaner_messages(self) -> None:
        transcripts = ["مرحبا", "أهلا", "صباح الخير"]
        messages = build_batch_cleaner_messages(transcripts)
        self.assertEqual(len(messages), 2)
        self.assertIn("1. <<<مرحبا>>>", messages[1]["content"])
        self.assertIn("3. <<<صباح الخير>>>", messages[1]["content"])

    def test_build_validator_messages(self) -> None:
        messages = build_validator_messages(
            original="مرحبا",
            processed="مرحبا.",
            dialect="msa",
            changes=["punct"],
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("مرحبا", messages[1]["content"])
        self.assertIn("msa", messages[1]["content"])


class TestTextUtils(unittest.TestCase):
    """Tests for language-agnostic text utilities."""

    def test_preprocess_text_generic(self) -> None:
        text = "Hello\x00  [noise]   world"
        result = preprocess_text_generic(text)
        self.assertEqual(result, "Hello world")

    def test_contains_expected_script(self) -> None:
        self.assertTrue(contains_expected_script("Hello world", "en"))
        self.assertTrue(contains_expected_script("你好世界", "zh"))
        self.assertTrue(contains_expected_script("नमस्ते", "hi"))
        self.assertTrue(contains_expected_script("നമസ്കാരം", "ml"))
        self.assertFalse(contains_expected_script("Hello", "zh"))
        self.assertFalse(contains_expected_script("12345", "en"))

    def test_expected_script_ratio_code_switching(self) -> None:
        # Hinglish: Latin chars count as expected for non-English languages
        ratio = expected_script_ratio("मैं office जा रहा हूं", "hi")
        self.assertEqual(ratio, 1.0)
        # But Devanagari in an English transcript lowers the ratio
        ratio_en = expected_script_ratio("मैं office", "en")
        self.assertLess(ratio_en, 1.0)

    def test_devanagari_to_western(self) -> None:
        self.assertEqual(devanagari_to_western("१२३ किताबें"), "123 किताबें")


class TestGenericPrompts(unittest.TestCase):
    """Tests for the generic multilingual prompt templates."""

    def test_build_generic_cleaner_messages(self) -> None:
        messages = build_generic_cleaner_messages("hello world", "en")
        self.assertEqual(len(messages), 2)
        self.assertIn("English", messages[0]["content"])
        self.assertIn("hello world", messages[1]["content"])

    def test_language_conventions_injected(self) -> None:
        zh_messages = build_generic_cleaner_messages("你好", "zh")
        self.assertIn("。", zh_messages[0]["content"])  # full-width punctuation
        hi_messages = build_generic_cleaner_messages("नमस्ते", "hi")
        self.assertIn("।", hi_messages[0]["content"])  # danda

    def test_build_generic_validator_messages(self) -> None:
        messages = build_generic_validator_messages(
            original="three books",
            processed="3 books.",
            language="en",
            changes=["itn", "punct"],
            auto_flags=["Cleaner reported low confidence: 0.20"],
        )
        self.assertEqual(len(messages), 2)
        self.assertIn("three books", messages[1]["content"])
        self.assertIn("adjudicate", messages[1]["content"])


class TestAgentRouting(unittest.TestCase):
    """Tests for language-based cleaner/validator routing."""

    def test_arabic_routes_to_specialized_agents(self) -> None:
        from data_processing.cleaner import CleanerAgent
        from data_processing.generic_cleaner import GenericCleanerAgent
        from data_processing.validator import ValidatorAgent
        from data_processing.worker import build_agents

        cleaner, validator = build_agents(None, PipelineConfig(), "ar")
        self.assertIs(type(cleaner), CleanerAgent)
        self.assertIs(type(validator), ValidatorAgent)
        self.assertNotIsInstance(cleaner, GenericCleanerAgent)

    def test_other_languages_route_to_generic_agents(self) -> None:
        from data_processing.generic_cleaner import GenericCleanerAgent
        from data_processing.generic_validator import GenericValidatorAgent
        from data_processing.worker import build_agents

        for language in ("en", "zh", "hi", "ml"):
            cleaner, validator = build_agents(None, PipelineConfig(), language)
            self.assertIsInstance(cleaner, GenericCleanerAgent)
            self.assertIsInstance(validator, GenericValidatorAgent)
            self.assertEqual(cleaner.language, language)
            self.assertEqual(validator.language, language)


class _ScriptedClient:
    """Fake VLLMClient returning scripted JSON responses in order."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def chat_completion_json(self, messages, **kwargs):
        self.calls.append(messages)
        return self._responses.pop(0)


def _cleaned_record(text: str, original: str = "\u0627\u0644\u0646\u0635 \u0627\u0644\u0623\u0635\u0644\u064a"):
    return CleanedRecord(
        audio_filepath="a.wav",
        text=text,
        duration=1.0,
        original_text=original,
        dialect="msa",
        confidence=0.8,
        changes=["clean"],
    )


class TestCorrector(unittest.TestCase):
    """Tests for the issue-driven CorrectorAgent and its prompt."""

    def test_build_corrector_messages_embeds_issues_and_language(self) -> None:
        from data_processing.prompts import build_corrector_messages

        msgs = build_corrector_messages(
            "orig text", "proc text", ["wrong harakat", "bad punctuation"], language="ar"
        )
        self.assertEqual(len(msgs), 2)
        system, user = msgs[0]["content"], msgs[1]["content"]
        self.assertIn("Arabic", system)
        self.assertIn("wrong harakat", user)
        self.assertIn("bad punctuation", user)
        self.assertIn("orig text", user)
        self.assertIn("proc text", user)

    def test_corrector_declines_without_issues(self) -> None:
        from data_processing.corrector import CorrectorAgent

        client = _ScriptedClient([])  # must never be called
        corrector = CorrectorAgent(client, language="ar")
        result = asyncio.run(corrector.correct(_cleaned_record("\u0627\u0644\u0646\u0635"), []))
        self.assertIsNone(result)
        self.assertEqual(len(client.calls), 0)

    def test_corrector_declines_on_null_output(self) -> None:
        from data_processing.corrector import CorrectorAgent

        client = _ScriptedClient([{"corrected_text": None}])
        corrector = CorrectorAgent(client, language="ar")
        result = asyncio.run(corrector.correct(_cleaned_record("\u0627\u0644\u0646\u0635"), ["x"]))
        self.assertIsNone(result)

    def test_corrector_declines_when_unchanged(self) -> None:
        from data_processing.corrector import CorrectorAgent

        client = _ScriptedClient([{"corrected_text": "\u0627\u0644\u0646\u0635"}])
        corrector = CorrectorAgent(client, language="ar")
        result = asyncio.run(corrector.correct(_cleaned_record("\u0627\u0644\u0646\u0635"), ["x"]))
        self.assertIsNone(result)

    def test_corrector_returns_changed_text(self) -> None:
        from data_processing.corrector import CorrectorAgent

        client = _ScriptedClient([{"corrected_text": "\u0646\u0635 \u0645\u0635\u062d\u062d"}])
        corrector = CorrectorAgent(client, language="ar")
        result = asyncio.run(corrector.correct(_cleaned_record("\u0627\u0644\u0646\u0635"), ["x"]))
        self.assertEqual(result, "\u0646\u0635 \u0645\u0635\u062d\u062d")


class TestValidatorCorrectionStage(unittest.TestCase):
    """Tests for Stage 2 issue-driven correction in validate_with_retry."""

    def _make_validator(self, responses: list[dict]):
        from data_processing.corrector import CorrectorAgent
        from data_processing.validator import ValidatorAgent

        client = _ScriptedClient(responses)
        config = PipelineConfig()
        corrector = CorrectorAgent(client, config=config, language="ar")
        validator = ValidatorAgent(client, config=config, corrector=corrector)
        return validator, client

    def test_stage2_rescues_reject_with_no_inline_correction(self) -> None:
        # invalid + issues + no corrected_text (rounds=0) -> corrector -> valid
        validator, client = self._make_validator([
            {"valid": False, "issues": ["wrong punctuation"], "corrected_text": None, "quality_score": 0.5},
            {"corrected_text": "\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0635\u062d\u062d"},
            {"valid": True, "issues": [], "corrected_text": None, "quality_score": 0.95},
        ])
        record = _cleaned_record("\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0639\u0627\u0644\u062c")
        final, accepted = asyncio.run(validator.validate_with_retry(record))
        self.assertTrue(accepted)
        self.assertEqual(final.text, "\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0635\u062d\u062d")
        self.assertIn("issue_corrected", final.changes)
        self.assertEqual(len(client.calls), 3)

    def test_stage2_reject_stands_when_corrector_declines(self) -> None:
        validator, client = self._make_validator([
            {"valid": False, "issues": ["x"], "corrected_text": None, "quality_score": 0.5},
            {"corrected_text": None},  # corrector declines -> no re-validate
        ])
        record = _cleaned_record("\u0627\u0644\u0646\u0635 \u0627\u0644\u0645\u0639\u0627\u0644\u062c")
        _final, accepted = asyncio.run(validator.validate_with_retry(record))
        self.assertFalse(accepted)
        self.assertEqual(len(client.calls), 2)

    def test_build_agents_gates_corrector_on_flag(self) -> None:
        from data_processing.worker import build_agents

        _, validator_on = build_agents(None, PipelineConfig(correction_enabled=True), "ar")
        self.assertIsNotNone(validator_on.corrector)
        _, validator_off = build_agents(None, PipelineConfig(correction_enabled=False), "ar")
        self.assertIsNone(validator_off.corrector)
        _, generic_on = build_agents(None, PipelineConfig(correction_enabled=True), "hi")
        self.assertIsNotNone(generic_on.corrector)
        self.assertEqual(generic_on.corrector.language, "hi")


class TestConfig(unittest.TestCase):
    """Tests for pipeline configuration."""

    def test_default_config(self) -> None:
        config = PipelineConfig()
        self.assertEqual(config.workers_per_node, 96)
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.vllm_tp, 8)

    def test_config_validation(self) -> None:
        config = PipelineConfig(workers_per_node=0)
        with self.assertRaises(ValueError):
            config.validate()

    def test_config_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "workers_per_node: 24\nbatch_size: 16\n",
                encoding="utf-8",
            )
            config = PipelineConfig.from_yaml(config_path)
            self.assertEqual(config.workers_per_node, 24)
            self.assertEqual(config.batch_size, 16)

    def test_vllm_base_url(self) -> None:
        config = PipelineConfig(vllm_host="localhost", vllm_port=8000)
        self.assertEqual(config.vllm_base_url, "http://localhost:8000/v1")


class TestRecordRange(unittest.TestCase):
    """Record-range slicing for spreading one manifest across nodes."""

    def test_parse_record_range(self) -> None:
        self.assertEqual(PipelineConfig().parse_record_range(), (0, None))
        self.assertEqual(
            PipelineConfig(record_range="100:200").parse_record_range(), (100, 200)
        )
        self.assertEqual(
            PipelineConfig(record_range="120000:").parse_record_range(),
            (120000, None),
        )
        self.assertEqual(
            PipelineConfig(record_range=":500").parse_record_range(), (0, 500)
        )

    def test_parse_record_range_invalid(self) -> None:
        for bad in ("200:100", "abc:2", "-5:10", "42"):
            with self.assertRaises(ValueError):
                PipelineConfig(record_range=bad).parse_record_range()

    def _write_manifest(self, path: Path, count: int) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(
                    json.dumps(
                        {"audio_filepath": f"a{i}.wav", "text": f"\u0646\u0635 {i}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def test_shard_manifest_slice_partition(self) -> None:
        """Slices [0:4), [4:8), [8:EOF) partition the manifest exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "m.jsonl"
            self._write_manifest(manifest, 10)

            seen: list[str] = []
            for start, end in ((0, 4), (4, 8), (8, None)):
                shard_dir = tmp / f"s_{start}"
                paths = shard_manifest(
                    manifest, 2, shard_dir, start=start, end=end
                )
                for path in paths:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        seen.append(json.loads(line)["audio_filepath"])
            self.assertEqual(sorted(seen), sorted(f"a{i}.wav" for i in range(10)))
            self.assertEqual(len(seen), 10)

    def test_shard_manifest_skip_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "m.jsonl"
            self._write_manifest(manifest, 6)
            skip = {("a0.wav", "\u0646\u0635 0"), ("a3.wav", "\u0646\u0635 3")}
            paths = shard_manifest(manifest, 2, tmp / "s", skip_keys=skip)
            kept = [
                json.loads(line)["audio_filepath"]
                for path in paths
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                sorted(kept), ["a1.wav", "a2.wav", "a4.wav", "a5.wav"]
            )

    def test_collect_processed_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            # cleaned output carries original_text; rejected carries text
            (out / "m_cleaned.jsonl").write_text(
                json.dumps(
                    {
                        "audio_filepath": "a0.wav",
                        "text": "\u0645\u064f\u0646\u0642\u0651\u062d",
                        "original_text": "\u0646\u0635 0",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (out / "m_r5_end_rejected.jsonl").write_text(
                json.dumps(
                    {"audio_filepath": "a7.wav", "text": "\u0646\u0635 7"},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            keys = collect_processed_keys(out, "m")
            self.assertEqual(
                keys,
                {("a0.wav", "\u0646\u0635 0"), ("a7.wav", "\u0646\u0635 7")},
            )


class TestManifestResilience(unittest.TestCase):
    """One corrupt line must not cost the whole shard."""

    def _manifest(self, tmp: Path, lines: list[bytes]) -> Path:
        path = tmp / "m.jsonl"
        path.write_bytes(b"\n".join(lines) + b"\n")
        return path

    def _good(self, name: str) -> bytes:
        return json.dumps(
            {"audio_filepath": f"{name}.wav", "text": "\u0646\u0635"},
            ensure_ascii=False,
        ).encode("utf-8")

    def test_malformed_json_line_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._manifest(
                Path(tmpdir),
                [self._good("a"), b'{"audio_filepath": "b.wav", "tex', self._good("c")],
            )
            records = list(read_manifest(path))
            self.assertEqual(
                [r.audio_filepath for r in records], ["a.wav", "c.wav"]
            )
            # Line numbers stay true to the file, so checkpoints remain valid
            self.assertEqual([r.line_number for r in records], [1, 3])

    def test_non_object_line_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._manifest(
                Path(tmpdir), [b'["not", "an", "object"]', self._good("a")]
            )
            self.assertEqual(
                [r.audio_filepath for r in read_manifest(path)], ["a.wav"]
            )

    def test_invalid_utf8_only_loses_its_own_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._manifest(
                Path(tmpdir),
                [self._good("a"), b'{"audio_filepath": "\xff\xfe.wav"}', self._good("c")],
            )
            records = list(read_manifest(path))
            self.assertEqual(
                [r.audio_filepath for r in records], ["a.wav", "c.wav"]
            )

    def test_strict_mode_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._manifest(Path(tmpdir), [self._good("a"), b"{oops"])
            with self.assertRaises(ValueError):
                list(read_manifest(path, strict=True))


class TestITNSoftFlags(unittest.TestCase):
    """Correction-safe ITN checks wired into the validator."""

    def test_clean_text_has_no_flags(self) -> None:
        from data_processing.itn import itn_soft_flags

        text = "\u0641\u064a \u0639\u0627\u0645 2019 \u0632\u0627\u0631 \u0627\u0644\u0645\u062f\u064a\u0646\u0629"
        self.assertEqual(itn_soft_flags(text, text), [])

    def test_dropped_number_is_flagged(self) -> None:
        from data_processing.itn import itn_soft_flags

        flags = itn_soft_flags(
            "\u0641\u064a \u0639\u0627\u0645 \u0632\u0627\u0631", "\u0641\u064a \u0639\u0627\u0645 2019 \u0632\u0627\u0631"
        )
        self.assertEqual(len(flags), 1)
        self.assertIn("2019", flags[0])

    def test_corrupted_number_is_flagged(self) -> None:
        from data_processing.itn import check_dropped_numbers

        self.assertTrue(check_dropped_numbers("\u0639\u0627\u0645 2109", "\u0639\u0627\u0645 2019"))

    def test_legitimate_reformats_are_not_flagged(self) -> None:
        from data_processing.itn import check_dropped_numbers

        # separators dropped, digits re-split/re-joined, stutter de-duplicated,
        # Arabic-Indic source digits normalized
        for cleaned, original in (
            ("1000", "1,000"),
            ("2019", "20 19"),
            ("5", "5 5"),
            ("2019", "\u0662\u0660\u0661\u0669"),
        ):
            self.assertEqual(
                check_dropped_numbers(cleaned, original), [], f"{original} -> {cleaned}"
            )

    def test_large_number_is_flagged(self) -> None:
        from data_processing.itn import itn_soft_flags

        flags = itn_soft_flags("\u0631\u0642\u0645 12345678901")
        self.assertTrue(any("12345678901" in f for f in flags))

    def test_number_noun_agreement_never_reaches_the_llm(self) -> None:
        """Prompt rule 2 forbids altering the counted noun — so never hint it."""
        from data_processing.itn import check_number_noun_agreement, itn_soft_flags

        text = "\u062e\u0645\u0633\u0629 \u0645\u0631\u0629"
        self.assertTrue(check_number_noun_agreement(text))  # diagnostic sees it
        self.assertEqual(itn_soft_flags(text, text), [])  # advisory path does not

    def test_enforce_digit_charset(self) -> None:
        from data_processing.itn import enforce_digit_charset

        self.assertEqual(
            enforce_digit_charset("\u0639\u0627\u0645 \u0662\u0660\u0661\u0669\u06d4"), "\u0639\u0627\u0645 2019."
        )

    def test_itn_flags_gated_by_config(self) -> None:
        from data_processing.validator import ValidatorAgent

        record = _cleaned_record("\u0639\u0627\u0645 \u0632\u0627\u0631", "\u0639\u0627\u0645 2019 \u0632\u0627\u0631")
        on = ValidatorAgent(None, config=PipelineConfig(itn_enabled=True))
        off = ValidatorAgent(None, config=PipelineConfig(itn_enabled=False))
        self.assertTrue(any("2019" in f for f in on._soft_flags(record)))
        self.assertFalse(any("2019" in f for f in off._soft_flags(record)))


class TestCorrectionCharset(unittest.TestCase):
    """Corrected text must get the same deterministic post-processing."""

    def test_corrected_text_is_charset_normalized(self) -> None:
        from data_processing.validator import ValidatorAgent

        client = _ScriptedClient([
            {
                "valid": False,
                "issues": ["x"],
                "corrected_text": "\u0639\u0627\u0645 \u0662\u0660\u0661\u0669\u06d4",
                "quality_score": 0.4,
            },
            {"valid": True, "issues": [], "quality_score": 0.95},
        ])
        validator = ValidatorAgent(client, config=PipelineConfig())
        final, accepted = asyncio.run(
            validator.validate_with_retry(_cleaned_record("\u0639\u0627\u0645 \u0633\u0627\u0628\u0642"))
        )
        self.assertTrue(accepted)
        self.assertEqual(final.text, "\u0639\u0627\u0645 2019.")

    def test_charset_only_correction_does_not_burn_a_round(self) -> None:
        """A 'correction' that normalizes back to the current text stops."""
        from data_processing.validator import ValidatorAgent

        client = _ScriptedClient([
            {
                "valid": False,
                "issues": ["x"],
                "corrected_text": "\u0639\u0627\u0645 \u0662\u0660\u0661\u0669",
                "quality_score": 0.4,
            },
        ])
        validator = ValidatorAgent(client, config=PipelineConfig())
        _final, accepted = asyncio.run(
            validator.validate_with_retry(_cleaned_record("\u0639\u0627\u0645 2019"))
        )
        self.assertFalse(accepted)
        self.assertEqual(len(client.calls), 1)  # no pointless re-validation

    def test_generic_validator_does_not_apply_arabic_charset(self) -> None:
        from data_processing.generic_validator import GenericValidatorAgent

        zh = GenericValidatorAgent(None, config=PipelineConfig(), language="zh")
        self.assertEqual(zh._normalize_corrected("\u0662\u0660\u0661\u0669"), "\u0662\u0660\u0661\u0669")
        hi = GenericValidatorAgent(None, config=PipelineConfig(), language="hi")
        self.assertEqual(hi._normalize_corrected("\u0968\u0966\u0967\u096f"), "2019")


class _FakeClient:
    """Stand-in for VLLMClient: an async context manager and nothing more."""

    def __init__(self, config) -> None:
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False


class _TaggingCleaner:
    """Returns one CleanedRecord per input, tagged with fixed `changes`."""

    def __init__(self, changes: list[str]) -> None:
        self.changes = changes

    async def process_batch(self, records):
        return [
            CleanedRecord(
                audio_filepath=r.audio_filepath,
                text=r.text,
                duration=r.duration,
                original_text=r.text,
                dialect="msa",
                confidence=0.9,
                changes=list(self.changes),
            )
            for r in records
        ]


class _CountingValidator:
    """Accepts everything, but records which files it was asked about."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def validate_with_retry(self, record):
        self.seen.append(record.audio_filepath)
        return record, True


class TestWorkerResumeAndSampling(unittest.TestCase):
    """Checkpoint invalidation and the validation-sampling exemptions."""

    def _shard(self, tmp: Path, count: int = 2) -> dict:
        tmp.mkdir(parents=True, exist_ok=True)
        shard = tmp / "shard_0000.jsonl"
        with shard.open("w", encoding="utf-8") as handle:
            for i in range(count):
                handle.write(
                    json.dumps(
                        {
                            "audio_filepath": f"a{i}.wav",
                            "text": "\u0646\u0635 \u0637\u0648\u064a\u0644 \u0643\u0641\u0627\u064a\u0629",
                            "duration": 1.0,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return {
            "shard_path": shard,
            "output_path": tmp / "out.jsonl",
            "rejected_path": tmp / "rej.jsonl",
            "checkpoint_path": tmp / "ckpt.json",
        }

    def _run(self, paths: dict, changes: list[str], **overrides):
        from data_processing import worker as worker_mod

        config = PipelineConfig(
            batch_size=overrides.pop("batch_size", 8),
            validation_sample_rate=overrides.pop("validation_sample_rate", 0.0),
            **overrides,
        )
        cleaner = _TaggingCleaner(changes)
        validator = _CountingValidator()
        probe = mock.AsyncMock(return_value=0)
        with mock.patch.object(worker_mod, "VLLMClient", _FakeClient), mock.patch.object(
            worker_mod, "build_agents", return_value=(cleaner, validator)
        ), mock.patch.object(worker_mod, "fill_missing_durations", probe):
            worker = worker_mod.Worker(
                worker_id=0, config=config, language="ar", **paths
            )
            stats = asyncio.run(worker.run())
        return stats, validator, probe

    @staticmethod
    def _lines(path: Path) -> list[dict]:
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_stale_checkpoint_is_discarded(self) -> None:
        """A checkpoint from different shard content must not skip records."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            paths["checkpoint_path"].write_text(
                json.dumps(
                    {
                        "last_line_number": 2,
                        "records_processed": 2,
                        "timestamp": 0.0,
                        "shard_signature": "0" * 32,
                    }
                ),
                encoding="utf-8",
            )
            stats, _validator, _probe = self._run(paths, ["clean"])
            self.assertEqual(stats.processed_records, 2)
            self.assertEqual(len(self._lines(paths["output_path"])), 2)

    def test_checkpoint_without_signature_is_discarded(self) -> None:
        """Checkpoints written before fingerprinting cannot be trusted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            paths["checkpoint_path"].write_text(
                json.dumps(
                    {
                        "last_line_number": 2,
                        "records_processed": 2,
                        "timestamp": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            stats, _validator, _probe = self._run(paths, ["clean"])
            self.assertEqual(stats.processed_records, 2)

    def test_matching_checkpoint_still_resumes(self) -> None:
        """The fingerprint must not defeat legitimate resume."""
        from data_processing.worker import compute_shard_signature

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            paths["checkpoint_path"].write_text(
                json.dumps(
                    {
                        "last_line_number": 1,
                        "records_processed": 1,
                        "timestamp": 0.0,
                        "shard_signature": compute_shard_signature(
                            paths["shard_path"]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            _stats, validator, _probe = self._run(
                paths, ["skipped_short"], validation_sample_rate=0.0
            )
            self.assertEqual(validator.seen, ["a1.wav"])

    def test_checkpoint_records_signature(self) -> None:
        from data_processing.worker import Checkpoint, compute_shard_signature

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            self._run(paths, ["clean"])
            checkpoint = Checkpoint.load(paths["checkpoint_path"])
            self.assertEqual(
                checkpoint.shard_signature,
                compute_shard_signature(paths["shard_path"]),
            )

    def test_error_fallback_is_rejected_without_an_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            stats, validator, _probe = self._run(paths, ["error_fallback"])
            self.assertEqual(validator.seen, [])  # never cleaned, never validated
            self.assertEqual(stats.accepted_records, 0)
            self.assertEqual(stats.rejected_records, 2)
            rejected = self._lines(paths["rejected_path"])
            self.assertEqual(len(rejected), 2)
            self.assertEqual(
                rejected[0]["rejection_reasons"], ["cleaner_error_fallback"]
            )
            self.assertEqual(self._lines(paths["output_path"]), [])

    def test_skipped_records_bypass_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for tag in ("skipped_short", "skipped_wrong_script"):
                paths = self._shard(Path(tmpdir) / tag)
                _stats, validator, _probe = self._run(paths, [tag])
                self.assertEqual(
                    validator.seen, ["a0.wav", "a1.wav"], f"tag={tag}"
                )

    def test_normal_records_respect_the_sample_rate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            stats, validator, _probe = self._run(paths, ["clean", "itn"])
            self.assertEqual(validator.seen, [])
            self.assertEqual(stats.accepted_records, 2)

    def test_durations_probed_before_cleaning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            _stats, _validator, probe = self._run(paths, ["clean"])
            probe.assert_awaited_once()
            self.assertEqual(probe.await_args.args[1], 32)

    def test_duration_probe_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = self._shard(Path(tmpdir))
            _stats, _validator, probe = self._run(
                paths, ["clean"], probe_missing_durations=False
            )
            probe.assert_not_awaited()


class TestDurationProbe(unittest.TestCase):
    """Header-only duration probing for records the manifest never carried."""

    def _record(self, path: str, duration: float | None):
        from data_processing.manifest_io import ManifestRecord

        return ManifestRecord(
            line_number=1,
            audio_filepath=path,
            text="\u0646\u0635",
            duration=duration,
            raw={},
        )

    def test_existing_durations_cost_no_io(self) -> None:
        from data_processing import duration_probe

        records = [self._record("/nonexistent/a.wav", 3.5)]
        with mock.patch.object(duration_probe, "probe_duration") as probe:
            filled = asyncio.run(duration_probe.fill_missing_durations(records))
        self.assertEqual(filled, 0)
        probe.assert_not_called()
        self.assertEqual(records[0].duration, 3.5)

    def test_missing_duration_is_read_from_the_header(self) -> None:
        import numpy as np
        import soundfile as sf

        from data_processing import duration_probe

        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "a.wav"
            sf.write(audio, np.zeros(16000, dtype="float32"), 16000)
            records = [self._record(str(audio), None)]
            filled = asyncio.run(duration_probe.fill_missing_durations(records))
            self.assertEqual(filled, 1)
            self.assertAlmostEqual(records[0].duration, 1.0, places=3)

    def test_unreadable_audio_leaves_duration_unset(self) -> None:
        from data_processing import duration_probe

        records = [self._record("/nonexistent/a.wav", None)]
        filled = asyncio.run(duration_probe.fill_missing_durations(records))
        self.assertEqual(filled, 0)
        self.assertIsNone(records[0].duration)


class TestOrchestratorJobIdentity(unittest.TestCase):
    """Reporting reads output paths off the job, so the job must carry them."""

    def _config(self, tmp: Path, **overrides) -> PipelineConfig:
        return PipelineConfig(
            output_dir=str(tmp / "out"),
            shard_dir=str(tmp / "shards"),
            checkpoint_dir=str(tmp / "ckpt"),
            workers_per_node=2,
            num_nodes=1,
            **overrides,
        )

    def _process(self, tmp: Path, config: PipelineConfig):
        from data_processing.orchestrator import ManifestJob, Orchestrator

        manifest = tmp / "train_ar.jsonl"
        with manifest.open("w", encoding="utf-8") as handle:
            for i in range(6):
                handle.write(
                    json.dumps(
                        {"audio_filepath": f"a{i}.wav", "text": f"\u0646\u0635 {i}"},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        class _NoWorkers(Orchestrator):
            async def _launch_workers(self, **kwargs):
                return []

        job = ManifestJob(path=manifest, language="ar")
        asyncio.run(_NoWorkers(config).process_manifest(job))
        return job

    def test_job_carries_resolved_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            job = self._process(tmp, self._config(tmp))
            self.assertEqual(job.manifest_name, "train_ar")
            self.assertEqual(
                job.output_path, tmp / "out" / "ar" / "train_ar_cleaned.jsonl"
            )
            self.assertEqual(
                job.rejected_path, tmp / "out" / "ar" / "train_ar_rejected.jsonl"
            )

    def test_slice_suffix_is_kept_in_output_identity(self) -> None:
        """Re-deriving names from path.stem would lose the range suffix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            job = self._process(tmp, self._config(tmp, record_range="2:4"))
            self.assertEqual(job.manifest_name, "train_ar_r2_4")
            self.assertEqual(job.output_path.name, "train_ar_r2_4_cleaned.jsonl")
            self.assertEqual(job.rejected_path.name, "train_ar_r2_4_rejected.jsonl")

    def test_report_carries_the_jobs_that_ran(self) -> None:
        from data_processing.orchestrator import OrchestratorReport

        self.assertEqual(OrchestratorReport().jobs, [])


class TestDateTimeITN(unittest.TestCase):
    """Date/time ITN: flag only field values no calendar can explain."""

    def _sanity(self, text: str) -> list[str]:
        from data_processing.itn import check_datetime_sanity

        return check_datetime_sanity(text)

    def test_well_formed_dates_and_times_pass(self) -> None:
        for text in (
            "الساعة 3:30 عصرًا",
            "12:00:59",
            "21/03/2024",
            "03/21/2024",  # M/D ordering is equally valid
            "2024-03-21",
            "2024-02-29",  # leap-generous on purpose
        ):
            self.assertEqual(self._sanity(text), [], text)

    def test_impossible_clock_fields_are_flagged(self) -> None:
        self.assertIn("25:70", self._sanity("الساعة 25:70")[0])
        self.assertIn("12:00:61", self._sanity("الساعة 12:00:61")[0])
        self.assertIn("3:70", self._sanity("मीटिंग 3:70 बजे है")[0])

    def test_citations_and_scores_are_not_clock_times(self) -> None:
        """Chapter:verse wears the same shape and holds values a clock cannot.

        "भजन 42:21" is a real Hindi manifest line; without a time cue nearby
        it used to be reported as an impossible hour.
        """
        self.assertEqual(self._sanity("जैसा कि हम भजन 42:21 में पढ़ते हैं"), [])
        self.assertEqual(self._sanity("البقرة 2:70"), [])
        self.assertEqual(self._sanity("البقرة 2:255"), [])
        self.assertEqual(self._sanity("النتيجة 3:1"), [])

    def test_hyphen_separated_reference_numbers_are_not_dates(self) -> None:
        """Every hyphen match in the real manifests was an ID, not a date."""
        self.assertEqual(self._sanity("وفي مقدمة القرار، 22-16-2015."), [])
        self.assertEqual(self._sanity("مجموعة صيدليات 19-0-11."), [])
        self.assertEqual(self._sanity("13-14-15"), [])

    def test_impossible_calendar_fields_are_flagged(self) -> None:
        self.assertIn("2024-13-45", self._sanity("2024-13-45")[0])
        self.assertIn("19/32/2024", self._sanity("19/32/2024")[0])

    def test_ambiguous_field_order_flags_only_when_no_reading_works(self) -> None:
        """21/03 vs 03/21 is unknowable here, so only flag broken-either-way."""
        self.assertEqual(self._sanity("31/12/2024"), [])  # valid as D/M
        self.assertEqual(self._sanity("12/31/2024"), [])  # valid as M/D
        self.assertTrue(self._sanity("31/31/2024"))  # valid as neither
        self.assertTrue(self._sanity("30/02/2024"))  # no February 30th either way

    def test_single_digit_second_field_never_matches(self) -> None:
        """Requiring two-digit minutes keeps scores and ratios out entirely."""
        self.assertEqual(self._sanity("المباراة انتهت 3:1"), [])
        self.assertEqual(self._sanity("النسبة 1:5"), [])

    def test_arabic_indic_digits_are_normalized_before_checking(self) -> None:
        flags = self._sanity("الساعة ٢٥:٧٠")
        self.assertEqual(len(flags), 1)
        self.assertIn("25:70", flags[0])

    def test_datetime_check_reaches_both_validator_flag_sets(self) -> None:
        from data_processing.itn import itn_soft_flags, numeric_soft_flags

        text = "الاجتماع الساعة 25:70"
        self.assertTrue(any("25:70" in f for f in numeric_soft_flags(text)))
        self.assertTrue(any("25:70" in f for f in itn_soft_flags(text)))

    def test_numeric_flags_exclude_the_arabic_word_check(self) -> None:
        """numeric_soft_flags is shared with zh/hi/ml, so it must be digit-only."""
        from data_processing.itn import itn_soft_flags, numeric_soft_flags

        text = "ثلاثة أربعة خمسة"
        self.assertTrue(itn_soft_flags(text, text))
        self.assertEqual(numeric_soft_flags(text, text), [])


class TestCodeSwitchPreservation(unittest.TestCase):
    """Words spoken in Latin must stay Latin (cleaner prompt rule 0)."""

    def _check(self, text: str, original: str) -> list[str]:
        from data_processing.text_utils import check_codeswitch_preservation

        return check_codeswitch_preservation(text, original)

    def test_transliterated_code_switch_is_flagged(self) -> None:
        flags = self._check(
            "عندنا اجتماع بعد اللانش بريك", "عندنا meeting بعد الـ lunch break"
        )
        self.assertEqual(len(flags), 1)
        self.assertIn("meeting", flags[0])

    def test_preserved_code_switch_is_not_flagged(self) -> None:
        self.assertEqual(
            self._check(
                "عندنا meeting بعد الـ lunch break.",
                "عندنا meeting بعد الـ lunch break",
            ),
            [],
        )

    def test_promoting_a_transliteration_to_latin_is_allowed(self) -> None:
        """The reverse direction is an allowed edit — gained tokens never flag."""
        self.assertEqual(self._check("قسم marketing", "قسم ماركتينج"), [])

    def test_removed_artifacts_are_not_mistaken_for_lost_words(self) -> None:
        """original_text is the raw manifest line, markers and markup included."""
        self.assertEqual(
            self._check("مرحبا بالعالم", "مرحبا [noise] بالعالم [music]"), []
        )
        self.assertEqual(self._check("مرحبا بالعالم", "مرحبا <br> بالعالم"), [])

    def test_case_and_trailing_punctuation_are_ignored(self) -> None:
        self.assertEqual(self._check("شركة Google.", "شركة google"), [])

    def test_latin_tokens_lowercases_and_dedupes_in_order(self) -> None:
        from data_processing.text_utils import latin_tokens

        self.assertEqual(
            latin_tokens("Vitamin C, vitamin c, OK"), ["vitamin", "c", "ok"]
        )


class TestScriptRatioFlags(unittest.TestCase):
    """The old absolute Arabic-ratio floor invited the violation rule 0 forbids."""

    def _flags(self, text: str, original: str) -> list[str]:
        from data_processing.validator import ValidatorAgent

        validator = ValidatorAgent(None, config=PipelineConfig())
        return validator._soft_flags(_cleaned_record(text, original))

    def test_code_switching_no_longer_reads_as_too_little_arabic(self) -> None:
        """At ~0.5 the old flag said "expected > 0.80" — and the only way for the
        LLM to comply was transliterating the Latin words."""
        text = "عندنا meeting بعد الـ lunch break"
        flags = self._flags(text, text)
        self.assertFalse(any("expected > 0.80" in f for f in flags), flags)
        self.assertFalse(any("transliterated or dropped" in f for f in flags), flags)

    def test_added_latin_is_judged_as_a_drop_against_the_original(self) -> None:
        flags = self._flags("هذا نص english text here now", "هذا نص عربي")
        self.assertTrue(any("Arabic ratio fell" in f for f in flags), flags)

    def test_lost_code_switch_is_flagged_though_the_ratio_rises(self) -> None:
        flags = self._flags(
            "عندنا اجتماع بعد اللانش بريك", "عندنا meeting بعد الـ lunch break"
        )
        self.assertTrue(any("meeting" in f for f in flags), flags)


class TestGenericValidatorFlags(unittest.TestCase):
    """The non-Arabic path ran no ITN or code-switch verification at all."""

    def _flags(self, text: str, original: str, language: str, **overrides):
        from data_processing.generic_validator import GenericValidatorAgent

        validator = GenericValidatorAgent(
            None, config=PipelineConfig(**overrides), language=language
        )
        return validator._soft_flags(_cleaned_record(text, original))

    def test_impossible_date_is_flagged_for_non_arabic(self) -> None:
        text = "बैठक 2024-13-45 को है"
        flags = self._flags(text, text, "hi")
        self.assertTrue(any("2024-13-45" in f for f in flags), flags)

    def test_itn_disabled_silences_the_numeric_flags(self) -> None:
        text = "बैठक 2024-13-45 को है"
        flags = self._flags(text, text, "hi", itn_enabled=False)
        self.assertFalse(any("2024-13-45" in f for f in flags), flags)

    def test_lost_english_is_flagged_in_chinese(self) -> None:
        flags = self._flags("我们讨论了开放沟通", "我们讨论了 open communication", "zh")
        self.assertTrue(any("open" in f for f in flags), flags)

    def test_english_skips_the_code_switch_check(self) -> None:
        """Every English token is Latin, so a lost word is an ordinary omission."""
        flags = self._flags("we discussed it", "we discussed the roadmap", "en")
        self.assertFalse(any("transliterated or dropped" in f for f in flags), flags)


class TestDateTimePrompts(unittest.TestCase):
    """What the speaker omitted is unrecoverable downstream, so the rules must
    reach the model that still has the original."""

    def test_arabic_itn_rule_covers_times_dates_and_calendars(self) -> None:
        system = build_cleaner_messages("نص")[0]["content"]
        for expected in ("TIMES:", "DATES:", "Hijri", "24-hour", "3:30"):
            self.assertIn(expected, system)

    def test_static_arabic_prompt_carries_the_same_guardrails(self) -> None:
        from data_processing.prompts import CLEANER_SYSTEM_PROMPT

        self.assertIn("Hijri", CLEANER_SYSTEM_PROMPT)
        self.assertIn("era marker", CLEANER_SYSTEM_PROMPT)

    def test_every_generic_language_gets_the_datetime_guardrail(self) -> None:
        """Generic rule 2 pointed at date conventions that did not exist."""
        for language in ("en", "zh", "hi", "ml", "fr"):  # fr exercises the fallback
            system = build_generic_cleaner_messages("text", language)[0]["content"]
            self.assertIn("never convert between calendar systems", system, language)

    def test_both_validators_are_told_impossible_values_are_fixable(self) -> None:
        from data_processing.prompts import VALIDATOR_SYSTEM_PROMPT

        self.assertIn("hour above 23", VALIDATOR_SYSTEM_PROMPT)
        generic = build_generic_validator_messages("o", "p", "zh", [])[0]["content"]
        self.assertIn("hour above 23", generic)


if __name__ == "__main__":
    unittest.main()
