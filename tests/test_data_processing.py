"""Tests for the data_processing package."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

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
    ManifestRecord,
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
        final, accepted = asyncio.run(validator.validate_with_retry(record))
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


if __name__ == "__main__":
    unittest.main()
