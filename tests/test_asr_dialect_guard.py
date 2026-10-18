import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from data_processing.asr_clean.dialect import formalizations
from data_processing.asr_clean.worker import ChunkProcessor, Item, RunPaths, WorkerConfig


def test_formalization_is_token_and_span_aware():
    assert formalizations("شو بدك تعمل هلق", "ماذا تريد أن تفعل الآن")
    assert formalizations("بيعمل كده عشان الفلوس", "يعمل هكذا من أجل الأموال")
    assert not formalizations("مدري ايه عمري ما اشتغلت", "مدري إيه عمري ما اشتغلت")
    assert not formalizations("ما تناديني ايه", "ما تناديني أي")  # existing ما is outside changed span
    assert not formalizations("بدي شوفك هلق", "بِدّي شوفك هلّق.")
    assert not formalizations("مشوار", "طريق")  # no substring وش


class FormalizingLLM:
    def chat(self, messages, **kwargs):
        assert "Formal grammar or fluency is NOT" in messages[0]["content"]
        return json.dumps([{"i": 0, "t": "ماذا تريد أن تفعل الآن", "s": "a"}]), {}


def test_campaign_quarantines_formal_asr_instead_of_publishing_it():
    cfg = WorkerConfig("unused", [], dialect_guard="quarantine", max_item_attempts=2)
    item = Item(0, 0, org_text="شو بدك تعمل هلق", asr_text="ماذا تريد أن تفعل الآن", source_dialect="SY")
    with ThreadPoolExecutor(1) as pool:
        processor = ChunkProcessor(cfg, None, FormalizingLLM(), pool, pool)
        processor._correct([item], "ar", "adjudicate")
    assert item.reject == "guard_dialect_erasure"
    assert not item.text
    assert "هلق" in item.detail
    assert item.llm_raw


@pytest.mark.parametrize("policy", [None, {"dialect_guard": "off"}])
def test_guard_cannot_resume_a_chunk_with_legacy_or_different_policy(tmp_path, policy):
    paths = RunPaths(str(tmp_path))
    chunk = {"id": "0000000", "start": 0}
    progress = {"status": "done"}
    if policy is not None:
        progress["policy"] = policy
    (paths.progress / "0000000.json").write_text(json.dumps(progress))
    cfg = WorkerConfig(str(tmp_path), [], dialect_guard="quarantine")
    with ThreadPoolExecutor(1) as pool:
        processor = ChunkProcessor(cfg, None, None, pool, pool)
        with pytest.raises(ValueError, match="new run"):
            processor.run(chunk, paths, None)
