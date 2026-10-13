import json
from collections import Counter
from pathlib import Path

from qasr_lab.io import file_hash
from qasr_lab.manifest_sampling import sample_record_lines


def test_samples_records_without_parsing_and_reuses_cache_without_reading_source(
    tmp_path, monkeypatch
):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": i, "text": "مرحبا" * (i % 20 + 1)}) for i in range(1000))
        + "\n\n"
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            json,
            "loads",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("Unexpected JSON parsing")
            ),
        )
        sampled, first = sample_record_lines(
            source, limit=12, seed=17, cache_directory=tmp_path / "cache"
        )
    assert len(sampled) == len({row["line_number"] for row in sampled}) == 12
    assert first["records_read"] == 1000
    assert first["sha256"] == file_hash(source)
    assert not first["all_records_considered"]
    original_open = Path.open
    with monkeypatch.context() as patch:

        def no_source_open(path, *args, **kwargs):
            assert path != source, "A warm cache must not read the source manifest"
            return original_open(path, *args, **kwargs)

        patch.setattr(Path, "open", no_source_open)
        repeated, second = sample_record_lines(
            source, limit=12, seed=17, cache_directory=tmp_path / "cache"
        )
    assert repeated == sampled and second["cache_hit"]
    with source.open("rb") as handle:
        for row in sampled:
            handle.seek(row["byte_offset"])
            assert handle.readline().decode() == row["record_json"]
            assert json.loads(row["record_json"])["id"] == row["line_number"] - 1


def test_uniform_record_sampling_does_not_favor_long_lines(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text(
        "\n".join(json.dumps({"id": i, "text": "x" * (10000 if i == 0 else 1)}) for i in range(10))
        + "\n"
    )
    counts = Counter()
    for seed in range(300):
        sampled, _ = sample_record_lines(
            source, limit=2, seed=seed, cache_directory=tmp_path / "cache"
        )
        counts.update(json.loads(row["record_json"])["id"] for row in sampled)
    # Broad fixed-seed check: a 10 KB row must not dominate nine short records.
    assert all(25 <= counts[index] <= 100 for index in range(10))


def test_source_changes_invalidate_cache_and_small_sources_are_complete(tmp_path):
    source = tmp_path / "records.jsonl"
    source.write_text('{"id": 1}\n\n')
    rows, first = sample_record_lines(source, limit=5, seed=42, cache_directory=tmp_path / "cache")
    assert first["all_records_considered"] and len(rows) == 1
    source.write_text('{"id": 2}\n\n')
    updated, second = sample_record_lines(
        source, limit=5, seed=42, cache_directory=tmp_path / "cache"
    )
    assert first["cache_path"] != second["cache_path"]
    assert not second["cache_hit"] and json.loads(updated[0]["record_json"])["id"] == 2
