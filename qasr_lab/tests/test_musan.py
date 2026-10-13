from pathlib import Path

import pytest

from qasr_lab.io import file_hash
from qasr_lab.musan import noise_directory, prepare_musan_noise
from qasr_lab.panel import identity_keys

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")


def make_corpus(root):
    for category in ("noise", "speech", "music"):
        (root / category / "fixture").mkdir(parents=True)
    rng = np.random.default_rng(17)
    for index in range(20):
        sf.write(root / "noise/fixture" / f"{index}.wav", rng.normal(0, 0.1, 4800), 16000)
    # These are deliberately invalid audio: reading either means we selected an unsafe partition.
    (root / "speech/fixture/never.wav").write_bytes(b"speech must not be read")
    (root / "music/fixture/never.wav").write_bytes(b"music must not be read")
    (root / "noise/fixture/duplicate.wav").write_bytes((root / "noise/fixture/0.wav").read_bytes())


def test_noise_partition_split_caps_crops_and_source_integrity(tmp_path):
    root = tmp_path / "musan"
    make_corpus(root)
    before = {path: file_hash(path) for path in root.rglob("*.wav")}
    kwargs = dict(languages=["en", "ar"], seed=42, min_seconds=0.1, max_seconds=0.2)
    panels, originals, provenance = prepare_musan_noise(
        root, tmp_path / "out", per_language=2, **kwargs
    )
    assert len(panels["train"]) == len(panels["validation"]) == 4
    assert provenance["counts"]["duplicate_files"] == 1
    assert provenance["unique_eligible_recordings"] == 20
    train_keys = set().union(*(identity_keys(row, "/") for row in panels["train"]))
    for row in panels["validation"]:
        assert not train_keys.intersection(identity_keys(row, "/"))
    for split in panels:
        assert {row["language"] for row in panels[split]} == {"ar", "en"}
        for row in panels[split]:
            assert row["text"] == "" and row["duration"] == 0.2
            assert sf.info(row["audio_filepath"]).frames == 3200
            assert row["audio_sha256"] == file_hash(row["audio_filepath"])
        assert all(
            Path(row["audio_filepath"]).is_relative_to(root / "noise") for row in originals[split]
        )
    repeated, _, _ = prepare_musan_noise(root, tmp_path / "repeat", per_language=3, **kwargs)
    # Changing the cap does not move recordings between splits or change selected crops.
    assert {row["recording_id"] for row in repeated["train"]}.isdisjoint(
        row["recording_id"] for row in repeated["validation"]
    )
    for split in panels:
        assert [(r["id"], r["audio_sha256"]) for r in panels[split]] == [
            (r["id"], r["audio_sha256"]) for r in repeated[split][:4]
        ]
    assert all(file_hash(path) == sha for path, sha in before.items())


def test_missing_partition_and_original_overlap_fail(tmp_path):
    root = tmp_path / "musan"
    make_corpus(root)
    with pytest.raises(ValueError, match="noise/ partition"):
        noise_directory(root / "music")
    keys = {("path", str(path.resolve())): {"speech-id"} for path in root.rglob("*.wav")}
    with pytest.raises(ValueError, match="overlaps configured validation"):
        prepare_musan_noise(
            root,
            tmp_path / "overlap",
            languages=["en"],
            per_language=2,
            seed=42,
            min_seconds=0.1,
            max_seconds=0.2,
            protected=keys,
        )


def test_duplicate_only_corpus_cannot_leak_into_two_splits(tmp_path):
    noise = tmp_path / "noise"
    noise.mkdir()
    sf.write(noise / "one.wav", np.zeros(16000), 16000)
    (noise / "copy.wav").write_bytes((noise / "one.wav").read_bytes())
    with pytest.raises(ValueError, match="two distinct"):
        prepare_musan_noise(
            noise,
            tmp_path / "out",
            languages=["en"],
            per_language=2,
            seed=42,
            min_seconds=0.1,
            max_seconds=2,
        )
