import json
import sqlite3

import numpy as np
import pytest
import soundfile as sf
from test_code_switch import complete, corpus  # noqa: F401

from data_processing.code_switch.append import plan_append
from data_processing.code_switch.common import LANGS, digest
from data_processing.code_switch.index import assemble, check_frozen, run, verify
from qasr.code_switch import OnlineCodeSwitch


def new_manifest(root, name="new-ar.wav"):
    audio = root.parent / name
    sf.write(audio, np.full(5000, .12, dtype=np.float32), 8000)
    manifest = root.parent / "train_ar_addition.jsonl"
    manifest.write_text(json.dumps({"audio_filepath": str(audio), "text": "شنو بدك", "duration": .625}) + "\n")
    return manifest


def test_additions_reuse_old_pools_and_preserve_ids_without_old_audio(corpus):  # noqa: F811
    root, _frozen, old, tokenizer = complete(corpus)
    hashes = {lang: digest(root / "assembled" / f"{lang}.sqlite") for lang in LANGS}
    with sqlite3.connect(root / "assembled/ar.sqlite") as db:
        original_rows = db.execute("SELECT * FROM pool ORDER BY id").fetchall()
    for audio in root.parent.glob("*.wav"):
        audio.unlink()
    manifest = new_manifest(root)
    updated = root.parent / "updated"
    p = plan_append(updated, root / "assembled/index.json", [manifest])
    assert {c["source"] for c in p["chunks"]} == {str(manifest)}
    run(updated, procs=1)
    assert not verify(updated, jobs=1, metadata_only=True)
    result = assemble(updated, jobs=1)
    assert result["pools"]["ar"]["count"] == old["pools"]["ar"]["count"] + 1
    with sqlite3.connect(updated / "assembled/ar.sqlite") as db:
        assert db.execute("SELECT * FROM pool WHERE id<? ORDER BY id", (len(original_rows),)).fetchall() == original_rows
        assert "شنو" in json.loads(db.execute("SELECT record FROM pool WHERE id=?", (len(original_rows),)).fetchone()[0])["text"]
    for lang in LANGS:
        assert digest(root / "assembled" / f"{lang}.sqlite") == hashes[lang]
        if lang != "ar":
            assert (updated / "assembled" / f"{lang}.sqlite").is_symlink()
            assert result["pools"][lang] == old["pools"][lang]
    OnlineCodeSwitch(updated / "assembled/index.json", tokenizer, allow_pilot=True)
    with pytest.raises(ValueError, match="Frozen plan"):
        plan_append(updated, root / "assembled/index.json", [manifest])
    with pytest.raises(ValueError, match="already indexed"):
        plan_append(root.parent / "again", updated / "assembled/index.json", [manifest])


def test_append_rejects_changed_base_and_cross_language_collision(corpus):  # noqa: F811
    root, _frozen, _old, _tokenizer = complete(corpus)
    manifest = new_manifest(root, "en-1.wav")  # same identity as an existing English partner
    updated = root.parent / "updated"
    plan = plan_append(updated, root / "assembled/index.json", [manifest])
    run(updated, procs=1)
    assert not verify(updated, jobs=1, metadata_only=True)
    with pytest.raises(ValueError, match="retained language"):
        assemble(updated, jobs=1)
    assert not (updated / "assembled/index.json").exists()
    with (root / "assembled/en.sqlite").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Retained code-switch dependency changed"):
        check_frozen(updated, plan)
