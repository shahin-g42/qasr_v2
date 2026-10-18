"""Validate an additive data contract against an immutable completed index."""

import copy
import hashlib
import json
import sqlite3
from pathlib import Path


def append_base(cfg, processor, source_index):
    from .data import data_contract

    source = Path(source_index).resolve()
    if source == Path(cfg["index_path"]).resolve():
        raise ValueError("Append requires a NEW index_path; the base is immutable")
    if not source.is_file() or source.with_name(source.name + ".scan.lock").exists():
        raise ValueError("Append requires a completed source index")
    with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as db:
        old = json.loads(db.execute("SELECT value FROM metadata WHERE key='index'").fetchone()[0])
    before, after = copy.deepcopy(old["data_contract"]), copy.deepcopy(data_contract(cfg))
    old_train, new_train = before.pop("train_manifest"), after.pop("train_manifest")
    if before != after or set(old_train) != set(new_train):
        raise ValueError("Append may only add training manifests; eval and filtering settings must match")
    additions = {}
    for lang, paths in old_train.items():
        proposed = new_train[lang]
        if proposed[:len(paths)] != paths or len(set(proposed)) != len(proposed):
            raise ValueError("Existing training manifests must remain an unchanged prefix without duplicates")
        additions[lang] = proposed[len(paths):]
    if not any(additions.values()):
        raise ValueError("No new training manifests to append")
    old_paths = {s["path"] for s in old["sources"]}
    new_paths = [p for paths in additions.values() for p in paths]
    if len(new_paths) != len(set(new_paths)) or old_paths.intersection(new_paths):
        raise ValueError("New manifests overlap existing sources or language groups")
    token_hash = hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if old["tokenizer_sha256"] != token_hash:
        raise ValueError("Tokenizer changed; cannot append to this index")
    return source, old, additions
