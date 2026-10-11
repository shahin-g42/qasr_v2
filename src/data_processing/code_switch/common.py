from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

VERSION = 2
PAIRS = ('ar-en', 'hi-en', 'ml-en', 'zh-en', 'ar-hi', 'ar-ml', 'ar-zh')
LANGS = ('ar', 'en', 'hi', 'ml', 'zh')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


@contextlib.contextmanager
def lock(path, *, shared=False):
    """Persistent lock inode: never unlink it, including after a crash.

    Requires cross-node POSIX flock support on the shared mount. No timed
    lease stealing: an expired heartbeat does not confer ownership.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as f:
        fcntl.flock(f.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_plan(root):
    plan = read_json(Path(root) / '_state/plan.json')
    if not plan or plan.get('version') != VERSION:
        raise ValueError('No supported frozen plan; run plan first with this RUN_ID')
    if plan['root'] != str(Path(root).resolve()):
        raise ValueError('Run directory moved: frozen output paths would be invalid')
    return plan


def chunk_paths(root, chunk):
    root = Path(root)
    cid = chunk['id']
    return (root / 'parts' / f'{cid}.jsonl', root / '_rejects' / f'{cid}.jsonl',
            root / '_state/progress' / f'{cid}.json', root / 'audio' / cid)
