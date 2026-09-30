"""Exhaustive, metadata-only ingestion and disk-backed worksets.

All artifact references are relative to the caller's run root; source paths are
absolute. ``contract_index:byte_offset`` identifies an occurrence (offsets in
uncompressed bytes for gzip). Raw lines, including their terminators, are saved
as base64. Completion reports are the commit boundary; orphan attempt files are
never discovered by globbing. Buffers hold at most one input record plus fixed
I/O buffers, and SQLite transactions are bounded by rows and serialized bytes.
An individual exceptionally large JSONL record necessarily needs its own memory.

The production driver, not this reusable module, gates the approved 26/11 list.
No API in this module opens audio, invokes an LLM, or applies a sample cap.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import os
import re
import socket
import sqlite3
import uuid
from collections import Counter, OrderedDict
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

import yaml

from .accent import detect
from .datasets.base import DatasetSpec, Kind
from .normalize import DiacriticPolicy, normalize
from .quality import gate

SCHEMA_VERSION = 1
CHUNK_ROWS = 8192
CHUNK_BYTES = 8 * 1024 * 1024
TRANSACTION_ROWS = 1000
TRANSACTION_BYTES = 4 * 1024 * 1024
MAX_OPEN_PARTS = 16
_CLUSTER = Path('/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr')
_DECLARED_ROOTS = {
    'v76': _CLUSTER / 'training_manifests/v7.6',
    'sft': _CLUSTER / 'q3asr_sft_manifests',
}
_LANGUAGE_NAMES = {'arabic': 'ar', 'english': 'en', 'hindi': 'hi',
                   'malayalam': 'ml', 'chinese': 'zh'}
# Same envelope grammar as scripts/prepare_q3asr_filter.py; importing that
# command would load unrelated LID/probing dependencies and change sys.path.
_ENVELOPE = re.compile(r'^language\s+(?P<lang>.+?)<asr_text>(?P<text>.*)$', re.DOTALL)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path: Path) -> None:
    if not path.exists():
        _mkdir(path.parent)
        path.mkdir(exist_ok=True)
        _sync_dir(path.parent)


def _atomic_json(path: Path, value: object) -> None:
    _mkdir(path.parent)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('xb') as handle:
        handle.write((_json(value) + '\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def _read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def _digest(path: Path, start: int = 0, end: int | None = None) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        handle.seek(start)
        remaining = (path.stat().st_size if end is None else end) - start
        if remaining < 0:
            raise ValueError('invalid byte range')
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise OSError(f'short read: {path}')
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _artifact(root: Path, reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'nonportable artifact reference: {reference}')
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'artifact escapes run root: {reference}')
    return path


def _stat(path: Path) -> dict:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f'not a regular input file: {path}')
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino, 'device': stat.st_dev}


@contextmanager
def _owner(path: Path, run_id: str):
    """Prefer the shared owner implementation once that module is available."""
    try:
        from .run_state import RunOwner
    except ModuleNotFoundError as exc:
        if exc.name != 'data_processing.run_state':
            raise
    else:
        with RunOwner(path, run_id):
            yield
        return
    _mkdir(path.parent)
    path.mkdir()  # Never steal a crashed owner's directory.
    _sync_dir(path.parent)
    token = uuid.uuid4().hex
    _atomic_json(path / 'owner.json', {'token': token, 'run_id': run_id,
                                     'pid': os.getpid(), 'host': socket.gethostname()})
    try:
        yield
    finally:
        if _read_json(path / 'owner.json')['token'] != token:
            raise RuntimeError('owner identity changed')
        (path / 'owner.json').unlink()
        path.rmdir()
        _sync_dir(path.parent)


class _JsonlWriter:
    def __init__(self, root: Path, path: Path):
        self.root, self.path = root, path
        _mkdir(path.parent)
        self.temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
        self.handle = self.temporary.open('xb', buffering=64 * 1024)
        self.digest = hashlib.sha256()
        self.rows = self.bytes = 0

    def write(self, row: dict) -> None:
        data = (_json(row) + '\n').encode()
        self.handle.write(data)
        self.digest.update(data)
        self.rows += 1
        self.bytes += len(data)

    def finish(self) -> dict:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        if self.path.exists():
            raise FileExistsError(f'immutable output already exists: {self.path}')
        os.replace(self.temporary, self.path)
        _sync_dir(self.path.parent)
        return {'path': _relative(self.root, self.path), 'rows': self.rows,
                'bytes': self.bytes, 'sha256': self.digest.hexdigest()}

    def close(self) -> None:
        self.handle.close()


def _verified_rows(root: Path, description: dict) -> Iterator[dict]:
    digest, count, size = hashlib.sha256(), 0, 0
    with _artifact(root, description['path']).open('rb') as handle:
        for line in handle:
            digest.update(line)
            count += 1
            size += len(line)
            yield json.loads(line)
    if (count, size, digest.hexdigest()) != (
        description['rows'], description['bytes'], description['sha256']
    ):
        raise RuntimeError(f'committed JSONL changed: {description["path"]}')


def _verify_file(root: Path, description: dict) -> None:
    path = _artifact(root, description['path'])
    if path.stat().st_size != description['bytes'] or _digest(path) != description['sha256']:
        raise RuntimeError(f'committed artifact changed: {path}')


def _language(lang: str) -> str:
    if not isinstance(lang, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', lang):
        raise ValueError(f'unsafe language key: {lang!r}')
    return lang


def load_contract(contract_path, v76_root, sft_root) -> list[dict]:
    """Return exact file specs, train then eval, sorted language/list order.

    Absolute entries must be inside the supplied roots or the exact approved
    cluster roots. Portable entries must start with training_manifests/v7.6/
    or q3asr_sft_manifests/. Traversal, globbing, ambiguous roots, duplicate
    declarations, symlink escapes, and missing declared files fail closed.
    """
    roots = {'v76': Path(v76_root).resolve(), 'sft': Path(sft_root).resolve()}
    if roots['v76'].is_relative_to(roots['sft']) or roots['sft'].is_relative_to(roots['v76']):
        raise ValueError('input roots must be disjoint')
    contract = _read_yaml(Path(contract_path))
    specs, seen = [], set()
    for split in ('train', 'eval'):
        mapping = contract.get(f'{split}_manifest')
        if not isinstance(mapping, dict):
            raise ValueError(f'{split}_manifest must be a language mapping')
        for lang in sorted(mapping):
            _language(lang)
            entries = mapping[lang]
            if not isinstance(entries, list):
                raise ValueError('manifest entries must be explicit lists')
            for entry in entries:
                if not isinstance(entry, str) or any(c in entry for c in '*?[]\x00'):
                    raise ValueError(f'invalid explicit manifest path: {entry!r}')
                declared = Path(entry)
                if '..' in declared.parts:
                    raise ValueError(f'manifest traversal: {entry}')
                matches = []
                for tree, root in roots.items():
                    prefixes = (root, _DECLARED_ROOTS[tree]) if declared.is_absolute() else (
                        Path('training_manifests/v7.6') if tree == 'v76'
                        else Path('q3asr_sft_manifests'),
                    )
                    for prefix in prefixes:
                        if declared.is_relative_to(prefix):
                            matches.append((tree, declared.relative_to(prefix)))
                            break
                if len(matches) != 1:
                    raise ValueError(f'manifest outside declared roots: {entry}')
                tree, relative = matches[0]
                if len(relative.parts) < 2 or relative.parts[0] != lang:
                    raise ValueError(f'manifest language/path mismatch: {entry}')
                if not (entry.endswith('.jsonl') or entry.endswith('.jsonl.gz')):
                    raise ValueError(f'unsupported manifest format: {entry}')
                path = (roots[tree] / relative).resolve(strict=True)
                if not path.is_relative_to(roots[tree]):
                    raise ValueError(f'manifest symlink escapes root: {entry}')
                _stat(path)
                if path in seen:
                    raise ValueError(f'duplicate manifest declaration: {entry}')
                seen.add(path)
                specs.append({'contract_index': len(specs), 'split': split, 'lang': lang,
                              'tree': tree, 'source': f'internal_{tree}_{lang}',
                              'declared_path': entry, 'relative_path': relative.as_posix(),
                              'path': str(path)})
    if not any(spec['split'] == 'train' for spec in specs):
        raise ValueError('contract has no train inputs')
    return specs


def _read_yaml(path: Path) -> dict:
    class UniqueLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            result = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if key in result:
                    raise ValueError(f'duplicate YAML key: {key}')
                result[key] = self.construct_object(value_node, deep=deep)
            return result

    with path.open(encoding='utf-8') as handle:
        result = yaml.load(handle, Loader=UniqueLoader)
    if not isinstance(result, dict):
        raise ValueError('contract must be a YAML mapping')
    return result


def input_inventory(contract_path, v76_root, sft_root) -> dict:
    """Initial stat inventory; content hashes are streamed during preparation."""
    specs = load_contract(contract_path, v76_root, sft_root)
    return {'schema_version': SCHEMA_VERSION, 'contract_path': str(Path(contract_path).resolve()),
            'contract_sha256': _digest(Path(contract_path)),
            'roots': {'v76': str(Path(v76_root).resolve()), 'sft': str(Path(sft_root).resolve())},
            'files': [{**spec, 'stat': _stat(Path(spec['path']))} for spec in specs]}


def _code_identity() -> dict:
    directory = Path(__file__).parent
    return {name: _digest(directory / name) for name in
            ('workset.py', 'datasets/base.py', 'normalize.py', 'quality.py', 'accent.py')}


def _plan_segments(files: list[dict], nodes: int, jobs: int) -> list[list[dict]]:
    slots = nodes * jobs
    plans = [[] for _ in range(slots)]
    train = [f for f in files if f['split'] == 'train']
    total = sum(f['stat']['size'] for f in train if not f['path'].endswith('.gz'))
    position = 0
    for spec in train:
        size = spec['stat']['size']
        if spec['path'].endswith('.gz') or size == 0:
            plans[spec['contract_index'] % slots].append({'spec': spec, 'start': 0, 'end': size})
            continue
        for slot in range(slots):
            start = max(total * slot // slots - position, 0)
            end = min(total * (slot + 1) // slots - position, size)
            if start < end:
                plans[slot].append({'spec': spec, 'start': start, 'end': end})
        position += size
    return plans


def _valid_string(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


def _canonical_row(raw: bytes, spec: dict, offset: int) -> dict:
    row = {'row_id': f'{spec["contract_index"]}:{offset}',
           'contract_index': spec['contract_index'], 'byte_offset': offset,
           'source_file': spec['path'], 'source': spec['source'], 'lang': spec['lang'],
           'split': spec['split'], 'raw_base64': base64.b64encode(raw).decode('ascii'),
           'raw_bytes': len(raw), 'audio_filepath': None, 'id': None, 'text': '',
           'normalized_text': '', 'duration': None, 'quality': 0.0, 'accent': 'unknown',
           'metadata_errors': [], 'repairable_flags': [], 'normalization': []}
    errors = row['metadata_errors']
    if not raw.strip():
        errors.append('blank_line')
        return row
    try:
        parsed = json.loads(raw.decode('utf-8'))
    except (ValueError, UnicodeError, RecursionError):
        errors.append('malformed_json')
        return row
    if not isinstance(parsed, dict):
        errors.append('not_an_object')
        return row
    adapter = DatasetSpec(spec['source'], spec['lang'], Kind.LOCAL_JSONL, 'internal')
    # Strict type checks before the permissive legacy adapters stringify values.
    mapped = dict(parsed)
    path = parsed.get('audio_filepath', parsed.get('wav_path'))
    if path is None:
        path = parsed.get('audio')
        if isinstance(path, dict):
            path = path.get('path') or path.get('audio_file_name')
    if _valid_string(path) and '\x00' not in path:
        mapped['audio_filepath'] = path
        row['audio_filepath'] = adapter.audio_filepath(mapped)
        row['id'] = hashlib.sha256(path.encode('utf-8')).hexdigest()
    else:
        errors.append('missing_or_invalid_path')
    text = parsed.get('text')
    if text is None:
        text = parsed.get('transcript', parsed.get('sentence'))
    declared_language = parsed.get('lang', parsed.get('language'))
    if isinstance(text, str):
        envelope = _ENVELOPE.match(text.strip())
        if envelope:
            declared_language = envelope.group('lang').strip()
            text = envelope.group('text').strip()
            row['normalization'].append('sft_envelope')
        elif '<asr_text>' in text or text.startswith('language '):
            row['repairable_flags'].append('unrecognized_text_envelope')
    if declared_language is not None:
        label = str(declared_language).strip().lower()
        label = _LANGUAGE_NAMES.get(label, label)
        if label not in ('', 'unknown', 'none', 'null', spec['lang']):
            errors.append('language_mismatch')
    if _valid_string(text):
        mapped['text'] = text
        row['text'] = adapter.text_of(mapped)
        normalized, rules = normalize(text, spec['lang'], DiacriticPolicy.PRESERVE_ALL)
        row['normalization'].extend(rules)
        if not normalized.strip():
            normalized = text
            row['repairable_flags'].append('empty_after_normalization')
        row['normalized_text'] = normalized
        row['accent'] = detect(normalized, spec['lang']).label
    else:
        errors.append('missing_or_invalid_text')
    try:
        duration = adapter.duration_of(parsed)
    except OverflowError:
        duration = None
    if parsed.get('duration') is None:
        errors.append('missing_duration')
    elif isinstance(parsed['duration'], bool) or duration is None or not math.isfinite(duration) or duration <= 0:
        errors.append('invalid_duration')
    else:
        row['duration'] = duration
    result = gate(row['normalized_text'], spec['lang'], row['duration'], require_duration=False)
    row['repairable_flags'].extend(result.reasons)
    row['quality'] = result.quality if result.ok else 0.0
    quality = parsed.get('quality', parsed.get('quality_score'))
    if quality is not None:
        if isinstance(quality, (float, int)) and not isinstance(quality, bool) and 0 <= quality <= 1 and math.isfinite(quality):
            row['quality'] = float(quality)
        else:
            row['repairable_flags'].append('invalid_quality')
    return row


def _prepare_segment(root: Path, directory: Path, segment: dict, number: int) -> dict:
    spec, start, end = segment['spec'], segment['start'], segment['end']
    path = Path(spec['path'])
    if _stat(path) != spec['stat']:
        raise RuntimeError(f'input changed before read: {path}')
    compressed = path.suffix == '.gz'
    digest = hashlib.sha256()
    chunks, counters = [], Counter(rows=0, source_invalid=0, repairable=0)
    prefix = directory / f's{number:04d}'
    quarantine = _JsonlWriter(root, prefix / 'quarantine.jsonl')
    writer = None
    try:
        with (gzip.open(path, 'rb') if compressed else path.open('rb')) as handle:
            if start and not compressed:
                handle.seek(start - 1)
                if handle.read(1) != b'\n':
                    handle.readline()
            actual_start = handle.tell()
            while compressed or handle.tell() < end:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                digest.update(raw)
                row = _canonical_row(raw, spec, offset)
                counters['rows'] += 1
                counters['source_invalid'] += bool(row['metadata_errors'])
                counters['repairable'] += bool(row['repairable_flags'])
                for reason in row['metadata_errors']:
                    counters[reason] += 1
                if writer is None:
                    writer = _JsonlWriter(root, prefix / f'rows{len(chunks):06d}.jsonl')
                writer.write(row)
                if row['metadata_errors']:
                    quarantine.write(row)
                if writer.rows >= CHUNK_ROWS or writer.bytes >= CHUNK_BYTES:
                    chunks.append(writer.finish())
                    writer = None
            actual_end = handle.tell()
        if writer is not None:
            chunks.append(writer.finish())
        quarantine_info = quarantine.finish()
    finally:
        quarantine.close()
        if writer is not None:
            writer.close()
    content_sha = _digest(path) if compressed else digest.hexdigest()
    if _stat(path) != spec['stat']:
        raise RuntimeError(f'input changed during read: {path}')
    return {'contract_index': spec['contract_index'], 'source': spec['source'],
            'lang': spec['lang'], 'split': spec['split'], 'path': spec['path'],
            'planned_start': start, 'planned_end': end, 'start': actual_start, 'end': actual_end,
            'offset_space': 'uncompressed' if compressed else 'file',
            'sha256': digest.hexdigest(), 'content_sha256': content_sha,
            'chunks': chunks, 'quarantine': quarantine_info, 'counts': dict(counters)}


def _prepare_task(arguments: tuple) -> list[dict]:
    root, directory, segments = arguments
    return [_prepare_segment(root, directory, segment, i) for i, segment in enumerate(segments)]


def _check_resume(root: Path, report: dict, identity: dict) -> None:
    if report.get('status') != 'complete' or report.get('identity') != identity:
        raise RuntimeError('incompatible or incomplete preparation report')
    body = {k: v for k, v in report.items() if k != 'fingerprint'}
    if report.get('fingerprint') != _fingerprint(body):
        raise RuntimeError('preparation report fingerprint changed')
    for segment in report['segments']:
        for descriptor in [*segment['chunks'], segment['quarantine']]:
            _verify_file(root, descriptor)
        path = Path(segment['path'])
        checksum = (_digest(path) if segment['offset_space'] == 'uncompressed'
                    else _digest(path, segment['start'], segment['end']))
        if checksum != segment['content_sha256']:
            raise RuntimeError(f'input fingerprint changed: {path}')
    if report.get('eval_index'):
        _verify_file(root, report['eval_index'])


def prepare_rank(root, *, contract_path, v76_root, sft_root, rank, nodes,
                 jobs=80, run_id='default') -> dict:
    """Prepare only this rank's train byte ranges; rank zero also indexes eval.

    ``jobs`` actual task buckets (up to available input bytes) are distributed
    over a process pool of that size. Gzip inputs are read serially and split
    into bounded durable output chunks, never seek-partitioned. Resume requires
    a complete compatible report and revalidates all referenced checksums.
    """
    if nodes < 1 or not 0 <= rank < nodes or jobs < 1:
        raise ValueError('invalid rank/nodes/jobs')
    root = Path(root).resolve()
    with _owner(root / 'owners' / f'prepare-rank{rank}', run_id):
        inventory = input_inventory(contract_path, v76_root, sft_root)
        identity = {'schema_version': SCHEMA_VERSION, 'run_id': run_id, 'rank': rank,
                    'nodes': nodes, 'jobs': jobs, 'inventory': inventory, 'code': _code_identity()}
        marker = root / 'prepare' / f'rank{rank}.json'
        if marker.exists():
            previous = _read_json(marker)
            if previous.get('status') == 'complete':
                _check_resume(root, previous, identity)
                if input_inventory(contract_path, v76_root, sft_root) != inventory:
                    raise RuntimeError('input inventory changed during resume validation')
                return previous
            if previous.get('identity') != identity:
                raise RuntimeError('cannot reuse failed preparation with different inputs')
        directory = root / 'prepare' / f'rank{rank}-{uuid.uuid4().hex}'
        plans = _plan_segments(inventory['files'], nodes, jobs)
        mine = [plans[rank + worker * nodes] for worker in range(jobs)]
        if rank == 0:
            for i, spec in enumerate(f for f in inventory['files'] if f['split'] == 'eval'):
                mine[i % jobs].append({'spec': spec, 'start': 0, 'end': spec['stat']['size']})
        tasks = [(root, directory / f'w{i:04d}', spans) for i, spans in enumerate(mine) if spans]
        report = {'status': 'fatal', 'stage': 'prepare', 'schema_version': SCHEMA_VERSION,
                  'identity': identity, 'run_id': run_id, 'rank': rank, 'nodes': nodes,
                  'segments': [], 'actual_tasks': len(tasks)}
        try:
            if jobs == 1 or len(tasks) <= 1:
                outputs = map(_prepare_task, tasks)
                for segments in outputs:
                    report['segments'].extend(segments)
            else:
                with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
                    for segments in executor.map(_prepare_task, tasks):
                        report['segments'].extend(segments)
            report['segments'].sort(key=lambda s: (s['contract_index'], s['planned_start']))
            if input_inventory(contract_path, v76_root, sft_root) != inventory:
                raise RuntimeError('input inventory changed during preparation')
            counts = Counter(rows=0, source_invalid=0, repairable=0)
            for segment in report['segments']:
                counts.update(segment['counts'])
            report['counts'] = dict(counts)
            report['train_rows'] = sum(s['counts'].get('rows', 0) for s in report['segments'] if s['split'] == 'train')
            report['eval_rows'] = counts['rows'] - report['train_rows']
            if rank == 0:
                report['eval_index'] = _build_eval(root, directory, report['segments'])
            report['status'] = 'complete'
            report['fingerprint'] = _fingerprint(report)
            _atomic_json(marker, report)
            return report
        except BaseException as exc:
            report['status'] = 'fatal'
            report['error'] = type(exc).__name__
            _atomic_json(marker, report)
            raise


def _connect(path: Path) -> sqlite3.Connection:
    _mkdir(path.parent)
    db = sqlite3.connect(path, isolation_level=None)
    try:
        if db.execute('PRAGMA journal_mode=DELETE').fetchone()[0] != 'delete':
            raise RuntimeError('SQLite DELETE journal unavailable')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA temp_store=FILE')
        db.execute('PRAGMA cache_size=-8192')
        if db.execute('PRAGMA synchronous').fetchone()[0] != 2:
            raise RuntimeError('SQLite FULL durability unavailable')
    except BaseException:
        db.close()
        raise
    return db


def _readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, isolation_level=None)


class _Transactions:
    def __init__(self, db):
        self.db, self.rows, self.bytes = db, 0, 0
        db.execute('BEGIN IMMEDIATE')

    def tick(self, size: int = 0):
        self.rows += 1
        self.bytes += size
        if self.rows >= TRANSACTION_ROWS or self.bytes >= TRANSACTION_BYTES:
            self.db.execute('COMMIT')
            self.db.execute('BEGIN IMMEDIATE')
            self.rows = self.bytes = 0

    def finish(self):
        self.db.execute('COMMIT')


def _db_description(root: Path, path: Path) -> dict:
    with path.open('rb') as handle:
        os.fsync(handle.fileno())
    _sync_dir(path.parent)
    return {'path': _relative(root, path), 'bytes': path.stat().st_size,
            'sha256': _digest(path)}


def _build_eval(root: Path, directory: Path, segments: list[dict]) -> dict:
    path = directory / 'eval.sqlite3'
    count = invalid = 0
    with closing(_connect(path)) as db:
        db.executescript('''
            CREATE TABLE exclusions (audio_filepath TEXT PRIMARY KEY) WITHOUT ROWID;
            CREATE TABLE occurrences (row_id TEXT PRIMARY KEY, audio_filepath TEXT,
                                      payload TEXT NOT NULL) WITHOUT ROWID;
            CREATE INDEX eval_path ON occurrences(audio_filepath);
        ''')
        transactions = _Transactions(db)
        for segment in segments:
            if segment['split'] != 'eval':
                continue
            for chunk in segment['chunks']:
                for row in _verified_rows(root, chunk):
                    payload = _json(row)
                    audio = row['audio_filepath']
                    db.execute('INSERT INTO occurrences VALUES (?,?,?)',
                               (row['row_id'], audio, payload))
                    if audio is not None:
                        db.execute('INSERT OR IGNORE INTO exclusions VALUES (?)', (audio,))
                    count += 1
                    invalid += bool(row['metadata_errors'])
                    transactions.tick(len(payload))
        transactions.finish()
        unique = db.execute('SELECT COUNT(*) FROM exclusions').fetchone()[0]
    return {**_db_description(root, path), 'rows': count, 'unique': unique,
            'source_invalid': invalid}


def _prepared_reports(root: Path, run_id: str | None = None) -> list[dict]:
    first = _read_json(root / 'prepare/rank0.json')
    nodes = first['nodes']
    reports = [first] + [_read_json(root / 'prepare' / f'rank{r}.json') for r in range(1, nodes)]
    common = {k: v for k, v in first['identity'].items() if k != 'rank'}
    if run_id is not None and common['run_id'] != run_id:
        raise RuntimeError('workset run_id differs from preparation')
    files, jobs = common['inventory']['files'], common['jobs']
    plans = _plan_segments(files, nodes, jobs)
    for rank, report in enumerate(reports):
        body = {k: v for k, v in report.items() if k != 'fingerprint'}
        if report['status'] != 'complete' or report.get('fingerprint') != _fingerprint(body):
            raise RuntimeError('preparation barrier contains an incomplete/corrupt report')
        if report['rank'] != rank or common != {k: v for k, v in report['identity'].items() if k != 'rank'}:
            raise RuntimeError('incompatible rank reports')
        expected = [(s['spec']['contract_index'], s['start'], s['end'])
                    for slot in range(rank, nodes * jobs, nodes) for s in plans[slot]]
        if rank == 0:
            expected.extend((f['contract_index'], 0, f['stat']['size']) for f in files if f['split'] == 'eval')
        actual = [(s['contract_index'], s['planned_start'], s['planned_end']) for s in report['segments']]
        if sorted(actual) != sorted(expected):
            raise RuntimeError('preparation segment coverage mismatch')
    return reports


def _languages(reports: list[dict]) -> list[str]:
    return sorted({f['lang'] for f in reports[0]['identity']['inventory']['files'] if f['split'] == 'train'})


def _workset_schema(db):
    db.executescript('''
        CREATE TABLE occurrences (
            row_id TEXT PRIMARY KEY, clip_id TEXT, audio_filepath TEXT,
            contract_index INTEGER NOT NULL, byte_offset INTEGER NOT NULL,
            source TEXT NOT NULL, source_invalid INTEGER NOT NULL,
            payload TEXT NOT NULL, disposition TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX occurrence_path ON occurrences(audio_filepath, contract_index, byte_offset);
        CREATE INDEX occurrence_clip ON occurrences(clip_id);
        CREATE TABLE clips (
            id TEXT PRIMARY KEY, audio_filepath TEXT NOT NULL UNIQUE,
            representative TEXT NOT NULL, payload TEXT NOT NULL, quality REAL NOT NULL,
            contract_index INTEGER NOT NULL, byte_offset INTEGER NOT NULL,
            source TEXT NOT NULL, text_bytes INTEGER NOT NULL,
            blocked_reason TEXT, eval_excluded INTEGER NOT NULL,
            provenance_count INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX clip_order ON clips(quality DESC, contract_index, byte_offset, audio_filepath);
    ''')


def _choose_group(rows: Iterator[dict]) -> dict:
    best = fallback = None
    count, minimum, maximum, mismatch = 0, None, None, False
    for row in rows:
        count += 1
        errors = row['metadata_errors']
        mismatch |= 'language_mismatch' in errors
        duration = row['duration']
        if duration is not None:
            minimum = duration if minimum is None else min(minimum, duration)
            maximum = duration if maximum is None else max(maximum, duration)
        key = (row['quality'], -row['contract_index'], -row['byte_offset'])
        if fallback is None or key > fallback[0]:
            fallback = (key, row)
        if (row['text'] and 'invalid_duration' not in errors and 'language_mismatch' not in errors
                and (best is None or key > best[0])):
            best = (key, row)
    selected = dict((best or fallback)[1])
    blocked = None
    if mismatch:
        blocked = 'language_mismatch'
    elif minimum is not None and maximum - minimum > max(0.1, 0.01 * minimum) + 1e-12:
        blocked = 'duration_conflict'
    elif not selected['text']:
        blocked = 'missing_or_invalid_text'
    elif 'invalid_duration' in selected['metadata_errors']:
        blocked = 'invalid_duration'
    elif minimum is None:
        blocked = 'missing_duration'
    elif selected['duration'] is None:
        selected['duration'] = minimum
        selected['normalization'] = [*selected['normalization'], 'duration_borrowed_same_path']
    selected['blocked_reason'] = blocked
    selected['provenance_count'] = count
    return selected


def _populate_clips(root: Path, lang: str, db, eval_path: Path):
    from itertools import groupby

    with closing(_readonly(eval_path)) as eval_db:
        cursor = db.execute('SELECT audio_filepath,payload FROM occurrences '
                            'WHERE audio_filepath IS NOT NULL ORDER BY audio_filepath,contract_index,byte_offset')
        transactions = _Transactions(db)
        for audio, group in groupby(cursor, key=lambda record: record[0]):
            chosen = _choose_group(json.loads(record[1]) for record in group)
            excluded = eval_db.execute('SELECT 1 FROM exclusions WHERE audio_filepath=?', (audio,)).fetchone() is not None
            item = {key: chosen[key] for key in ('id', 'audio_filepath', 'lang', 'text',
                    'normalized_text', 'duration', 'accent', 'quality', 'source',
                    'normalization', 'repairable_flags', 'metadata_errors', 'provenance_count')}
            item['provenance_ref'] = {'database': f'worksets/{lang}.sqlite3',
                                      'table': 'occurrences', 'clip_id': chosen['id']}
            payload = _json(item)
            db.execute('INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (chosen['id'], audio, chosen['row_id'], payload, chosen['quality'],
                        chosen['contract_index'], chosen['byte_offset'], chosen['source'],
                        len(chosen['text'].encode('utf-8')), chosen['blocked_reason'],
                        int(excluded), chosen['provenance_count']))
            transactions.tick(len(payload))
        transactions.finish()
    _dispositions(db)


def _dispositions(db):
    cursor = db.execute('''SELECT o.row_id, c.representative, c.blocked_reason, c.eval_excluded
        FROM occurrences o LEFT JOIN clips c ON c.id=o.clip_id ORDER BY o.row_id''')
    transactions = _Transactions(db)
    for row_id, representative, blocked, excluded in cursor:
        if representative is None:
            disposition = 'source_invalid'
        elif excluded:
            disposition = 'eval_excluded'
        elif row_id != representative:
            disposition = 'duplicate'
        else:
            disposition = 'blocked' if blocked else 'eligible'
        db.execute('UPDATE occurrences SET disposition=? WHERE row_id=?', (disposition, row_id))
        transactions.tick(len(row_id) + len(disposition))
    transactions.finish()


def _inventory(db) -> dict:
    total, invalid = db.execute('SELECT COUNT(*),COALESCE(SUM(source_invalid),0) FROM occurrences').fetchone()
    unique, eligible, blocked, excluded, text_bytes, eligible_bytes = db.execute('''
        SELECT COUNT(*), COALESCE(SUM(eval_excluded=0 AND blocked_reason IS NULL),0),
        COALESCE(SUM(eval_excluded=0 AND blocked_reason IS NOT NULL),0),
        COALESCE(SUM(eval_excluded),0),
        COALESCE(SUM(CASE WHEN eval_excluded=0 THEN text_bytes ELSE 0 END),0),
        COALESCE(SUM(CASE WHEN eval_excluded=0 AND blocked_reason IS NULL THEN text_bytes ELSE 0 END),0)
        FROM clips''').fetchone()
    dispositions = dict(db.execute('SELECT disposition,COUNT(*) FROM occurrences GROUP BY disposition'))
    sources = {source: {'occurrences': count, 'shards': 0, 'bytes': 0, 'rows': 0}
               for source, count in db.execute('SELECT source,COUNT(*) FROM occurrences GROUP BY source')}
    for source, count, size in db.execute('SELECT source,COUNT(*),SUM(text_bytes) FROM clips WHERE eval_excluded=0 GROUP BY source'):
        sources[source].update(shards=1, bytes=size, rows=count)
    return {'source_rows': total, 'source_invalid': invalid,
            'source_invalid_unidentified': dispositions.get('source_invalid', 0),
            'unique': unique, 'eligible': eligible, 'blocked': blocked, 'eval_excluded': excluded,
            'duplicate': total - dispositions.get('source_invalid', 0) - unique,
            'dispositions': dispositions, 'candidate_count': eligible + blocked,
            'shards': int(eligible + blocked + invalid > 0), 'bytes': text_bytes,
            'eligible_text_bytes': eligible_bytes, 'est_rows': eligible + blocked,
            'needs_llm_fraction': 1.0, 'llm_rows': eligible, 'sources': sources,
            'work_estimate': {'clips': eligible, 'text_utf8_bytes': eligible_bytes},
            'blocked_reasons': dict(db.execute('SELECT blocked_reason,COUNT(*) FROM clips '
                                               'WHERE blocked_reason IS NOT NULL AND eval_excluded=0 GROUP BY blocked_reason'))}


def _replay_database_commit(root: Path, descriptor: dict) -> dict:
    """Finish a checksummed DB/report commit without trusting an orphan file."""
    report = descriptor['report']
    target = _artifact(root, report['database']['path'])
    current = _digest(target) if target.exists() else None
    if current != report['database']['sha256']:
        if current != descriptor['before_sha256']:
            raise RuntimeError('database changed outside its committed publication')
        _verify_file(root, descriptor['staged_database'])
        os.replace(_artifact(root, descriptor['staged_database']['path']), target)
        _sync_dir(target.parent)
    _atomic_json(root / 'worksets' / f'{report["lang"]}.json', report)
    return report


def _publish_database(root: Path, staged: Path, target: Path, report: dict, commit: Path) -> dict:
    staged_info = _db_description(root, staged)
    report = {**report, 'database': {**staged_info, 'path': _relative(root, target)}}
    descriptor = {'status': 'complete', 'report': report, 'staged_database': staged_info,
                  'before_sha256': _digest(target) if target.exists() else None}
    _atomic_json(commit, descriptor)
    return _replay_database_commit(root, descriptor)


def _copy_database(source: Path, destination: Path) -> None:
    # Owners have stopped every writer. A bounded file copy, not a giant SQL
    # write transaction, preserves the committed original until publication.
    with source.open('rb') as reader, destination.open('xb') as writer:
        while chunk := reader.read(1024 * 1024):
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    _sync_dir(destination.parent)


def build_language(root, lang, *, run_id='default') -> dict:
    """Read this language's prepared train segments once into a unique workset.

    ``source_invalid`` counts all initially metadata-invalid occurrences, so it
    overlaps duplicate/representative counts. ``dispositions`` is the disjoint
    row reconciliation; ``unique = eligible + blocked + eval_excluded``.
    Provenance is indexed by clip_id, retaining every original occurrence.
    """
    root, lang = Path(root).resolve(), _language(lang)
    with _owner(root / 'owners' / f'workset-{lang}', run_id):
        reports = _prepared_reports(root, run_id)
        if lang not in _languages(reports):
            raise ValueError(f'language not declared in train contract: {lang}')
        fingerprint = _fingerprint([r['fingerprint'] for r in reports])
        marker, final = root / 'worksets' / f'{lang}.json', root / 'worksets' / f'{lang}.sqlite3'
        commit = root / 'worksets' / f'{lang}.build-commit.json'
        if not marker.exists() and commit.exists():
            descriptor = _read_json(commit)
            if descriptor['report'].get('input_fingerprint') != fingerprint:
                raise RuntimeError('incompatible database publication')
            _replay_database_commit(root, descriptor)
        if marker.exists():
            previous = _read_json(marker)
            if previous.get('input_fingerprint') != fingerprint or previous.get('run_id') != run_id or previous.get('status') != 'complete':
                raise RuntimeError('incompatible workset report')
            _verify_file(root, previous['database'])
            return previous
        eval_index = reports[0]['eval_index']
        _verify_file(root, eval_index)
        temporary = final.with_name(f'.{lang}.{uuid.uuid4().hex}.sqlite3')
        source_quarantines = []
        with closing(_connect(temporary)) as db:
            _workset_schema(db)
            transactions = _Transactions(db)
            for report in reports:
                for segment in report['segments']:
                    if segment['split'] != 'train' or segment['lang'] != lang:
                        continue
                    _verify_file(root, segment['quarantine'])
                    source_quarantines.append(segment['quarantine'])
                    for chunk in segment['chunks']:
                        for row in _verified_rows(root, chunk):
                            payload = _json(row)
                            db.execute('INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)',
                                       (row['row_id'], row['id'], row['audio_filepath'],
                                        row['contract_index'], row['byte_offset'], row['source'],
                                        int(bool(row['metadata_errors'])), payload, 'pending'))
                            transactions.tick(len(payload))
            transactions.finish()
            _populate_clips(root, lang, db, _artifact(root, eval_index['path']))
            inventory = _inventory(db)
        result = {**inventory, 'schema_version': SCHEMA_VERSION, 'status': 'complete',
                  'run_id': run_id, 'lang': lang, 'finalized': False,
                  'input_fingerprint': fingerprint,
                  'eval_index': eval_index, 'source_quarantines': source_quarantines}
        return _publish_database(root, temporary, final, result, commit)


def finalize_worksets(root) -> dict:
    """Rank-zero-only phase: disk-backed cross-language joins, then exact recount.

    All language reports and exclusive owners must be available before this
    phase. A shared disk index replaces any corpus-sized Python identity set.
    The final marker is authoritative; interrupted finalization is replayable.
    """
    root = Path(root).resolve()
    reports = _prepared_reports(root)
    run_id, langs = reports[0]['run_id'], _languages(reports)
    with _owner(root / 'owners/finalize-worksets', run_id), ExitStack() as owners:
        for lang in langs:
            owners.enter_context(_owner(root / 'owners' / f'workset-{lang}', run_id))
        fingerprint = _fingerprint([r['fingerprint'] for r in reports])
        completion = root / 'worksets/finalized.json'
        if not completion.exists():
            for lang in langs:
                commit = root / 'worksets' / f'{lang}.finalize-commit.json'
                if commit.exists():
                    descriptor = _read_json(commit)
                    if descriptor['report'].get('input_fingerprint') != fingerprint:
                        raise RuntimeError('incompatible finalization publication')
                    _replay_database_commit(root, descriptor)
        markers = {lang: _read_json(root / 'worksets' / f'{lang}.json') for lang in langs}
        for report in markers.values():
            if report.get('status') != 'complete' or report.get('run_id') != run_id or report.get('input_fingerprint') != fingerprint:
                raise RuntimeError('all compatible language worksets must finish before finalization')
            _verify_file(root, report['database'])
        if completion.exists():
            previous = _read_json(completion)
            if previous.get('input_fingerprint') != fingerprint:
                raise RuntimeError('incompatible finalization')
            for report in markers.values():
                _verify_file(root, report['database'])
            return previous['inventories']
        index_path = root / 'worksets' / f'cross-language-{uuid.uuid4().hex}.sqlite3'
        with closing(_connect(index_path)) as index:
            index.execute('CREATE TABLE identities (audio_filepath TEXT, lang TEXT, PRIMARY KEY(audio_filepath,lang)) WITHOUT ROWID')
            transactions = _Transactions(index)
            for lang in langs:
                with closing(_readonly(root / 'worksets' / f'{lang}.sqlite3')) as db:
                    for (audio,) in db.execute('SELECT audio_filepath FROM clips'):
                        index.execute('INSERT INTO identities VALUES (?,?)', (audio, lang))
                        transactions.tick(len(audio.encode('utf-8')))
            transactions.finish()
            index.execute('CREATE TABLE conflicts (audio_filepath TEXT PRIMARY KEY) WITHOUT ROWID')
            transactions = _Transactions(index)
            for (audio,) in index.execute('SELECT audio_filepath FROM identities GROUP BY audio_filepath HAVING COUNT(*)>1'):
                index.execute('INSERT INTO conflicts VALUES (?)', (audio,))
                transactions.tick(len(audio.encode('utf-8')))
            transactions.finish()
            for lang in langs:
                if markers[lang]['finalized']:
                    continue
                path = root / 'worksets' / f'{lang}.sqlite3'
                staged = path.with_name(f'.{lang}.finalize-{uuid.uuid4().hex}.sqlite3')
                _copy_database(path, staged)
                with closing(_connect(staged)) as db:
                    db.execute('ATTACH DATABASE ? AS cross_language', (index_path.as_uri() + '?mode=ro',))
                    conflicts = db.execute('SELECT c.audio_filepath FROM clips c '
                                           'JOIN cross_language.conflicts x ON c.audio_filepath=x.audio_filepath '
                                           'ORDER BY c.audio_filepath')
                    transactions = _Transactions(db)
                    for (audio,) in conflicts:
                        db.execute("UPDATE clips SET blocked_reason='cross_language_conflict' WHERE audio_filepath=?", (audio,))
                        transactions.tick(len(audio.encode('utf-8')))
                    transactions.finish()
                    _dispositions(db)
                    markers[lang].update(_inventory(db))
                markers[lang]['finalized'] = True
                commit = root / 'worksets' / f'{lang}.finalize-commit.json'
                markers[lang] = _publish_database(root, staged, path, markers[lang], commit)
        _atomic_json(completion, {'status': 'complete', 'run_id': run_id,
                                 'input_fingerprint': fingerprint, 'inventories': markers,
                                 'cross_language_index': _db_description(root, index_path)})
        return markers


def partition_language(root, lang, nparts) -> dict:
    """Physically split the sorted candidate stream once using assemble's hash.

    At most MAX_OPEN_PARTS buffered files are open. Metadata-blocked unique
    identities are included; eval identities and nonrepresentative duplicates
    stay only in the provenance tables. A complete report freezes nparts.
    """
    if not isinstance(nparts, int) or isinstance(nparts, bool) or nparts < 1:
        raise ValueError('nparts must be a positive integer')
    root, lang = Path(root).resolve(), _language(lang)
    finalized = _read_json(root / 'worksets/finalized.json')
    if finalized.get('status') != 'complete':
        raise RuntimeError('worksets have not completed finalization')
    workset = finalized['inventories'][lang]
    run_id = workset['run_id']
    with _owner(root / 'owners' / f'partition-{lang}', run_id):
        marker = root / 'partitions' / f'{lang}.json'
        if marker.exists():
            previous = _read_json(marker)
            if previous.get('status') != 'complete' or previous.get('run_id') != run_id or previous['nparts'] != nparts or previous['workset_sha256'] != workset['database']['sha256']:
                raise RuntimeError('cannot repartition a committed language')
            for part in previous['parts']:
                _verify_file(root, part)
            return previous
        _verify_file(root, workset['database'])
        directory = root / 'partitions' / lang
        _mkdir(directory)
        attempt = uuid.uuid4().hex
        temporaries = [directory / f'.p{part:04d}.{attempt}.tmp' for part in range(nparts)]
        stats = [{'part': part, 'rows': 0, 'bytes': 0, 'sha': hashlib.sha256()} for part in range(nparts)]
        for path in temporaries:
            with path.open('xb'):
                pass
        opened = OrderedDict()
        try:
            with closing(_readonly(_artifact(root, workset['database']['path']))) as db:
                for payload, blocked in db.execute('SELECT payload,blocked_reason FROM clips WHERE eval_excluded=0 '
                                                    'ORDER BY quality DESC,contract_index,byte_offset,audio_filepath'):
                    item = json.loads(payload)
                    if blocked is not None:
                        item['blocked_reason'] = blocked
                    digest = hashlib.blake2b(item['audio_filepath'].encode('utf-8'), digest_size=8).digest()
                    part = int.from_bytes(digest, 'big') % nparts
                    if part not in opened:
                        if len(opened) >= MAX_OPEN_PARTS:
                            _, old = opened.popitem(last=False)
                            old.close()
                        opened[part] = temporaries[part].open('ab', buffering=64 * 1024)
                    opened.move_to_end(part)
                    data = (_json(item) + '\n').encode()
                    opened[part].write(data)
                    stats[part]['rows'] += 1
                    stats[part]['bytes'] += len(data)
                    stats[part]['sha'].update(data)
        finally:
            for handle in opened.values():
                handle.close()
        parts = []
        for part, temporary in enumerate(temporaries):
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            path = directory / f'p{part:04d}.jsonl'
            os.replace(temporary, path)
            parts.append({'part': part, 'path': _relative(root, path), 'rows': stats[part]['rows'],
                          'bytes': stats[part]['bytes'], 'sha256': stats[part]['sha'].hexdigest()})
        _sync_dir(directory)
        result = {'status': 'complete', 'schema_version': SCHEMA_VERSION, 'run_id': run_id,
                  'lang': lang, 'nparts': nparts, 'parts': parts,
                  'rows': sum(p['rows'] for p in parts), 'bytes': sum(p['bytes'] for p in parts),
                  'workset_sha256': workset['database']['sha256']}
        if result['rows'] != workset['candidate_count']:
            raise RuntimeError('partition row accounting mismatch')
        _atomic_json(marker, result)
        return result


def iter_partition(root, lang, part, start=0) -> Iterator[dict]:
    """Stream only the assigned file, skipping ``start`` input rows in O(start).

    Index checkpoints are supported; byte-cursor checkpoints are not yet part
    of this API. The checksum is verified on exhaustion, including skipped rows.
    """
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError('start must be a nonnegative input index')
    root, lang = Path(root).resolve(), _language(lang)
    report = _read_json(root / 'partitions' / f'{lang}.json')
    if not isinstance(part, int) or isinstance(part, bool) or not 0 <= part < report['nparts']:
        raise ValueError('partition index out of range')
    if report.get('status') != 'complete':
        raise RuntimeError('partition is not committed')
    for index, row in enumerate(_verified_rows(root, report['parts'][part])):
        if index >= start:
            yield row
