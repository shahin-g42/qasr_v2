"""Frozen, distributed short-clip indexing. No generated audio is stored."""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import signal
import socket
import sqlite3
import time
import uuid
from collections import Counter
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from queue import Empty, SimpleQueue

import yaml

from ..asr_clean.plan import _byte_ranges, _limit_offset, iter_chunk_lines
from ..asr_clean.text import local_path
from .common import (
    LANGS,
    PAIRS,
    VERSION,
    atomic_json,
    chunk_paths,
    digest,
    load_plan,
    lock,
    read_json,
    sync_dir,
)


def canonical_audio(raw, manifest):
    path = Path(local_path(raw))
    return str(path if path.is_absolute() else (Path(manifest).parent / path).resolve())


def identity(path):
    # Conservatively exclude UUID basename aliases, as in check_manifest_overlap.
    return Path(path).name


def tokenizer_from(path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def validate_config(cfg):
    required = {'source_root', 'tokenizer_path', 'preview_count', 'pair_weights',
                'language_policy', 'min_source_seconds', 'max_source_seconds', 'max_seconds',
                'sample_rate', 'gap_seconds', 'segment_counts', 'max_target_tokens',
                'max_attempts', 'max_source_uses', 'chunk_mb', 'seed'}
    if set(cfg) != required:
        raise ValueError(f'Config keys differ: missing={required - set(cfg)}, unknown={set(cfg) - required}')
    if set(cfg['pair_weights']) != set(PAIRS):
        raise ValueError('Specify all seven supported pairs exactly once')
    if any(type(v) is not int or v <= 0 for v in cfg['pair_weights'].values()):
        raise ValueError('Pair weights must be positive integers')
    if cfg['language_policy'] not in ('first', 'major'):
        raise ValueError('language_policy must be first or major')
    for key in ('sample_rate', 'max_target_tokens', 'max_attempts', 'max_source_uses', 'chunk_mb'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(cfg['seed']) is not int or cfg['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('min_source_seconds', 'max_source_seconds', 'max_seconds', 'gap_seconds'):
        if not math.isfinite(cfg[key]):
            raise ValueError(f'{key} must be finite')
    if not (0 < cfg['min_source_seconds'] < cfg['max_source_seconds'] <= cfg['max_seconds'] <= 35):
        raise ValueError('Require 0 < min_source < max_source <= max_seconds <= 35')
    if type(cfg['preview_count']) is not int or cfg['preview_count'] < 14 or cfg['gap_seconds'] < 0:
        raise ValueError('preview_count must be an integer >= 14; gap_seconds must be >= 0')
    if 2 * cfg['max_source_seconds'] + cfg['gap_seconds'] > cfg['max_seconds']:
        raise ValueError('Two maximum-length sources plus a gap must fit max_seconds')
    if not cfg['segment_counts'] or any(type(n) is not int or n not in (2, 3, 4) for n in cfg['segment_counts']):
        raise ValueError('segment_counts must contain only 2, 3 or 4')


def _scan_source(path):
    h = hashlib.sha256()
    lines = 0
    before = Path(path).stat()
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            lines += bool(line.strip())
    after = Path(path).stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f'Manifest changed during planning: {path}')
    return {'path': path, 'rows': lines, 'size': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': h.hexdigest()}


def plan(root, config, jobs=5, samples=None, limit=None):
    """One planner; the final plan is published only after all dependencies exist."""
    root = Path(root).resolve()
    state = root / '_state'
    with lock(state / 'lifecycle.lock'), lock(state / 'planner.lock'):
        if (state / 'plan.json').exists():
            raise ValueError('Frozen plan already exists; choose a new RUN_ID')
        if limit is not None and (limit < 1 or samples is None):
            raise ValueError('--limit requires --samples (a pilot run) and must be positive')
        cfg = yaml.safe_load(Path(config).read_text())
        validate_config(cfg)
        source = Path(cfg['source_root']).resolve()
        train = [(lang, str(p.resolve())) for lang in LANGS
                 for p in sorted((source / lang / 'data').glob('train_*.jsonl'))]
        evaluation = [str(p.resolve()) for p in sorted(source.glob('*/data/eval_*.jsonl'))]
        if {lang for lang, _ in train} != set(LANGS) or not evaluation:
            raise ValueError('Need training manifests for all five languages and eval manifests under LANG/data')
        print(f'Planning: scanning {len(train)} train and {len(evaluation)} eval manifests with {jobs} workers', flush=True)
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            scans = list(pool.map(_scan_source, [p for _, p in train] + evaluation))
        excluded = set()
        for path in evaluation:
            with open(path) as f:
                for raw in f:
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    audio = row.get('audio_filepath') or row.get('wav_path')
                    if not isinstance(audio, str) or not audio:
                        raise ValueError(f'Eval record lacks an audio path: {path}')
                    excluded.add(identity(canonical_audio(audio, path)))
        atomic_json(state / 'excluded.json', sorted(excluded))
        tokenizer_dir = state / 'tokenizer'
        tokenizer = tokenizer_from(cfg['tokenizer_path'])
        tokenizer.save_pretrained(tokenizer_dir)
        from qasr.code_switch import tokenizer_fingerprint
        token_hash = tokenizer_fingerprint(tokenizer)
        chunks = []
        for lang, path in train:
            size = _limit_offset(path, limit) if limit is not None else None
            for start, end in _byte_ranges(path, cfg['chunk_mb'] << 20, size):
                chunks.append({'id': f'{len(chunks):07d}', 'lang': lang, 'source': path,
                               'start': start, 'end': end})
        base = sum(s['rows'] for s in scans[:len(train)])
        target = cfg['preview_count']
        if samples is not None:
            if samples < len(PAIRS) * 2:
                raise ValueError('Pilot needs at least 14 preview samples')
            target = samples
        result = {'version': VERSION, 'root': str(root), 'config': cfg, 'chunks': chunks,
                  'sources': scans, 'baseline_records': base, 'preview_samples': target,
                  'planned_source_rows': sum(min(s['rows'], limit) if limit else s['rows'] for s in scans[:len(train)]),
                  'pilot': samples is not None, 'created': time.time(), 'tokenizer_fingerprint': token_hash,
                  'excluded_sha256': digest(state / 'excluded.json'),
                  'tokenizer_hashes': {p.name: digest(p) for p in tokenizer_dir.iterdir() if p.is_file()}}
        atomic_json(state / 'plan.json', result)
        print(f'Frozen {len(chunks)} chunks, {base:,} source rows, {target:,} preview samples'
              + (' (PILOT)' if samples is not None else ''), flush=True)
        return result


def check_frozen(root, p, full=False):
    if p.get('append_base'):
        from .append import check_base
        check_base(p)
    for src in p['sources']:
        st = Path(src['path']).stat()
        if (st.st_size, st.st_mtime_ns) != (src['size'], src['mtime_ns']):
            raise ValueError(f"Frozen manifest changed: {src['path']}")
        if full and digest(src['path']) != src['sha256']:
            raise ValueError(f"Frozen manifest hash differs: {src['path']}")
    state = Path(root) / '_state'
    if p.get('derived_index') and digest(state / 'quarantine.jsonl') != p['derived_index']['quarantine_sha256']:
        raise ValueError('Frozen quarantine changed')
    if digest(state / 'excluded.json') != p['excluded_sha256']:
        raise ValueError('Eval exclusions changed')
    for name, expected in p['tokenizer_hashes'].items():
        if digest(state / 'tokenizer' / name) != expected:
            raise ValueError('Frozen tokenizer changed')


def _row(raw, chunk, offset, cfg, excluded, tokenizer):
    import soundfile as sf

    from qasr.audio import AudioLoadingError, load_mono_audio
    try:
        record = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None, 'invalid_json'
    if not isinstance(record, dict):
        return None, 'invalid_record'
    text = record.get('text')
    audio = record.get('audio_filepath') or record.get('wav_path')
    if not isinstance(text, str) or not text.strip() or not any(c.isalnum() for c in text):
        return None, 'empty_text'
    if not isinstance(audio, str) or not audio:
        return None, 'missing_path'
    audio = canonical_audio(audio, chunk['source'])
    if identity(audio) in excluded:
        return None, 'eval_overlap'
    tokens = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    # Leave room for at least one complete partner transcript and a join.
    if tokens > max(1, (cfg['max_target_tokens'] - 4) // 2):
        return None, 'source_tokens'
    try:
        info = sf.info(audio)
        if not cfg['min_source_seconds'] <= info.duration <= cfg['max_source_seconds']:
            return None, 'source_duration'
        waveform = load_mono_audio(audio, cfg['sample_rate'])
    except FileNotFoundError:
        raise
    except (AudioLoadingError, sf.LibsndfileError) as exc:
        # Missing/inaccessible files can mean an unmounted Lustre source. Do
        # not silently turn a storage outage into millions of exclusions.
        if not Path(audio).is_file():
            raise OSError(f'Source unavailable: {audio}') from exc
        return None, 'decode_error'
    frames = len(waveform)
    duration = frames / cfg['sample_rate']
    if not cfg['min_source_seconds'] <= duration <= cfg['max_source_seconds']:
        return None, 'decoded_duration'
    return {'audio_filepath': audio, 'text': text, 'duration': duration,
            'frames': frames, 'waveform_sha256': hashlib.sha256(waveform.tobytes()).hexdigest(),
            'tokens': tokens, 'language': chunk['lang'],
            'source': chunk['source'], 'source_offset': offset}, None


def _index_chunk(root, p, chunk, tokenizer, excluded, stop):
    part, rejects, progress, _ = chunk_paths(root, chunk)
    old = read_json(progress, {})
    if old.get('done'):
        return
    for path in (part, rejects):
        path.parent.mkdir(parents=True, exist_ok=True)
    offset = old.get('offset', chunk['start'])
    counts = Counter(old.get('rejects', {}))
    kept = old.get('kept', 0)
    processed = old.get('processed', 0)
    with part.open('r+b' if part.exists() else 'w+b') as out, rejects.open('r+b' if rejects.exists() else 'w+b') as rej:
        for fh, key in ((out, 'bytes'), (rej, 'reject_bytes')):
            committed = old.get(key, 0)
            if os.fstat(fh.fileno()).st_size < committed:
                raise ValueError(f"Committed output shortened: {chunk['id']}; verify --reset")
            fh.truncate(committed)
            fh.seek(committed)
        batch = 0

        def commit(done=False):
            for fh in (out, rej):
                fh.flush()
                os.fsync(fh.fileno())
            atomic_json(progress, {'offset': offset, 'bytes': out.tell(), 'reject_bytes': rej.tell(),
                'kept': kept, 'processed': processed, 'rejects': dict(counts), 'done': done,
                'updated': time.time(), 'host': socket.gethostname(),
                'sha256': digest(part) if done else None,
                'reject_sha256': digest(rejects) if done else None})

        for _, end, raw in iter_chunk_lines(chunk['source'], offset, chunk['end']):
            if stop.exists():
                commit()
                return
            if raw.strip():
                record, reason = _row(raw, chunk, offset, p['config'], excluded, tokenizer)
                if record:
                    out.write((json.dumps(record, ensure_ascii=False) + '\n').encode())
                    kept += 1
                else:
                    counts[reason] += 1
                    rej.write((json.dumps({'source_offset': offset, 'reason': reason}) + '\n').encode())
                processed += 1
            offset = end
            batch += 1
            if batch >= 100:
                commit()
                batch = 0
        commit(done=True)


def worker(root, session, slot):
    p = load_plan(root)
    state = Path(root) / '_state'
    with lock(state / 'lifecycle.lock', shared=True):
        tokenizer = tokenizer_from(state / 'tokenizer')
        excluded = set(read_json(state / 'excluded.json'))
        stop = state / 'stop' / session
        chunks = p['chunks']
        # Different starting points reduce contention without fixing ownership.
        start = int(hashlib.sha256(f'{session}:{slot}'.encode()).hexdigest(), 16) % max(1, len(chunks))
        for i in range(len(chunks)):
            if stop.exists():
                return
            chunk = chunks[(start + i) % len(chunks)]
            if read_json(chunk_paths(root, chunk)[2], {}).get('done'):
                continue
            try:
                with lock(state / 'locks' / chunk['id']):
                    _index_chunk(root, p, chunk, tokenizer, excluded, stop)
            except BlockingIOError:
                continue


def run(root, procs):
    root = Path(root).resolve()
    state = root / '_state'
    host = socket.gethostname().split('.')[0]
    session = f'{host}-{uuid.uuid4().hex}'
    with lock(state / f'node-{host}.lock'), lock(state / 'lifecycle.lock', shared=True):
        p = load_plan(root)
        check_frozen(root, p)
        marker = state / 'sessions' / f'{session}.json'
        stop = state / 'stop' / session
        atomic_json(marker, {'host': host, 'pid': os.getpid(), 'started': time.time(), 'processed_at_start': status(root)['processed']})
        stop.parent.mkdir(parents=True, exist_ok=True)
        previous = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, lambda *_: stop.touch())
        children = []
        try:
            ctx = mp.get_context('spawn')
            for slot in range(procs):
                child = ctx.Process(target=worker, args=(str(root), session, slot))
                child.start()
                children.append(child)
            while any(child.is_alive() for child in children):
                if any(child.exitcode not in (None, 0) for child in children):
                    stop.touch()
                for child in children:
                    child.join(timeout=0.2)
            if any(child.exitcode != 0 for child in children):
                raise RuntimeError('Index worker failed; inspect logs, fix the cause, then run again')
        finally:
            stop.touch()
            for child in children:
                child.join()
            marker.unlink(missing_ok=True)
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def stop_run(root, all_nodes=False):
    state = Path(root) / '_state'
    host = socket.gethostname().split('.')[0]
    (state / 'stop').mkdir(parents=True, exist_ok=True)
    count = 0
    for path in (state / 'sessions').glob('*.json'):
        session = read_json(path)
        if all_nodes or session['host'] == host:
            (state / 'stop' / path.stem).touch()
            count += 1
    print(f'Stop requested for {count} sessions; workers commit before exiting')


def status(root):
    p = load_plan(root)
    result = {'chunks': len(p['chunks']), 'done': 0, 'started': 0, 'processed': 0,
              'eligible': 0, 'source_rows': p['planned_source_rows'], 'preview_samples': p['preview_samples'],
              'rejects': Counter(), 'by_language': Counter(), 'sessions': []}
    for c in p['chunks']:
        st = read_json(chunk_paths(root, c)[2], {})
        result['started'] += bool(st)
        result['done'] += bool(st.get('done'))
        result['processed'] += st.get('processed', 0)
        result['eligible'] += st.get('kept', 0)
        result['by_language'][c['lang']] += st.get('kept', 0)
        result['rejects'].update(st.get('rejects', {}))
    result['sessions'] = [read_json(path) for path in (Path(root) / '_state/sessions').glob('*.json')]
    sessions = result['sessions']
    if sessions:
        earliest = min(sessions, key=lambda s: s['started'])
        elapsed = max(1, time.time() - earliest['started'])
        rate = max(0, result['processed'] - earliest['processed_at_start']) / elapsed
        result['records_per_second'] = round(rate, 2)
        result['eta_hours'] = round((result['source_rows'] - result['processed']) / rate / 3600, 2) if rate else None
    result['index_ready'] = (Path(root) / 'assembled/index.json').exists()
    return result


def _verify_one(task):
    root, chunk, cfg, audio, metadata_only = task
    import soundfile as sf
    part, rejects, progress, _ = chunk_paths(root, chunk)
    try:
        st = read_json(progress, {})
        if not st.get('done') or st['offset'] != chunk['end']:
            raise ValueError('incomplete chunk')
        if part.stat().st_size != st['bytes'] or rejects.stat().st_size != st['reject_bytes']:
            raise ValueError('committed size mismatch')
        part_hash, reject_hash = hashlib.sha256(), hashlib.sha256()
        n = 0
        with part.open('rb') as f:
            for raw in f:
                part_hash.update(raw)
                row = json.loads(raw)
                if row['language'] != chunk['lang'] or not row['text'].strip():
                    raise ValueError('language/text mismatch')
                if not cfg['min_source_seconds'] <= row['duration'] <= cfg['max_source_seconds']:
                    raise ValueError('duration outside source limits')
                if audio:
                    from qasr.audio import load_mono_audio
                    waveform = load_mono_audio(row['audio_filepath'], cfg['sample_rate'])
                    if len(waveform) != row['frames'] or hashlib.sha256(waveform.tobytes()).hexdigest() != row['waveform_sha256']:
                        raise ValueError('source audio content or duration changed')
                elif not metadata_only:
                    info = sf.info(row['audio_filepath'])
                    if abs(info.duration - row['duration']) > 2 / cfg['sample_rate']:
                        raise ValueError('source header duration changed')
                n += 1
        reject_counts = Counter()
        with rejects.open('rb') as f:
            for raw in f:
                reject_hash.update(raw)
                reject_counts[json.loads(raw)['reason']] += 1
        if part_hash.hexdigest() != st['sha256'] or reject_hash.hexdigest() != st['reject_sha256']:
            raise ValueError('committed hash mismatch')
        if n != st['kept'] or reject_counts != Counter(st['rejects']) or n + sum(reject_counts.values()) != st['processed']:
            raise ValueError('committed counters mismatch')
        return chunk['id'], None
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return chunk['id'], str(exc)


def _verify_source_hash(source):
    started = time.monotonic()
    hasher = hashlib.sha256()
    size = 0
    next_report = 4 << 30
    with open(source['path'], 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            hasher.update(block)
            size += len(block)
            if size >= next_report:
                print(f"verify: {Path(source['path']).name}: {size / (1 << 30):.1f}/"
                      f"{source['size'] / (1 << 30):.1f} GiB hashed", flush=True)
                next_report += 4 << 30
    if hasher.hexdigest() != source['sha256']:
        raise ValueError(f"Frozen manifest hash differs: {source['path']}")
    return Path(source['path']).name, time.monotonic() - started


def _verification_snapshot(root, p, jobs=1):
    """Metadata change detector for reusing a completed verification under the lock.

    This covers index artifacts, not the millions of source audio files. Audio
    validation remains a point-in-time audit; training hashes decoded partners.
    """
    root = Path(root)
    paths = [root / '_state/plan.json', root / '_state/excluded.json']
    paths.extend(root / '_state/tokenizer' / name for name in p['tokenizer_hashes'])
    paths.extend(Path(source['path']) for source in p['sources'])
    for chunk in p['chunks']:
        paths.extend(chunk_paths(root, chunk)[:3])
    workers = min(max(1, jobs), 32, len(paths))
    print(f'verify: snapshotting {len(paths):,} file signatures with {workers} threads', flush=True)
    started = last_report = time.monotonic()

    def signature(path):
        try:
            st = path.stat()
            return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino]
        except FileNotFoundError:
            return None

    snapshot = {}
    completed = SimpleQueue()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for path in paths:
            future = pool.submit(signature, path)
            futures[future] = str(path)
            future.add_done_callback(completed.put)
        while futures:
            try:
                future = completed.get(timeout=5)
                snapshot[futures.pop(future)] = future.result()
            except Empty:
                pass
            now = time.monotonic()
            if now - last_report >= 5 or not futures:
                print(f'verify: signatures {len(snapshot):,}/{len(paths):,} '
                      f'({now - started:.1f}s elapsed)', flush=True)
                last_report = now
    return snapshot


def _save_verification(root, before, p, audio=False, metadata_only=False, jobs=1):
    after = _verification_snapshot(root, p, jobs)
    if after != before:
        raise ValueError('Index inputs changed during verification; verify again')
    atomic_json(Path(root) / '_state/verified.json', {
        'plan_sha256': digest(Path(root) / '_state/plan.json'), 'at': time.time(),
        'decoded_audio': audio, 'mode': 'audio' if audio else 'metadata' if metadata_only else 'headers',
        'snapshot': after})


def verify_unlocked(root, p, jobs, audio=False, metadata_only=False):
    if audio and metadata_only:
        raise ValueError('--audio and --metadata-only are mutually exclusive')
    print('verify: checking frozen manifest metadata, exclusions and tokenizer', flush=True)
    check_frozen(root, p)
    bad = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        # Retained manifests/pools keep their immutable receipt and metadata.
        # Only the additions need content hashing or audio validation.
        new_sources = set(p['append_base']['new_sources']) if p.get('append_base') else None
        sources = [source for source in p['sources']
                   if new_sources is None or source['path'] in new_sources]
        print(f"verify: hashing {len(sources)} frozen manifests; "
              f"at most {min(jobs, len(sources))} concurrent files", flush=True)
        futures = [pool.submit(_verify_source_hash, source) for source in sources]
        for i, future in enumerate(as_completed(futures), 1):
            name, elapsed = future.result()
            print(f'verify: manifests {i}/{len(futures)}: {name} ({elapsed:.1f}s)', flush=True)
        mode = 'metadata only (no audio opens)' if metadata_only else 'decoded audio' if audio else 'audio headers'
        print(f'verify: checking {len(p["chunks"])} chunks, mode={mode}, jobs={jobs}', flush=True)
        futures = [pool.submit(_verify_one, (str(root), c, p['config'], audio, metadata_only)) for c in p['chunks']]
        for i, future in enumerate(as_completed(futures), 1):
            cid, error = future.result()
            if error:
                bad[cid] = error
            if i == 1 or i % 100 == 0 or i == len(futures):
                print(f'verify: {i}/{len(futures)} chunks, {len(bad)} bad', flush=True)
    return bad


def verify(root, jobs, reset=False, audio=False, metadata_only=False):
    print(f'verify: starting; jobs={jobs}; loading plan from {root}', flush=True)
    root = Path(root).resolve()
    with lock(root / '_state/lifecycle.lock'):
        p = load_plan(root)
        if reset and (root / 'assembled').exists():
            raise ValueError('Published indexes are immutable; use a new RUN_ID to replace them')
        # A failed explicit audit must invalidate any previous success receipt.
        (root / '_state/verified.json').unlink(missing_ok=True)
        before = _verification_snapshot(root, p, jobs)
        bad = verify_unlocked(root, p, jobs, audio, metadata_only)
        published = read_json(root / 'assembled/index.json')
        if published:
            if published['plan_sha256'] != digest(root / '_state/plan.json'):
                bad['index'] = 'published index plan hash differs'
            for lang, pool in published['pools'].items():
                path = root / 'assembled' / f'{lang}.sqlite'
                if not path.is_file() or digest(path) != pool['sha256']:
                    bad[f'pool-{lang}'] = 'published pool missing or corrupt'
        if reset:
            for c in p['chunks']:
                if c['id'] in bad:
                    for path in chunk_paths(root, c)[:3]:
                        path.unlink(missing_ok=True)
        if not bad:
            _save_verification(root, before, p, audio, metadata_only, jobs)
        return bad


def allocate(total, weights):
    denominator = sum(weights.values())
    quotas = {key: total * value // denominator for key, value in weights.items()}
    remaining = total - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(total * weights[key] % denominator), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _assemble_language(task):
    root, output, lang, chunks = task
    path = Path(output) / f'{lang}.sqlite'
    # Incomplete assembly may be rerun, but published assembled/ is immutable.
    path.unlink(missing_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE pool (id INTEGER PRIMARY KEY, audio_key TEXT UNIQUE NOT NULL, record TEXT NOT NULL)')
    count, duplicates, total_seconds = 0, 0, 0.0
    started = time.monotonic()
    last_report = started
    print(f'assemble: {lang}: building from {len(chunks)} chunks', flush=True)
    for chunk_number, chunk in enumerate(chunks, 1):
        with chunk_paths(root, chunk)[0].open() as f:
            for raw in f:
                row = json.loads(raw)
                cursor = db.execute('INSERT OR IGNORE INTO pool VALUES (?,?,?)',
                                    (count, identity(row['audio_filepath']), raw.strip()))
                if cursor.rowcount:
                    count += 1
                    total_seconds += row['duration']
                else:
                    duplicates += 1
        db.commit()
        if time.monotonic() - last_report >= 30 or chunk_number == len(chunks):
            print(f'assemble: {lang}: {chunk_number}/{len(chunks)} chunks, {count:,} unique records, '
                  f'{count / max(1, time.monotonic() - started):.0f} records/s', flush=True)
            last_report = time.monotonic()
    db.close()
    print(f'assemble: {lang}: hashing completed SQLite file', flush=True)
    with path.open('rb') as f:
        os.fsync(f.fileno())
    return lang, {'count': count, 'duplicates': duplicates, 'seconds': total_seconds,
                  'sha256': digest(path), 'size': path.stat().st_size}


def assemble(root, jobs):
    print(f'assemble: starting; jobs={jobs}; loading plan from {root}', flush=True)
    root = Path(root).resolve()
    with lock(root / '_state/lifecycle.lock'):
        p = load_plan(root)
        output = root / 'assembled'
        if output.exists():
            raise ValueError('assembled/ is already published; do not mutate an index used by training')
        check_frozen(root, p)
        before = _verification_snapshot(root, p, jobs)
        receipt = read_json(root / '_state/verified.json', {})
        if (receipt.get('snapshot') == before
                and receipt.get('plan_sha256') == digest(root / '_state/plan.json')):
            print(f"assemble: reusing successful {receipt.get('mode', 'unknown')} verification; "
                  "all index input signatures unchanged", flush=True)
        else:
            print('assemble: no current verification receipt; verifying first', flush=True)
            bad = verify_unlocked(root, p, jobs)
            if bad:
                raise ValueError(f'{len(bad)} chunks incomplete/bad; run verify for details')
            _save_verification(root, before, p, jobs=jobs)
        temporary = root / '_assembling'
        temporary.mkdir(exist_ok=True)
        tasks = [(str(root), str(temporary), lang, [c for c in p['chunks'] if c['lang'] == lang]) for lang in LANGS]
        print(f'assemble: {min(jobs, len(LANGS))} SQLite writers (one per language); '
              'increasing jobs beyond five does not increase database writers', flush=True)
        if p.get('append_base'):
            from .append import assemble_pools
            pools = assemble_pools(root, temporary, p)
        else:
            with ProcessPoolExecutor(max_workers=min(jobs, len(LANGS))) as pool:
                pools = dict(pool.map(_assemble_language, tasks))
        # Pools must have disjoint source identities, including path aliases.
        with sqlite3.connect(':memory:') as db:
            for lang in LANGS:
                db.execute(f'ATTACH DATABASE ? AS {lang}', (str(temporary / f'{lang}.sqlite'),))
            for i, a in enumerate(() if p.get('append_base') else LANGS):
                for b in LANGS[i+1:]:
                    small, large = sorted((a, b), key=lambda lang: pools[lang]['count'])
                    print(f'assemble: checking identities {small} -> {large}', flush=True)
                    # CROSS JOIN fixes the outer scan to the smaller pool; the
                    # large pool is probed through its unique audio_key index.
                    if db.execute(f'SELECT 1 FROM {small}.pool a CROSS JOIN {large}.pool b '
                                  'WHERE b.audio_key=a.audio_key LIMIT 1').fetchone():
                        raise ValueError(f'Conflicting language pools share audio identities: {a}, {b}')
        cfg = p['config']
        targets = allocate(p['preview_samples'], cfg['pair_weights'])
        if any(count < 1 for count in targets.values()):
            raise ValueError('Target/weights leave a requested language pair empty')
        reuse = Counter()
        for pair, count in targets.items():
            a, b = pair.split('-')
            na, nb = pools[a]['count'], pools[b]['count']
            reserved = count * cfg['max_attempts']
            if min(na, nb) < 2 or reserved > na * nb:
                raise ValueError(f'Insufficient unique combinations for {pair}: {na} x {nb}, reserve={reserved}')
            for lang in (a, b):
                reuse[lang] += (2 if max(cfg['segment_counts']) > 2 else 1) * math.ceil(reserved / pools[lang]['count'])
        report = {'pools': pools, 'pair_targets': targets, 'source_use_upper_bounds': dict(reuse),
                  'preview_samples': p['preview_samples'], 'baseline_records': p['baseline_records']}
        atomic_json(root / '_reports/inventory.json', report)
        if any(n > cfg['max_source_uses'] for n in reuse.values()):
            raise ValueError('Configured source reuse limit is insufficient; inspect _reports/inventory.json')
        result = {**report, 'version': VERSION, 'config': cfg, 'pilot': p['pilot'],
                  'root': str(root), 'plan_sha256': digest(root / '_state/plan.json'),
                  'sources': p['sources'], 'tokenizer_hashes': p['tokenizer_hashes'],
                  'tokenizer_fingerprint': p['tokenizer_fingerprint']}
        if _verification_snapshot(root, p, jobs) != before:
            raise ValueError('Index inputs changed during assembly; verify again')
        atomic_json(temporary / 'index.json', result)
        sync_dir(temporary)
        os.replace(temporary, output)
        sync_dir(root)
        print(f'Published online index: {output / "index.json"}', flush=True)
        return result
