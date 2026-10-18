"""Publish a Malayalam-only revision; reuse immutable non-Malayalam pools."""
from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

from .common import LANGS, atomic_json, digest, load_plan, lock, sync_dir
from .index import (
    _assemble_language,
    _byte_ranges,
    _index_chunk,
    _scan_source,
    check_frozen,
    identity,
    tokenizer_from,
)


def _refresh_chunk(task):
    root, plan, chunk = task
    tokenizer = tokenizer_from(Path(root) / '_state/tokenizer')
    excluded = set(json.loads((Path(root) / '_state/excluded.json').read_text()))
    _index_chunk(root, plan, chunk, tokenizer, excluded, Path(root) / '_state/stop-refresh')
    return chunk['id']


def refresh_ml(source_index, destination, quarantine, jobs=8):
    """Re-decode only current Malayalam training sources and publish a new index.

    Non-ML sources and every eval source must still match their frozen metadata.
    Existing non-ML pools are referenced by symlink, never copied or modified.
    """
    source_index = Path(source_index).resolve()
    destination = Path(destination).resolve()
    if jobs < 1:
        raise ValueError('jobs must be positive')
    if destination.exists():
        raise ValueError('Destination exists; choose a new RUN_ID')
    old = json.loads(source_index.read_text())
    source_root = Path(old['root'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with lock(source_root / '_state/lifecycle.lock'), lock(destination.with_suffix('.refresh.lock')):
        frozen = load_plan(source_root)
        if frozen.get('derived_index'):
            raise ValueError('Refresh from the original full index, not an earlier revision')
        original_index_hash = digest(source_index)
        if digest(source_root / '_state/plan.json') != old['plan_sha256']:
            raise ValueError('Published index/plan hash mismatch')
        if old['sources'] != frozen['sources'] or old['config'] != frozen['config']:
            raise ValueError('Published sources/config differ from plan')
        ml_paths = {c['source'] for c in frozen['chunks'] if c['lang'] == 'ml'}
        if not ml_paths:
            raise ValueError('No Malayalam source chunks in original plan')
        unchanged = copy.deepcopy(frozen)
        unchanged['sources'] = [s for s in frozen['sources'] if s['path'] not in ml_paths]
        check_frozen(source_root, unchanged)
        for lang in LANGS:
            pool = source_index.parent / f'{lang}.sqlite'
            if pool.stat().st_size != old['pools'][lang]['size']:
                raise ValueError(f'Published pool size changed: {pool}')
        quarantine = Path(quarantine).resolve()
        quarantine_hash = digest(quarantine)
        excluded = set(json.loads((source_root / '_state/excluded.json').read_text()))
        quarantine_names = set()
        with quarantine.open() as stream:
            for line in stream:
                if line.strip():
                    audio = json.loads(line).get('audio_filepath')
                    if not isinstance(audio, str) or not audio:
                        raise ValueError('Quarantine row requires audio_filepath')
                    quarantine_names.add(identity(audio))
        excluded.update(quarantine_names)
        print(f'refresh-ml: freezing {len(ml_paths)} manifests; excluding {len(quarantine_names):,} recordings', flush=True)
        revised = {path: _scan_source(path) for path in sorted(ml_paths)}
        plan = copy.deepcopy(frozen)
        plan['root'] = str(destination)
        plan['sources'] = [revised.get(s['path'], s) for s in frozen['sources']]
        plan['chunks'] = []
        for path in sorted(ml_paths):
            for start, end in _byte_ranges(path, plan['config']['chunk_mb'] << 20):
                plan['chunks'].append({'id': f"{len(plan['chunks']):07d}", 'lang': 'ml',
                                       'source': path, 'start': start, 'end': end})
        plan['derived_index'] = {'kind': 'refresh-ml', 'source_index': str(source_index),
                                 'source_index_sha256': original_index_hash,
                                 'quarantine_sha256': quarantine_hash}
        plan['baseline_records'] = frozen['baseline_records'] + sum(
            revised[s['path']]['rows'] - s['rows'] for s in frozen['sources'] if s['path'] in revised)
        # Private staging prevents a partial refresh from becoming a loadable index.
        staging = Path(tempfile.mkdtemp(prefix=destination.name + '.building-', dir=destination.parent))
        try:
            state = staging / '_state'
            state.mkdir()
            shutil.copytree(source_root / '_state/tokenizer', state / 'tokenizer')
            shutil.copyfile(quarantine, state / 'quarantine.jsonl')
            atomic_json(state / 'excluded.json', sorted(excluded))
            plan['excluded_sha256'] = digest(state / 'excluded.json')
            atomic_json(state / 'plan.json', plan)
            with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context('spawn')) as pool:
                pending = {pool.submit(_refresh_chunk, (str(staging), plan, c)) for c in plan['chunks']}
                total = len(pending)
                print(f'refresh-ml: processing {total} Malayalam chunks with {jobs} workers', flush=True)
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                    print(f'refresh-ml: {total - len(pending)}/{total} Malayalam chunks complete; '
                          f'{len(pending)} pending', flush=True)
            assembled = staging / 'assembled'
            assembled.mkdir()
            _, ml = _assemble_language((str(staging), str(assembled), 'ml', plan['chunks']))
            if ml['count'] < 2:
                raise ValueError('Too few eligible Malayalam partners')
            pools = copy.deepcopy(old['pools'])
            pools['ml'] = ml
            with sqlite3.connect(assembled / 'ml.sqlite') as db:
                for lang in LANGS:
                    if lang == 'ml':
                        continue
                    original_pool = (source_index.parent / f'{lang}.sqlite').resolve()
                    print(f'refresh-ml: checking Malayalam overlap with {lang}', flush=True)
                    db.execute('ATTACH DATABASE ? AS other', (original_pool.as_uri() + '?mode=ro',))
                    if any(db.execute('SELECT 1 FROM other.pool WHERE audio_key=?', (name,)).fetchone()
                           for name in quarantine_names):
                        raise ValueError(f'Quarantine intersects unchanged {lang} pool; cannot reuse it')
                    if db.execute('SELECT 1 FROM main.pool a CROSS JOIN other.pool b '
                                  'WHERE a.audio_key=b.audio_key LIMIT 1').fetchone():
                        raise ValueError(f'Malayalam overlaps {lang} partner pool')
                    db.execute('DETACH DATABASE other')
                    (assembled / f'{lang}.sqlite').symlink_to(original_pool)
            reuse = dict.fromkeys(LANGS, 0)
            for pair, count in old['pair_targets'].items():
                a, b = pair.split('-')
                reserved = count * plan['config']['max_attempts']
                if reserved > pools[a]['count'] * pools[b]['count']:
                    raise ValueError(f'Insufficient combinations for {pair}')
                for lang in (a, b):
                    reuse[lang] += (2 if max(plan['config']['segment_counts']) > 2 else 1) * math.ceil(reserved / pools[lang]['count'])
            if max(reuse.values()) > plan['config']['max_source_uses']:
                raise ValueError('Revised pool exceeds preview reuse limit')
            result = {**old, 'root': str(destination), 'sources': plan['sources'], 'pools': pools,
                      'baseline_records': plan['baseline_records'], 'source_use_upper_bounds': reuse,
                      'plan_sha256': digest(state / 'plan.json')}
            check_frozen(staging, plan)
            if digest(quarantine) != quarantine_hash or digest(state / 'quarantine.jsonl') != quarantine_hash:
                raise ValueError('Quarantine changed during refresh')
            if digest(source_index) != original_index_hash:
                raise ValueError('Source index changed during refresh')
            atomic_json(assembled / 'index.json', result)
            sync_dir(assembled)
            if destination.exists():
                raise ValueError('Destination appeared during refresh')
            os.rename(staging, destination)
            sync_dir(destination.parent)
        except BaseException:
            shutil.rmtree(staging)
            raise
        print(f"Published: {destination / 'assembled/index.json'}; keep original pools in place", flush=True)
        return result
