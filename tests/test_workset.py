"""Miniature exhaustive-ingestion contracts; no audio or external services."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import sqlite3
from contextlib import closing
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from data_processing import workset as ws


def _line(row):
    return (json.dumps(row, ensure_ascii=False) + '\n').encode('utf-8')


def _row(path='/audio/a.wav', text='This is the original spoken sentence.', duration=2, **extra):
    return {'audio_filepath': path, 'text': text, 'duration': duration, **extra}


def _fixture(directory, sources):
    roots = {'v76': directory / 'v76', 'sft': directory / 'sft'}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    contract = {'train_manifest': {}, 'eval_manifest': {}}
    for (tree, lang, name), rows in sources.items():
        path = roots[tree] / lang / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = rows if isinstance(rows, bytes) else b''.join(_line(row) for row in rows)
        path.write_bytes(gzip.compress(raw, mtime=0) if name.endswith('.gz') else raw)
        split = 'eval' if name.startswith('eval') else 'train'
        contract[f'{split}_manifest'].setdefault(lang, []).append(str(path))
    path = directory / 'contract.yaml'
    path.write_text(yaml.safe_dump(contract, sort_keys=False))
    return {'contract_path': path, 'v76_root': roots['v76'], 'sft_root': roots['sft']}


def _prepare(root, options, nodes=1, jobs=1):
    return [ws.prepare_rank(root, **options, rank=rank, nodes=nodes, jobs=jobs, run_id='test')
            for rank in range(nodes)]


def _prepared_rows(root, reports):
    return sorted((row for report in reports for segment in report['segments']
                   for chunk in segment['chunks'] for row in ws._verified_rows(root, chunk)),
                  key=lambda row: (row['contract_index'], row['byte_offset']))


def _build(root, options, langs=('en',), nodes=1, jobs=1):
    _prepare(root, options, nodes, jobs)
    for lang in langs:
        ws.build_language(root, lang, run_id='test')
    return ws.finalize_worksets(root)


def _items(root, lang='en', parts=1):
    ws.partition_language(root, lang, parts)
    return [row for part in range(parts) for row in ws.iter_partition(root, lang, part)]


def test_contract_exact_order_inventory_and_no_discovery(tmp_path):
    options = _fixture(tmp_path, {
        ('sft', 'zh', 'train_z.jsonl'): [],
        ('v76', 'en', 'train_second.jsonl'): [],
        ('v76', 'en', 'nested/train_first.jsonl'): [],
        ('sft', 'en', 'eval_test.jsonl'): [],
    })
    (options['v76_root'] / 'en/undeclared.jsonl').write_text('not input')
    specs = ws.load_contract(**options)
    assert [s['contract_index'] for s in specs] == list(range(4))
    assert [s['lang'] for s in specs] == ['en', 'en', 'zh', 'en']
    assert specs[1]['relative_path'] == 'en/nested/train_first.jsonl'
    assert specs[-1]['split'] == 'eval'
    inventory = ws.input_inventory(**options)
    assert len(inventory['files']) == 4
    assert all(f['stat']['size'] == 0 for f in inventory['files'])
    json.dumps(inventory)


@pytest.mark.parametrize('entry', ['/external/en/train.jsonl', '../en/train.jsonl',
                                  'https://example.com/train.jsonl',
                                  'training_manifests/v7.6/en/*.jsonl',
                                  'training_manifests/v7.6/../en/train.jsonl',
                                  '/evil/q3asr_sft_manifests/en/train.jsonl'])
def test_contract_rejects_external_glob_and_traversal(tmp_path, entry):
    options = _fixture(tmp_path, {('v76', 'en', 'train.jsonl'): []})
    options['contract_path'].write_text(yaml.safe_dump({'train_manifest': {'en': [entry]}, 'eval_manifest': {}}))
    with pytest.raises(ValueError):
        ws.load_contract(**options)


def test_contract_missing_eval_and_duplicate_and_symlink_escape(tmp_path):
    options = _fixture(tmp_path, {('v76', 'en', 'train.jsonl'): [], ('sft', 'en', 'eval.jsonl'): []})
    contract = yaml.safe_load(options['contract_path'].read_text())
    contract['eval_manifest']['en'][0] = str(options['sft_root'] / 'en/missing.jsonl')
    options['contract_path'].write_text(yaml.safe_dump(contract))
    with pytest.raises(FileNotFoundError):
        ws.load_contract(**options)
    original = contract['train_manifest']['en'][0]
    contract['eval_manifest']['en'] = [original]
    options['contract_path'].write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match='duplicate'):
        ws.load_contract(**options)
    outside = tmp_path / 'outside.jsonl'
    outside.write_text('')
    link = options['sft_root'] / 'en/link.jsonl'
    link.symlink_to(outside)
    contract['eval_manifest']['en'] = [str(link)]
    options['contract_path'].write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match='symlink'):
        ws.load_contract(**options)


def test_approved_cluster_roots_map_exactly_without_basename_flattening(tmp_path):
    options = _fixture(tmp_path, {('v76', 'en', 'nested/train.jsonl'): [_row()]})
    entry = str(ws._DECLARED_ROOTS['v76'] / 'en/nested/train.jsonl')
    contract = {'train_manifest': {'en': [entry]}, 'eval_manifest': {}}
    options['contract_path'].write_text(yaml.safe_dump(contract))
    assert ws.load_contract(**options)[0]['path'] == str(options['v76_root'] / 'en/nested/train.jsonl')
    contract['train_manifest']['en'] = ['training_manifests/v7.6/en/nested/train.jsonl']
    options['contract_path'].write_text(yaml.safe_dump(contract))
    assert ws.load_contract(**options)[0]['relative_path'] == 'en/nested/train.jsonl'


def test_serial_parallel_byte_edges_unicode_gzip_and_every_row(tmp_path):
    raw = (_line(_row(text='مرحبا café 中文')) + b'\n  \r\n{broken}\n[1,2]\n'
           + b'\xff\n' + _line(_row('/audio/b.wav', duration=None))
           + _line(_row('/audio/c.wav', text=42)) + _line(_row('/audio/d.wav'))[:-1])
    options = _fixture(tmp_path / 'inputs', {
        ('v76', 'en', 'train.jsonl'): raw,
        ('sft', 'en', 'train_gzip.jsonl.gz'): [_row('/gzip/a.wav'), _row('/gzip/b.wav')],
        ('sft', 'zh', 'eval.jsonl'): [{'audio': '/excluded.wav', 'text': 'language Chinese<asr_text>你好'}],
    })
    serial_root, parallel_root = tmp_path / 'serial', tmp_path / 'parallel'
    serial = _prepare(serial_root, options)
    parallel = _prepare(parallel_root, options, nodes=2, jobs=3)
    left, right = _prepared_rows(serial_root, serial), _prepared_rows(parallel_root, parallel)
    assert left == right
    assert len(left) == len(raw.splitlines()) + 3
    assert len({row['row_id'] for row in right}) == len(right)
    assert any('blank_line' in row['metadata_errors'] for row in right)
    assert any('malformed_json' in row['metadata_errors'] for row in right)
    assert any('not_an_object' in row['metadata_errors'] for row in right)
    segments = sorted((segment for report in parallel for segment in report['segments']
                       if segment['path'].endswith('/train.jsonl') and segment['end'] > segment['start']),
                      key=lambda segment: segment['start'])
    assert segments[0]['start'] == 0 and segments[-1]['end'] == len(raw)
    assert all(a['end'] == b['start'] for a, b in pairwise(segments))
    for segment in segments:
        assert segment['sha256'] == hashlib.sha256(raw[segment['start']:segment['end']]).hexdigest()
    assert parallel[1]['eval_rows'] == 0
    for row in right:
        assert len(base64.b64decode(row['raw_base64'])) == row['raw_bytes']


def test_all_newline_boundary_positions_and_eighty_real_task_buckets(tmp_path):
    raw = b'\n' + _line(_row(text='你好 café')) + b'\r\n' + _line(_row('/audio/z.wav'))[:-1]
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): raw})
    spec = ws.input_inventory(**options)['files'][0]
    plans = ws._plan_segments([spec], nodes=1, jobs=80)
    assert len(plans) == 80 and all(plans)
    root = tmp_path / 'boundaries'
    result = []
    for number, spans in enumerate(plans):
        result.extend(ws._prepare_task((root, root / f'w{number}', spans)))
    records = _prepared_rows(root, [{'segments': result}])
    assert b''.join(base64.b64decode(row['raw_base64']) for row in records) == raw
    assert [row['byte_offset'] for row in records] == [0, 1, 1 + len(_line(_row(text='你好 café'))),
                                                       3 + len(_line(_row(text='你好 café')))]


def test_gzip_serial_split_and_no_audio_probe(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('audio must not be probed')

    from data_processing import duration_probe

    monkeypatch.setattr(duration_probe, 'probe_duration', forbidden)
    monkeypatch.setattr(ws, 'CHUNK_ROWS', 2)
    options = _fixture(tmp_path / 'inputs', {('sft', 'en', 'train.jsonl.gz'): [_row(f'/not-real/{i}.wav') for i in range(7)]})
    root = tmp_path / 'run'
    reports = _prepare(root, options)
    segment = reports[0]['segments'][0]
    assert [c['rows'] for c in segment['chunks']] == [2, 2, 2, 1]
    assert segment['offset_space'] == 'uncompressed'
    assert segment['content_sha256'] == hashlib.sha256(Path(segment['path']).read_bytes()).hexdigest()
    ws.build_language(root, 'en', run_id='test')
    assert ws.finalize_worksets(root)['en']['eligible'] == 7


def test_best_representative_duration_salvage_and_all_provenance(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [
        _row('/same.wav', 'Lower quality sentence.', 4, quality=0.2),
        _row('/same.wav', 'Best original sentence.', None, quality=0.9),
        _row('/same.wav', None, 4.02, quality=1),
        _row('/repeat-a.wav', 'Repeated sentence.', 2),
        _row('/repeat-b.wav', 'Repeated sentence.', 2),
    ]})
    root = tmp_path / 'run'
    inventory = _build(root, options)['en']
    assert inventory['source_rows'] == 5
    assert inventory['unique'] == inventory['eligible'] == 3
    assert inventory['duplicate'] == inventory['source_invalid'] == 2
    assert sum(inventory['dispositions'].values()) == 5
    items = _items(root)
    best = next(row for row in items if row['audio_filepath'] == '/same.wav')
    assert best['text'] == 'Best original sentence.' and best['duration'] == 4
    assert best['provenance_count'] == 3
    assert 'duration_borrowed_same_path' in best['normalization']
    assert inventory['bytes'] == sum(len(item['text'].encode('utf-8')) for item in items)
    reference = best['provenance_ref']
    with closing(sqlite3.connect(root / reference['database'])) as db:
        provenance = db.execute('SELECT payload,disposition FROM occurrences WHERE clip_id=?', (best['id'],)).fetchall()
    assert len(provenance) == 3
    assert {status for _, status in provenance} == {'eligible', 'duplicate'}
    assert all(json.loads(payload)['raw_base64'] for payload, _ in provenance)
    assert ws.build_language(root, 'en', run_id='test') == inventory


@pytest.mark.parametrize(('durations', 'blocked'), [
    ([None, 2], None), ([None, None], 'missing_duration'), ([1, 1.1], None),
    ([10, 10.1], None), ([10, 10.1002], 'duration_conflict'),
    ([1, 1.21, 1.09], 'duration_conflict'), ([0, 2], None),
    ([False], 'invalid_duration'), ([float('nan')], 'invalid_duration'),
    ([float('inf')], 'invalid_duration'), ([-2], 'invalid_duration'),
])
def test_duration_metadata_rules(tmp_path, durations, blocked):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [
        _row(duration=duration, quality=1 - i * 0.1) for i, duration in enumerate(durations)]})
    root = tmp_path / 'run'
    inv = _build(root, options)['en']
    item, = _items(root)
    assert item.get('blocked_reason') == blocked
    assert inv['blocked'] == int(blocked is not None)
    assert inv['eligible'] == int(blocked is None)
    assert inv['shards'] == 1 and inv['est_rows'] == 1


def test_eval_union_includes_other_languages_and_sft_envelopes(tmp_path):
    options = _fixture(tmp_path / 'inputs', {
        ('v76', 'en', 'train.jsonl'): [_row('/other-lang-eval.wav'), _row('/sft-eval.wav'), _row('/keep.wav')],
        ('v76', 'ar', 'eval.jsonl'): [{'wav_path': '/other-lang-eval.wav', 'transcript': 'مرحبا'}],
        ('sft', 'zh', 'eval.jsonl'): [{'audio': {'path': '/sft-eval.wav'}, 'text': 'language Chinese<asr_text>你好'}],
    })
    root = tmp_path / 'run'
    inv = _build(root, options)['en']
    assert inv['unique'] == 3 and inv['eval_excluded'] == 2 and inv['eligible'] == 1
    assert [row['audio_filepath'] for row in _items(root)] == ['/keep.wav']
    with closing(sqlite3.connect(root / inv['eval_index']['path'])) as db:
        assert db.execute('SELECT COUNT(*) FROM occurrences').fetchone()[0] == 2
    with closing(sqlite3.connect(root / inv['database']['path'])) as db:
        assert db.execute("SELECT COUNT(*) FROM occurrences WHERE disposition='eval_excluded'").fetchone()[0] == 2


def test_cross_language_conflicts_and_invalid_rows_remain_cataloged(tmp_path):
    options = _fixture(tmp_path / 'inputs', {
        ('v76', 'ar', 'train.jsonl'): [_row('/shared.wav', 'مرحبا بالعالم')],
        ('v76', 'en', 'train.jsonl'): [_row('/shared.wav'), _row(path=None), _row('/text-missing.wav', text=None)],
    })
    root = tmp_path / 'run'
    _prepare(root, options)
    ws.build_language(root, 'en', run_id='test')
    with pytest.raises(FileNotFoundError):
        ws.finalize_worksets(root)
    ws.build_language(root, 'ar', run_id='test')
    inventory = ws.finalize_worksets(root)
    assert inventory['ar']['eligible'] == inventory['en']['eligible'] == 0
    assert inventory['ar']['blocked'] == 1 and inventory['en']['blocked'] == 2
    assert all(info['shards'] == 1 for info in inventory.values())
    assert inventory['en']['source_invalid_unidentified'] == 1
    assert _items(root, 'ar')[0]['blocked_reason'] == 'cross_language_conflict'
    assert {item['blocked_reason'] for item in _items(root, 'en')} == {'cross_language_conflict', 'missing_or_invalid_text'}
    assert ws.finalize_worksets(root) == inventory


def test_ties_exact_paths_safe_normalization_and_repairable_flags(tmp_path):
    options = _fixture(tmp_path / 'inputs', {
        ('v76', 'ar', 'train_first.jsonl'): [_row('/exact/../a.wav', 'ٱلله 111111 [ضحك]', 2, quality=0), _row('/same.wav', 'الأول', 2, quality=0.5)],
        ('sft', 'ar', 'train_second.jsonl'): [{'audio': '/same.wav', 'text': 'language Arabic<asr_text>الثاني', 'duration': 2, 'quality': 0.5}, _row('/a.wav', 'نعم نعم نعم نعم نعم نعم نعم', 2)],
    })
    root = tmp_path / 'run'
    inv = _build(root, options, langs=('ar',))['ar']
    items = _items(root, 'ar')
    assert inv['eligible'] == 3 and inv['blocked'] == 0
    assert next(item for item in items if item['audio_filepath'] == '/same.wav')['text'] == 'الأول'
    flagged = next(item for item in items if item['audio_filepath'] == '/exact/../a.wav')
    assert 'ٱ' in flagged['normalized_text'] and '111111' in flagged['normalized_text']
    assert flagged['repairable_flags'] and flagged['quality'] == 0


def test_physical_partitions_deterministic_bounded_and_repeated_text_uncapped(tmp_path, monkeypatch):
    monkeypatch.setattr(ws, 'TRANSACTION_ROWS', 3)
    monkeypatch.setattr(ws, 'TRANSACTION_BYTES', 256)
    monkeypatch.setattr(ws, 'MAX_OPEN_PARTS', 2)
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [
        _row(f'/audio/{i}.wav', 'Identical café transcript.', 2, quality=(i % 10) / 10) for i in range(131)]})
    checksums = []
    for name, nodes, jobs in [('serial', 1, 1), ('parallel', 2, 2)]:
        root = tmp_path / name
        inv = _build(root, options, nodes=nodes, jobs=jobs)['en']
        assert inv['eligible'] == inv['est_rows'] == 131 and inv['needs_llm_fraction'] == 1
        report = ws.partition_language(root, 'en', 9)
        items = [item for part in range(9) for item in ws.iter_partition(root, 'en', part)]
        assert len(items) == len({item['id'] for item in items}) == 131
        assert report['rows'] == sum(part['rows'] for part in report['parts']) == 131
        assert len(report['parts']) == 9
        for part in report['parts']:
            data = (root / part['path']).read_bytes()
            assert len(data) == part['bytes'] and hashlib.sha256(data).hexdigest() == part['sha256']
            rows = list(ws.iter_partition(root, 'en', part['part']))
            assert list(ws.iter_partition(root, 'en', part['part'], start=2)) == rows[2:]
            assert all(int.from_bytes(hashlib.blake2b(row['audio_filepath'].encode(), digest_size=8).digest(), 'big') % 9 == part['part'] for row in rows)
            assert [row['quality'] for row in rows] == sorted((row['quality'] for row in rows), reverse=True)
        checksums.append([part['sha256'] for part in report['parts']])
        assert ws.partition_language(root, 'en', 9) == report
        with pytest.raises(RuntimeError, match='repartition'):
            ws.partition_language(root, 'en', 2)
    assert checksums[0] == checksums[1]


def test_resume_requires_complete_compatible_unchanged_inputs_and_outputs(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row()]})
    root = tmp_path / 'run'
    report = _prepare(root, options)[0]
    assert _prepare(root, options)[0] == report
    with pytest.raises(RuntimeError, match='incompatible'):
        ws.prepare_rank(root, **options, rank=0, nodes=1, jobs=2, run_id='test')
    chunk = report['segments'][0]['chunks'][0]
    (root / chunk['path']).write_text('tampered\n')
    with pytest.raises(RuntimeError, match='changed'):
        _prepare(root, options)
    source = options['v76_root'] / 'en/train.jsonl'
    source.write_bytes(_line(_row('/new.wav')))
    with pytest.raises(RuntimeError, match='incompatible'):
        _prepare(root, options)


def test_fatal_io_or_commit_failure_does_not_publish_success_and_retries_cleanly(tmp_path, monkeypatch):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row(), _row('/b.wav')]})
    root = tmp_path / 'run'
    original = ws._atomic_json

    def fail_commit(path, value):
        if path.name == 'rank0.json' and value['status'] == 'complete':
            raise OSError('injected commit failure')
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(ws, '_atomic_json', fail_commit)
        with pytest.raises(OSError, match='commit failure'):
            _prepare(root, options)
    assert ws._read_json(root / 'prepare/rank0.json')['status'] == 'fatal'
    with pytest.raises(RuntimeError, match='incomplete'):
        ws.build_language(root, 'en', run_id='test')
    report = _prepare(root, options)[0]
    assert report['train_rows'] == 2
    assert len(_prepared_rows(root, [report])) == 2


def test_source_changes_during_stream_fail_closed(tmp_path, monkeypatch):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row()]})
    source = options['v76_root'] / 'en/train.jsonl'
    original = ws._canonical_row

    def mutate(raw, spec, offset):
        result = original(raw, spec, offset)
        with source.open('ab') as handle:
            handle.write(b'\n')
        return result

    monkeypatch.setattr(ws, '_canonical_row', mutate)
    root = tmp_path / 'run'
    with pytest.raises(RuntimeError, match='changed'):
        _prepare(root, options)
    assert ws._read_json(root / 'prepare/rank0.json')['status'] == 'fatal'


def test_incomplete_rank_barrier_and_sqlite_durability(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row()]})
    root = tmp_path / 'run'
    ws.prepare_rank(root, **options, rank=0, nodes=2, jobs=1, run_id='test')
    with pytest.raises(FileNotFoundError):
        ws.build_language(root, 'en', run_id='test')
    with closing(ws._connect(tmp_path / 'settings.sqlite3')) as db:
        assert db.execute('PRAGMA journal_mode').fetchone()[0] == 'delete'
        assert db.execute('PRAGMA synchronous').fetchone()[0] == 2
        assert db.execute('PRAGMA temp_store').fetchone()[0] == 1


def test_extreme_numeric_metadata_and_repairable_controls(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [
        _row('/huge.wav', duration=10 ** 400, quality=10 ** 400),
        _row('/control.wav', text='A spoken\x00 sentence with artifacts.'),
        _row('/string-duration.wav', duration='2.5'),
        _row('/boolean-text.wav', text=True),
    ]})
    root = tmp_path / 'run'
    inv = _build(root, options)['en']
    assert inv['eligible'] == inv['blocked'] == 2
    items = {item['audio_filepath']: item for item in _items(root)}
    assert items['/huge.wav']['blocked_reason'] == 'invalid_duration'
    assert 'invalid_quality' in items['/huge.wav']['repairable_flags']
    assert '\x00' in items['/control.wav']['text']
    assert '\x00' not in items['/control.wav']['normalized_text']
    assert 'strip_artifacts' in items['/control.wav']['normalization']
    assert items['/string-duration.wav']['duration'] == 2.5


@pytest.mark.parametrize('failure', ['before_rename', 'after_rename', 'before_marker'])
def test_finalization_commit_replays_after_interruption(tmp_path, monkeypatch, failure):
    options = _fixture(tmp_path / 'inputs', {
        ('v76', 'en', 'train.jsonl'): [_row('/shared.wav')],
        ('v76', 'ar', 'train.jsonl'): [_row('/shared.wav', 'مرحبا بالعالم')],
    })
    root = tmp_path / 'run'
    _prepare(root, options)
    for lang in ('ar', 'en'):
        ws.build_language(root, lang, run_id='test')
    original_json, original_replace = ws._atomic_json, ws.os.replace

    def crash_json(path, value):
        if failure == 'after_rename' and path.name == 'ar.json' and value.get('finalized'):
            raise OSError('injected publication interruption')
        if failure == 'before_marker' and path.name == 'finalized.json':
            raise OSError('injected publication interruption')
        original_json(path, value)

    def crash_replace(source, target):
        if failure == 'before_rename' and 'finalize-' in str(source) and str(target).endswith('ar.sqlite3'):
            raise OSError('injected publication interruption')
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(ws, '_atomic_json', crash_json)
        patch.setattr(ws.os, 'replace', crash_replace)
        with pytest.raises(OSError, match='interruption'):
            ws.finalize_worksets(root)
    assert not (root / 'worksets/finalized.json').exists()
    inventory = ws.finalize_worksets(root)
    assert all(report['eligible'] == 0 and report['blocked'] == 1 for report in inventory.values())
    assert all(_items(root, lang)[0]['blocked_reason'] == 'cross_language_conflict' for lang in inventory)


def test_build_database_commit_replays_without_reading_segments_again(tmp_path, monkeypatch):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row()]})
    root = tmp_path / 'run'
    _prepare(root, options)
    original = ws._atomic_json

    def crash(path, value):
        if path.name == 'en.json':
            raise OSError('report publication failure')
        original(path, value)

    with monkeypatch.context() as patch:
        patch.setattr(ws, '_atomic_json', crash)
        with pytest.raises(OSError, match='publication'):
            ws.build_language(root, 'en', run_id='test')

    def no_reread(*args, **kwargs):
        raise AssertionError('committed database must replay without reingestion')

    monkeypatch.setattr(ws, '_verified_rows', no_reread)
    assert ws.build_language(root, 'en', run_id='test')['eligible'] == 1


def test_corrupt_workset_and_preparation_reports_fail_closed(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row()]})
    root = tmp_path / 'run'
    _prepare(root, options)
    info = ws.build_language(root, 'en', run_id='test')
    with closing(sqlite3.connect(root / info['database']['path'])) as db:
        db.execute("UPDATE clips SET blocked_reason='unrecorded change'")
        db.commit()
    with pytest.raises(RuntimeError, match='artifact changed'):
        ws.finalize_worksets(root)
    report = ws._read_json(root / 'prepare/rank0.json')
    report['train_rows'] += 1
    ws._atomic_json(root / 'prepare/rank0.json', report)
    with pytest.raises(RuntimeError, match='fingerprint'):
        _prepare(root, options)


def test_partition_reads_only_assigned_file_and_rejects_corruption(tmp_path, monkeypatch):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row(f'/{i}.wav') for i in range(20)]})
    root = tmp_path / 'run'
    _build(root, options)
    report = ws.partition_language(root, 'en', 3)
    original, opened = ws._verified_rows, []

    def tracking(root, descriptor):
        opened.append(descriptor['path'])
        yield from original(root, descriptor)

    monkeypatch.setattr(ws, '_verified_rows', tracking)
    list(ws.iter_partition(root, 'en', 1, start=1))
    assert opened == [report['parts'][1]['path']]
    with (root / report['parts'][1]['path']).open('ab') as handle:
        handle.write(b'{}\n')
    with pytest.raises(RuntimeError, match='changed'):
        list(ws.iter_partition(root, 'en', 1))


def test_exclusive_owner_never_steals_existing_owner(tmp_path):
    owner = tmp_path / 'owners/slice'
    with (ws._owner(owner, 'first'), pytest.raises((FileExistsError, RuntimeError)),
          ws._owner(owner, 'second')):
        pytest.fail('ownership was stolen')
    with ws._owner(owner, 'first'):
        assert owner.exists()


def test_duplicate_yaml_keys_are_not_silently_discarded(tmp_path):
    options = _fixture(tmp_path, {('v76', 'en', 'train.jsonl'): []})
    options['contract_path'].write_text('train_manifest: {}\ntrain_manifest: {}\neval_manifest: {}\n')
    with pytest.raises(ValueError, match='duplicate YAML'):
        ws.load_contract(**options)


def test_empty_inputs_idle_ranks_and_empty_physical_parts(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): []})
    root = tmp_path / 'run'
    reports = _prepare(root, options, nodes=3, jobs=2)
    assert all(report['counts']['rows'] == 0 for report in reports)
    assert sum(report['actual_tasks'] for report in reports) == 1
    info = ws.build_language(root, 'en', run_id='test')
    assert info['source_rows'] == info['candidate_count'] == info['bytes'] == 0
    ws.finalize_worksets(root)
    parts = ws.partition_language(root, 'en', 4)['parts']
    assert len(parts) == 4
    assert all(part['rows'] == part['bytes'] == 0 for part in parts)
    assert all(list(ws.iter_partition(root, 'en', part)) == [] for part in range(4))


def test_sft_training_envelope_and_same_file_byte_offset_tie(tmp_path):
    options = _fixture(tmp_path / 'inputs', {('sft', 'en', 'train.jsonl'): [
        {'audio': '/sft.wav', 'text': 'language English<asr_text>Original café sentence.', 'duration': 2, 'quality': 0.5},
        {'audio': '/sft.wav', 'text': 'language English<asr_text>Later sentence.', 'duration': 2, 'quality': 0.5},
    ]})
    root = tmp_path / 'run'
    info = _build(root, options)['en']
    item, = _items(root)
    assert info['duplicate'] == 1 and item['provenance_count'] == 2
    assert item['text'] == 'Original café sentence.'
    assert item['normalization'][0] == 'sft_envelope'


def test_midstream_io_error_is_fatal_not_row_quarantine(tmp_path, monkeypatch):
    options = _fixture(tmp_path / 'inputs', {('v76', 'en', 'train.jsonl'): [_row(), _row('/second.wav')]})
    root = tmp_path / 'run'
    original = ws._canonical_row

    def fail_after_first(raw, spec, offset):
        if offset:
            raise OSError('source I/O failure')
        return original(raw, spec, offset)

    monkeypatch.setattr(ws, '_canonical_row', fail_after_first)
    with pytest.raises(OSError, match='source I/O'):
        _prepare(root, options)
    report = ws._read_json(root / 'prepare/rank0.json')
    assert report['status'] == 'fatal' and report['error'] == 'OSError'
