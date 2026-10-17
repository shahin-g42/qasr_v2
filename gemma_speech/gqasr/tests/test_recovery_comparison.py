import sqlite3
from pathlib import Path

import pytest
from gqasr.comparison_audit import merge_shards, save_rows, shard_jobs
from gqasr.processing import LANGUAGES
from gqasr.recovery_comparison import select_multilingual


def test_selection_checks_all_languages_and_references(tmp_path):
    db_path = tmp_path/'index.sqlite'
    audio = tmp_path/'audio.wav'
    audio.touch()
    rows = []
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE samples (split TEXT,language TEXT,ordinal INTEGER,audio TEXT,text TEXT,duration REAL,source TEXT,line INTEGER)')
        for lang in LANGUAGES:
            db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?,?)', ('eval',lang,0,str(audio),'reference',1.,'source',1))
            rows.append(dict(id=f'eval:{lang}:0', language=lang, reference='reference'))
    predictions = tmp_path/'predictions.jsonl'
    save_rows(predictions, rows)
    samples = select_multilingual(db_path, predictions)
    assert {r['language'] for r in samples} == set(LANGUAGES)
    rows[0]['reference'] = 'incorrect'
    save_rows(predictions, rows)
    with pytest.raises(ValueError, match='Reference mismatch'):
        select_multilingual(db_path, predictions)
    save_rows(predictions, rows[1:])
    with pytest.raises(ValueError, match='all five'):
        select_multilingual(db_path, predictions)


def test_auto_only_shards_merge_without_forced_predictions(tmp_path):
    rows = [dict(id='eval:ar:0', reference='مرحبا', language='ar', audit_group='ar')]
    jobs = [dict(name='recovery', output=str(tmp_path/'merged'), modes=['auto'],
                 command=['worker', '--samples', 'x', '--output', 'y', '--modes', 'auto'])]
    tasks = shard_jobs(jobs, rows, tmp_path, 8)
    assert tasks[0]['modes'] == ['auto']
    shard = Path(tasks[0]['output'])
    shard.mkdir()
    save_rows(shard/'predictions-auto.jsonl', [{**rows[0], 'hypothesis':'مرحبا'}])
    merge_shards(jobs,tasks,rows)
    assert (tmp_path/'merged/predictions-auto.jsonl').exists()
    assert not (tmp_path/'merged/predictions-chinese.jsonl').exists()
