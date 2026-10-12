"""Shared index, virtual coverage and strict post-augmentation duration contracts."""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from data_processing.code_switch.common import LANGS, chunk_paths, lock, read_json
from data_processing.code_switch.index import (
    _index_chunk,
    allocate,
    assemble,
    plan,
    run,
    status,
    verify,
)
from qasr.augmentation import AudioAugmenter, AugmentationConfig
from qasr.code_switch import OnlineCodeSwitch, diagonal_pair
from qasr.data import ResilientAudioDataset


def make_tokenizer(path):
    backend = Tokenizer(WordLevel({'[UNK]': 0, '[PAD]': 1}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]')
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.fixture
def corpus(tmp_path):
    tok = make_tokenizer(tmp_path / 'tokenizer')
    cfg = yaml.safe_load(Path('configs/code_switch/full_v1.yaml').read_text())
    cfg.update(source_root=str(tmp_path / 'source'), tokenizer_path=str(tmp_path / 'tokenizer'),
               max_source_uses=200, sample_rate=8000)
    for lang in LANGS:
        folder = tmp_path / 'source' / lang / 'data'
        folder.mkdir(parents=True)
        rows = []
        for i in range(16):
            audio = tmp_path / f'{lang}-{i}.wav'
            waveform = np.full(4000 + 80 * i, (i + 1) / 100, dtype=np.float32)
            sf.write(audio, waveform, 8000)
            rows.append({'audio_filepath': str(audio), 'duration': len(waveform) / 8000,
                         'text': f'هٰذَا شَدَّة {lang} {i}'})
        # Eval overlap must never survive into the partner pools.
        (folder / f'eval_{lang}.jsonl').write_text(json.dumps(rows[0]) + '\n')
        (folder / f'train_{lang}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg))
    return tmp_path / 'run', config, tok


def complete(corpus):
    root, config, tokenizer = corpus
    frozen = plan(root, config, jobs=1, samples=28)
    run(root, procs=2)
    assert not verify(root, jobs=2, audio=True)
    index = assemble(root, jobs=2)
    return root, frozen, index, tokenizer


def test_end_to_end_resume_and_preview_samples(corpus):
    root, frozen, index, tokenizer = complete(corpus)
    st = status(root)
    assert st['done'] == st['chunks'] == 5
    assert st['processed'] == st['source_rows'] == 80
    assert st['eligible'] == 75
    assert st['rejects'] == {'eval_overlap': 5}
    assert sum(index['pair_targets'].values()) == 28
    for spec in index['pools'].values():
        assert spec['count'] == 15
    with pytest.raises(ValueError, match='Pilot'):
        OnlineCodeSwitch(root / 'assembled/index.json', tokenizer)
    mixer = OnlineCodeSwitch(root / 'assembled/index.json', tokenizer, allow_pilot=True)
    parts = mixer.parts()
    assert sum(map(len, parts)) == 28
    signatures = set()
    for part in parts:
        for i in range(len(part)):
            first, replay = part[i], part[i]
            np.testing.assert_array_equal(first['waveform'], replay['waveform'])
            assert first['text'] == replay['text']
            assert first['language'] == part.language
            assert {s['language'] for s in first['segments']} == set(part.pair.split('-'))
            assert all(not s['audio_filepath'].endswith('-0.wav') for s in first['segments'])
            assert 'شَدَّة' in first['text']
            assert first['duration'] == len(first['waveform']) / 8000 <= 35
            signature = tuple(s['audio_filepath'] for s in first['segments'])
            assert signature not in signatures
            signatures.add(signature)
    before = [read_json(chunk_paths(root, c)[2]) for c in frozen['chunks']]
    run(root, procs=2)
    assert before == [read_json(chunk_paths(root, c)[2]) for c in frozen['chunks']]
    assert not verify(root, jobs=1)
    with pytest.raises(ValueError, match='immutable'):
        verify(root, jobs=1, reset=True)


def test_committed_prefix_resume_and_corrupt_reset(corpus):
    root, config, tokenizer = corpus
    frozen = plan(root, config, jobs=1, samples=28)
    c = frozen['chunks'][0]
    stopped = root / 'stop-now'
    stopped.touch()
    excluded = set(read_json(root / '_state/excluded.json'))
    _index_chunk(root, frozen, c, tokenizer, excluded, stopped)
    part, rejects, progress, _ = chunk_paths(root, c)
    assert not read_json(progress)['done']
    part.write_bytes(b'uncommitted broken tail')
    rejects.write_bytes(b'also uncommitted')
    stopped.unlink()
    _index_chunk(root, frozen, c, tokenizer, excluded, stopped)
    assert read_json(progress)['kept'] == 15
    run(root, procs=1)
    part.write_bytes(b'broken committed output')
    assert c['id'] in verify(root, jobs=1, reset=True)
    assert not progress.exists()
    run(root, procs=2)
    assert not verify(root, jobs=1)


def test_frozen_sources_refuse_change(corpus):
    root, config, _ = corpus
    frozen = plan(root, config, jobs=1, samples=28)
    path = Path(frozen['sources'][0]['path'])
    path.write_text(path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        run(root, procs=1)


def test_cartesian_bijection_and_balanced_prefixes():
    for na, nb in [(3, 3), (4, 6), (7, 3), (1, 5)]:
        rows = [diagonal_pair(i, na, nb) for i in range(na * nb)]
        assert len(set(rows)) == na * nb
        for stop in range(1, len(rows) + 1):
            for axis, size in ((0, na), (1, nb)):
                counts = [sum(row[axis] == i for row in rows[:stop]) for i in range(size)]
                assert max(counts) - min(counts) <= 1


def _probe_lock(path, queue):
    try:
        with lock(path):
            queue.put('acquired')
    except BlockingIOError:
        queue.put('busy')


def test_exclusive_claims_across_processes(tmp_path):
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    path = tmp_path / 'lock'
    with lock(path):
        child = ctx.Process(target=_probe_lock, args=(path, queue))
        child.start()
        assert queue.get(timeout=30) == 'busy'
        child.join(timeout=30)
        assert child.exitcode == 0
    with lock(path):
        pass


def test_speed_perturbation_respects_final_limit():
    cfg = AugmentationConfig(speed_perturb={'enabled': True, 'rates': [0.85], 'p': 1.0})
    augmenter = AudioAugmenter(cfg, sampling_rate=100)
    waveform = np.ones(3500, dtype=np.float32)
    assert len(augmenter.augment_waveform(waveform)) > 3500  # old configs retain behaviour
    assert len(augmenter.augment_waveform(waveform, max_audio_seconds=35)) == 3500
    short = np.ones(1000, dtype=np.float32)
    assert len(augmenter.augment_waveform(short, max_audio_seconds=35)) > 1000


def test_resilient_wrapper_uses_composed_audio():
    original = np.ones(8000, dtype=np.float32)
    feature = {'waveform': original, 'audio_path': '/must-not-be-opened.wav', 'text': 'hello'}
    dataset = ResilientAudioDataset([feature], sampling_rate=8000,
        min_audio_seconds=0.1, max_audio_seconds=35)
    assert dataset[0]['waveform'] is original


def test_exact_target_allocation():
    weights = {'ar-en': 40, 'hi-en': 20, 'ml-en': 5, 'zh-en': 5, 'ar-hi': 20, 'ar-ml': 5, 'ar-zh': 5}
    assert sum(allocate(39678685, weights).values()) == 39678685


def test_config_validates_probability():
    from qasr.config import TrainConfig
    cfg = TrainConfig.from_yaml('configs/v8.0/02_full_10node.yaml')
    cfg.code_switch_index = '/index.json'
    cfg.validate()
    cfg.code_switch_probability = 1.1
    with pytest.raises(ValueError, match='probability'):
        cfg.validate()


def test_large_virtual_smoke_subset_does_not_materialize_population():
    from qasr.train import _sample_subset

    class Huge:
        def __len__(self):
            return 1_000_000_000

    subset = _sample_subset(Huge(), 16, 42)
    assert len(subset) == 16
    assert len(set(subset.indices)) == 16
    assert subset.indices == _sample_subset(Huge(), 16, 42).indices


def test_training_keeps_original_parts_and_budget(corpus):
    from types import SimpleNamespace

    from qasr.config import TrainConfig
    from qasr.data import JsonlSpeechDataset
    from qasr.train import _build_code_switcher
    root, config_path, tokenizer = corpus
    plan(root, config_path, jobs=1)  # production semantics, no pilot override
    run(root, procs=2)
    assemble(root, jobs=1)
    cfg = TrainConfig.from_yaml('configs/v8.1/02_full_10node.yaml')
    source = Path(yaml.safe_load(config_path.read_text())['source_root'])
    cfg.train_manifest = {lang: [str(source / lang / 'data' / f'train_{lang}.jsonl')] for lang in LANGS}
    cfg.code_switch_index = str(root / 'assembled/index.json')
    parts = [JsonlSpeechDataset(path, language=lang, min_duration_seconds=0.1, max_duration_seconds=35)
             for path, lang in cfg.train_manifest_specs]
    originals = list(parts)
    args = SimpleNamespace(world_size=2, should_save=True, max_steps=-1, output_dir=str(root / 'training'))
    processor = SimpleNamespace(tokenizer=tokenizer, feature_extractor=SimpleNamespace(sampling_rate=8000))
    _build_code_switcher(cfg, processor, args, parts)
    assert parts[:5] == originals
    assert parts == originals
    assert sum(map(len, parts)) == 80
    assert args.max_steps == -1
    assert cfg.max_steps == 75469
    cfg.validate()
    receipt = Path(args.output_dir) / 'code_switch_index.json'
    assert receipt.is_file()
    receipt.write_text(json.dumps({'sha256': 'a different dataset'}))
    with pytest.raises(ValueError, match='different code-switch index'):
        _build_code_switcher(cfg, processor, args, list(originals))


def test_lifecycle_lock_prevents_reset_and_double_run(corpus):
    root, config, _ = corpus
    plan(root, config, jobs=1, samples=28)
    with lock(root / '_state/lifecycle.lock', shared=True):
        with pytest.raises(BlockingIOError):
            verify(root, jobs=1, reset=True)
        with pytest.raises(BlockingIOError):
            assemble(root, jobs=1)
    import socket
    host = socket.gethostname().split('.')[0]
    with lock(root / '_state' / f'node-{host}.lock'), pytest.raises(BlockingIOError):
        run(root, procs=1)


def test_insufficient_reuse_budget_refuses_publication(corpus):
    root, config, _ = corpus
    cfg = yaml.safe_load(config.read_text())
    cfg['max_source_uses'] = 1
    config.write_text(yaml.safe_dump(cfg))
    plan(root, config, jobs=1, samples=28, limit=8)
    run(root, procs=2)
    assert status(root)['processed'] == status(root)['source_rows'] == 40
    with pytest.raises(ValueError, match='reuse limit'):
        assemble(root, jobs=1)
    assert (root / '_reports/inventory.json').exists()
    assert not (root / 'assembled/index.json').exists()


def test_major_label_whole_segment_budget_and_audio_mutation(corpus):
    root, config, tokenizer = corpus
    cfg = yaml.safe_load(config.read_text())
    cfg.update(language_policy='major', max_source_seconds=0.66, max_seconds=1.5,
               segment_counts=[4])
    config.write_text(yaml.safe_dump(cfg))
    plan(root, config, jobs=1, samples=28)
    run(root, procs=2)
    assemble(root, jobs=1)
    mixer = OnlineCodeSwitch(root / 'assembled/index.json', tokenizer, allow_pilot=True)
    for part in mixer.parts():
        row = part[0]
        assert len(row['segments']) == 2  # four clips cannot fit; remove whole optional clips
        assert row['duration'] <= 1.5
        frames = {s['language']: s['frames'] for s in row['segments']}
        assert frames[row['language']] == max(frames.values())
        assert row['text'] == ' '.join(s['text'] for s in row['segments'])
    part = mixer.parts()[0]
    row = part[0]
    audio = row['segments'][0]['audio_filepath']
    data, rate = sf.read(audio)
    sf.write(audio, -data, rate)  # same duration, different samples
    with pytest.raises(RuntimeError, match='Indexed source audio changed'):
        part[0]
    assert verify(root, jobs=1, audio=True)  # must catch same-length source mutation


def test_index_rejects_overlong_transcript_before_audio_decode(corpus):
    from data_processing.code_switch.index import _row
    _root, config, tokenizer = corpus
    cfg = yaml.safe_load(config.read_text())
    raw = json.dumps({'audio_filepath': '/does-not-need-decoding.wav', 'text': 'word ' * 600})
    row, reason = _row(raw, {'source': 'source.jsonl', 'lang': 'ar'}, 0, cfg, set(), tokenizer)
    assert row is None
    assert reason == 'source_tokens'


def test_stop_marks_sessions_without_signalling_unrelated_processes(tmp_path):
    from data_processing.code_switch.common import atomic_json
    from data_processing.code_switch.index import stop_run
    atomic_json(tmp_path / '_state/sessions/a.json', {'host': 'other-node', 'pid': 1})
    stop_run(tmp_path, all_nodes=True)
    assert (tmp_path / '_state/stop/a').is_file()


def test_launcher_uses_new_config_and_port():
    import os
    import subprocess
    result = subprocess.run(['bash', 'scripts/run_v81_phase2_node.sh', '9', 'master'],
        env={**os.environ, 'DRY_RUN': '1'}, text=True, capture_output=True, check=True)
    assert 'configs/v8.1/02_full_10node.yaml' in result.stdout
    assert '--master_port=29581' in result.stdout
    assert '--node_rank=9' in result.stdout


def test_verify_reports_corrupt_progress_as_bad_chunk(tmp_path):
    from data_processing.code_switch.index import _verify_one
    progress = tmp_path / '_state/progress/0000000.json'
    progress.parent.mkdir(parents=True)
    progress.write_text('{broken progress')
    cid, error = _verify_one((tmp_path, {'id': '0000000', 'end': 0}, {}, False, False))
    assert cid == '0000000'
    assert error


def test_bernoulli_mixing_preserves_anchor_and_reports_fallback(corpus):
    import random
    root, _, _, tokenizer = complete(corpus)
    mixer = OnlineCodeSwitch(root / 'assembled/index.json', tokenizer, allow_pilot=True)
    anchor = {'audio_path': str(root.parent / 'ar-1.wav'), 'language': 'ar',
              'waveform': np.ones(4000, dtype=np.float32), 'text': 'هٰذَا شَدَّة'}
    clean, audit = mixer.augment(anchor, 0, rng=random.Random(1))
    assert clean is anchor and not audit['triggered']
    rng = random.Random(912)
    triggered = applied = 0
    pairs = set()
    for _ in range(1000):
        mixed, audit = mixer.augment(anchor, 0.4, rng=rng)
        triggered += audit['triggered']
        applied += audit['applied']
        if audit['applied']:
            pairs.add(audit['pair'])
            np.testing.assert_array_equal(mixed['waveform'][:4000], anchor['waveform'])
            assert mixed['text'].startswith(anchor['text'] + ' ')
            assert mixed['text'] == ' '.join(s['text'] for s in mixed['segments'])
            assert mixed['language'] == 'ar'
            assert mixed['duration'] <= 35
        else:
            assert mixed is anchor
    assert 330 < triggered < 470
    assert applied == triggered
    assert pairs == {'ar-en', 'ar-hi', 'ar-ml', 'ar-zh'}
    long = {**anchor, 'waveform': np.ones(35 * 8000, dtype=np.float32)}
    unchanged, audit = mixer.augment(long, 1, rng=random.Random(1))
    assert unchanged is long and audit == {'triggered': True, 'applied': False, 'reason': 'duration_budget'}
    for lang in LANGS:
        row = {**anchor, 'language': lang}
        first, _ = mixer.augment(row, 1, rng=random.Random(77))
        second, _ = mixer.augment(row, 1, rng=random.Random(77))
        np.testing.assert_array_equal(first['waveform'], second['waveform'])
        assert first['text'] == second['text']


def test_cluster_cli_with_real_index_and_spawn_workers(corpus):
    import subprocess
    import sys

    from test_qasr_processor import toy_processor

    root, index_config, _ = corpus
    processor = toy_processor()
    model_path = root.parent / 'processor'
    processor.save_pretrained(model_path)
    settings = yaml.safe_load(index_config.read_text())
    settings.update(tokenizer_path=str(model_path), sample_rate=16000)
    index_config.write_text(yaml.safe_dump(settings))
    plan(root, index_config, jobs=1, samples=28)
    run(root, procs=2)
    assemble(root, jobs=1)
    config = yaml.safe_load(Path('configs/v8.1/02_full_10node.yaml').read_text())
    config.update(model_name_or_path=str(model_path),
                  code_switch_index=str(root / 'assembled/index.json'),
                  train_manifest={lang: [str(root.parent / 'source' / lang / 'data' / f'train_{lang}.jsonl')]
                                  for lang in LANGS})
    noise = root.parent / 'musan'
    noise.mkdir()
    sf.write(noise / 'noise.wav', np.random.default_rng(1).normal(0, .05, 16000), 16000)
    config['augmentation']['noise_injection']['noise_dir'] = str(noise)
    training_config = root.parent / 'train.yaml'
    training_config.write_text(yaml.safe_dump(config))
    output = root.parent / 'diagnostic'
    result = subprocess.run([sys.executable, 'scripts/check_v81_dataloader.py',
        '--config', str(training_config), '--allow-pilot', '--batches', '3', '--workers', '2',
        '--output', str(output)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / 'rank-0.json').read_text())
    assert report['counts']['examples'] == 60
    assert report['counts']['applied'] > 0
    assert report['counts']['substitutions'] == 0
    assert report['counts']['anchor:ar'] == 30
    assert report['counts']['label:ar'] == 30


def test_verification_receipt_reuse_and_metadata_mode(corpus, monkeypatch):
    import data_processing.code_switch.index as indexing
    root, config, _ = corpus
    frozen = plan(root, config, jobs=1, samples=28)
    run(root, procs=2)
    # Metadata mode must work even when the original audio is offline.
    audio = root.parent / 'ar-1.wav'
    hidden = audio.with_suffix('.offline')
    audio.rename(hidden)
    assert not verify(root, jobs=2, metadata_only=True)
    receipt = read_json(root / '_state/verified.json')
    assert receipt['mode'] == 'metadata'
    # A failed header audit must invalidate that prior receipt.
    assert verify(root, jobs=1)
    assert not (root / '_state/verified.json').exists()
    hidden.rename(audio)
    assert not verify(root, jobs=2, metadata_only=True)
    # Reuse is conditional on all input signatures, not simply receipt existence.
    before = indexing._verification_snapshot(root, frozen)
    part = chunk_paths(root, frozen['chunks'][0])[0]
    import os
    st = part.stat()
    original = part.read_bytes()
    part.write_bytes(original)
    os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert indexing._verification_snapshot(root, frozen) != before  # ctime catches it
    assert not verify(root, jobs=2, metadata_only=True)
    def no_repeat(*args, **kwargs):
        raise AssertionError('assemble repeated verification despite unchanged receipt')
    monkeypatch.setattr(indexing, 'verify_unlocked', no_repeat)
    assemble(root, jobs=2)


def test_stale_receipt_cannot_hide_corrupted_chunk(corpus):
    import os
    root, config, _ = corpus
    frozen = plan(root, config, jobs=1, samples=28)
    run(root, procs=2)
    assert not verify(root, jobs=2, metadata_only=True)
    part = chunk_paths(root, frozen['chunks'][0])[0]
    st = part.stat()
    raw = part.read_bytes()
    # Change the committed bytes but preserve both size and mtime.
    part.write_bytes(raw.replace(b'"text":', b'"texx":', 1))
    os.utime(part, ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(ValueError, match='incomplete/bad'):
        assemble(root, jobs=2)
    assert not (root / 'assembled').exists()


def test_cross_language_conflict_still_rejected_after_cached_verify(corpus):
    root, config, _ = corpus
    english = root.parent / 'source/en/data/train_en.jsonl'
    rows = [json.loads(line) for line in english.read_text().splitlines()]
    rows[1]['audio_filepath'] = str(root.parent / 'ar-1.wav')
    english.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan(root, config, jobs=1, samples=28)
    run(root, procs=2)
    assert not verify(root, jobs=2, metadata_only=True)
    with pytest.raises(ValueError, match='Conflicting language pools'):
        assemble(root, jobs=2)
    assert not (root / 'assembled').exists()
