"""Online, anchor-preserving augmentation backed by an immutable short-clip index.

Training uses augment() with a per-example Bernoulli gate. compose()/parts()
provide bounded, deterministic listening previews only, never training entries.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import sqlite3
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audio import AudioLoadingError, load_mono_audio

LOGGER = logging.getLogger('qasr')


def tokenizer_fingerprint(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    # Runtime padding/truncation settings do not define the vocabulary/normalizer.
    backend.pop('padding', None)
    backend.pop('truncation', None)
    return hashlib.sha256(json.dumps(backend, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def diagonal_pair(position, na, nb):
    """Bijection over a Cartesian product, balanced prefixes in both pools."""
    period = math.lcm(na, nb)
    if not 0 <= position < na * nb:
        raise ValueError('Unique candidate space exhausted')
    return position % na, (position + position // period) % nb


class OnlineCodeSwitch:
    def __init__(self, index_path, tokenizer, *, allow_pilot=False, retokenize=False):
        self.index_path = str(Path(index_path).resolve())
        self.index = json.loads(Path(index_path).read_text())
        if self.index['version'] != 2:
            raise ValueError('Unsupported code-switch index version')
        if self.index['pilot'] and not allow_pilot:
            raise ValueError('Pilot index cannot be used for a production training run')
        plan_path = Path(self.index['root']) / '_state/plan.json'
        frozen = json.loads(plan_path.read_text())
        if frozen.get('version') != 2 or frozen['root'] != str(Path(self.index['root']).resolve()):
            raise ValueError('Invalid or moved frozen code-switch plan')
        def digest(path):
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if (digest(Path(self.index['root']) / '_state/plan.json') != self.index['plan_sha256']
                or self.index['config'] != frozen['config']
                or self.index['preview_samples'] != frozen['preview_samples']
                or sum(self.index['pair_targets'].values()) != frozen['preview_samples']):
            raise ValueError('Published code-switch index differs from its frozen plan')
        self.cfg = dict(self.index['config'])
        self.tokenizer = tokenizer
        if not retokenize and tokenizer_fingerprint(tokenizer) != self.index['tokenizer_fingerprint']:
            raise ValueError('Code-switch index tokenizer differs from training tokenizer')
        for lang, spec in self.index['pools'].items():
            path = Path(index_path).parent / f'{lang}.sqlite'
            if path.stat().st_size != spec['size']:
                raise ValueError(f'Code-switch pool changed: {path}')
        self._connections = {}
        self._pid = os.getpid()
        self._permutations = {}
        self._counts = Counter()

    def __getstate__(self):
        return {**self.__dict__, '_connections': {}, '_pid': None}

    def _pool(self, lang):
        if self._pid != os.getpid():
            self._connections = {}
            self._pid = os.getpid()
        if lang not in self._connections:
            path = Path(self.index_path).parent / f'{lang}.sqlite'
            self._connections[lang] = sqlite3.connect(path.as_uri() + '?mode=ro&immutable=1', uri=True)
        return self._connections[lang]

    def _permuted(self, pair, lang, i):
        key = (pair, lang)
        n = self.index['pools'][lang]['count']
        if key not in self._permutations:
            seed = int(hashlib.sha256(f'{self.cfg["seed"]}:{pair}:{lang}'.encode()).hexdigest(), 16)
            rng = random.Random(seed)
            a = rng.randrange(1, n)
            while math.gcd(a, n) != 1:
                a = (a + 1) % n or 1
            self._permutations[key] = a, rng.randrange(n)
        a, b = self._permutations[key]
        return (a * i + b) % n

    def _record(self, pair, lang, i):
        row = self._pool(lang).execute('SELECT record FROM pool WHERE id=?',
                                      (self._permuted(pair, lang, i),)).fetchone()
        if row is None:
            raise ValueError('Index has a missing pool row; rebuild/verify the index')
        return json.loads(row[0])

    def compose(self, pair, slot, first_language):
        a, b = pair.split('-')
        na, nb = (self.index['pools'][lang]['count'] for lang in (a, b))
        quota = self.index['pair_targets'][pair]
        cfg = self.cfg
        reasons = []
        for attempt in range(cfg['max_attempts']):
            candidate = slot + attempt * quota
            ia, ib = diagonal_pair(candidate, na, nb)
            # The first occurrence of each language identifies this candidate.
            order = [(a, ia), (b, ib), (a, (ia + 1) % na), (b, (ib + 1) % nb)]
            if first_language == b:
                order = [order[1], order[0], order[3], order[2]]
            count = cfg['segment_counts'][slot % len(cfg['segment_counts'])]
            records = [self._record(pair, lang, i) for lang, i in order[:count]]
            waveforms = []
            try:
                for record in records:
                    waveform = load_mono_audio(record['audio_filepath'], cfg['sample_rate'])
                    if (len(waveform) != record['frames']
                            or hashlib.sha256(waveform.tobytes()).hexdigest() != record['waveform_sha256']):
                        raise ValueError('Source audio content or duration changed after indexing')
                    waveforms.append(waveform)
            except (AudioLoadingError, ValueError) as exc:
                raise RuntimeError('Indexed source audio changed or became unavailable; restore it or rebuild the index') from exc
            gap = np.zeros(round(cfg['gap_seconds'] * cfg['sample_rate']), dtype=np.float32)
            # Remove whole optional segments to fit; never cut words/audio.
            while len(records) >= 2:
                text = ' '.join(record['text'] for record in records)
                ids = self.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                frames = sum(len(w) for w in waveforms) + len(gap) * (len(records) - 1)
                if len(ids) <= cfg['max_target_tokens'] and frames <= math.floor(cfg['max_seconds'] * cfg['sample_rate']):
                    break
                records.pop()
                waveforms.pop()
            if len(records) < 2:
                reasons.append('duration/token budget')
                continue
            language_frames = Counter()
            segments = []
            offset = 0
            for record, waveform in zip(records, waveforms, strict=True):
                language_frames[record['language']] += len(waveform)
                segments.append({**record, 'start_frame': offset, 'end_frame': offset + len(waveform)})
                offset += len(waveform) + len(gap)
            language = first_language
            if cfg['language_policy'] == 'major':
                # Counter insertion order follows segment order, so ties use first.
                language = max(language_frames, key=language_frames.get)
            joined = []
            for i, waveform in enumerate(waveforms):
                if i:
                    joined.append(gap)
                joined.append(waveform)
            self._counts[pair] += 1
            total = sum(self._counts.values())
            if total == 1 or total % 1000 == 0:
                LOGGER.info('Online code-switch worker pid=%d: %s', os.getpid(), dict(self._counts))
            return {'audio_path': records[0]['audio_filepath'], 'text': text, 'language': language,
                    'duration': frames / cfg['sample_rate'], 'waveform': np.concatenate(joined),
                    'transcript_ids': ids, 'line_number': slot + 1, 'code_switch': True,
                    'code_switch_id': f'{pair}:{slot}', 'candidate': candidate, 'pair': pair,
                    'segments': segments, 'anchor_language': first_language}
        raise RuntimeError(f'Cannot compose {pair}:{slot} after {cfg["max_attempts"]} attempts: {reasons}')

    def augment(self, feature, p=0.4, *, rng=None):
        result, audit = self._augment(feature, p, rng=rng)
        self._counts['seen'] += 1
        self._counts['triggered'] += int(audit['triggered'])
        self._counts['applied'] += int(audit['applied'])
        self._counts[audit['reason']] += 1
        if self._counts['seen'] == 1 or self._counts['seen'] % 1000 == 0:
            LOGGER.info('Code-switch augmentation pid=%d: %s', os.getpid(), dict(self._counts))
        return result, audit

    def _augment(self, feature, p=0.4, *, rng=None):
        """Bernoulli gate per example; retain the original anchor and whole clips.

        An attempted mix may fall back unchanged when duration/token limits leave
        no room. Index corruption and missing audio are errors, never fallbacks.
        """
        if not 0 <= p <= 1:
            raise ValueError('Code-switch probability must be in [0, 1]')
        rng = rng or random
        if rng.random() >= p:
            return feature, {'triggered': False, 'applied': False, 'reason': 'gate'}
        audit = {'triggered': True, 'applied': False}
        language = feature.get('language')
        choices = [(pair, weight) for pair, weight in self.cfg['pair_weights'].items()
                   if language in pair.split('-') and weight > 0]
        if not choices or not feature['text'].strip():
            return feature, {**audit, 'reason': 'ineligible_anchor'}
        waveform = feature.get('waveform')
        if waveform is None:
            waveform = load_mono_audio(feature['audio_path'], self.cfg['sample_rate'])
        sr = self.cfg['sample_rate']
        gap = np.zeros(round(self.cfg['gap_seconds'] * sr), dtype=np.float32)
        limit = math.floor(self.cfg['max_seconds'] * sr)
        if len(waveform) + len(gap) + round(self.cfg['min_source_seconds'] * sr) > limit:
            return feature, {**audit, 'reason': 'duration_budget'}
        pair = rng.choices([c[0] for c in choices], weights=[c[1] for c in choices])[0]
        partner = next(lang for lang in pair.split('-') if lang != language)
        waves = [waveform]
        segments = [{'text': feature['text'], 'language': language,
                     'audio_filepath': str(feature['audio_path']), 'frames': len(waveform)}]
        text = feature['text']
        ids = None
        used = {Path(feature['audio_path']).name}
        wanted = rng.choice(self.cfg['segment_counts'])
        for position in range(1, wanted):
            next_lang = partner if position % 2 else language
            accepted = False
            for _ in range(self.cfg['max_attempts']):
                record = self._record(pair, next_lang, rng.randrange(self.index['pools'][next_lang]['count']))
                if (Path(record['audio_filepath']).name in used
                        or Path(record['audio_filepath']).name in getattr(self, '_excluded_audio_names', ())):
                    continue
                frames = sum(map(len, waves)) + record['frames'] + len(gap) * len(waves)
                candidate_text = text + ' ' + record['text']
                candidate_ids = self.tokenizer(candidate_text, add_special_tokens=False,
                                               truncation=False)['input_ids']
                if frames > limit or len(candidate_ids) > self.cfg['max_target_tokens']:
                    continue
                extra = load_mono_audio(record['audio_filepath'], sr)
                if (len(extra) != record['frames']
                        or hashlib.sha256(extra.tobytes()).hexdigest() != record['waveform_sha256']):
                    raise ValueError('Indexed source audio changed; restore it or rebuild the index')
                waves.append(extra)
                segments.append(record)
                used.add(Path(record['audio_filepath']).name)
                text, ids, accepted = candidate_text, candidate_ids, True
                break
            if not accepted:
                break
        if len(waves) == 1:
            return feature, {**audit, 'reason': 'no_fitting_partner'}
        joined = np.concatenate([item for i, wave in enumerate(waves)
                                 for item in ([gap, wave] if i else [wave])])
        assigned = language
        if self.cfg['language_policy'] == 'major':
            durations = Counter()
            for segment in segments:
                durations[segment['language']] += segment['frames']
            assigned = max(durations, key=durations.get)
        result = {**feature, 'waveform': joined, 'text': text, 'language': assigned,
                  'duration': len(joined) / sr, 'transcript_ids': ids,
                  'code_switch': True, 'pair': pair, 'segments': segments}
        return result, {**audit, 'applied': True, 'reason': 'mixed', 'pair': pair,
                        'segments': len(segments), 'seconds': len(joined) / sr}

    def parts(self):
        parts = []
        for pair, size in self.index['pair_targets'].items():
            a, b = pair.split('-')
            split = (size + 1) // 2
            if split:
                parts.append(CodeSwitchPart(self, pair, a, 0, split))
            if size - split:
                parts.append(CodeSwitchPart(self, pair, b, split, size - split))
        return parts


class CodeSwitchPart:
    def __init__(self, mixer, pair, language, start, size):
        self.mixer, self.pair, self.language, self.start, self.size = mixer, pair, language, start, size
        # Weighted stratified sampling uses counts. Hours are deliberately not
        # estimated: duration-based balanced sampling is disallowed for this mode.
        self.stats = SimpleNamespace(total_records=size, kept_records=size, total_kept_hours=0,
                                     skipped_by_duration=0, skipped_by_text=0, missing_duration=size)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index < 0:
            index += self.size
        if not 0 <= index < self.size:
            raise IndexError(index)
        return self.mixer.compose(self.pair, self.start + index, self.language)
