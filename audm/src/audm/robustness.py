"""Opt-in, bounded waveform augmentation and explicit empty-target replacement."""

import json
import logging
import math
import random
import shutil
import subprocess
from collections import Counter, deque
from pathlib import Path

import numpy as np
from qasr.audio import load_mono_audio, validate_audio_duration
from qasr.augmentation import build_augmenter
from qasr.data import JsonlSpeechDataset
from scipy import signal

LOG = logging.getLogger("audm.robustness")
SR = 16000


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def _waveform(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    return wave


def _rir(path):
    rir = load_mono_audio(path, SR)
    if np.max(np.abs(rir)) == 0:
        raise ValueError(f"Invalid/zero RIR: {path}")
    rir = rir[np.argmax(np.abs(rir)) :]
    return rir / max(np.linalg.norm(rir.astype(np.float64)), 1e-8)


def validate_negative_record(row, require_audio_type=False):
    if not isinstance(row, dict):
        raise ValueError("Non-speech records require a JSON object")
    kind = row.get("audio_type")
    if (require_audio_type and kind is None) or (
        kind is not None and kind not in ("music", "noise", "silence")
    ):
        raise ValueError("Non-speech records require audio_type music, noise or silence")
    if not any(k in row for k in ("text", "transcript")):
        raise ValueError("Non-speech records require an empty text field")
    if any(
        not isinstance(row[k], str) or row[k].strip() for k in ("text", "transcript") if k in row
    ):
        raise ValueError("Non-speech manifests require explicitly empty text strings")
    return kind


class RobustAugmenter:
    """Worker-local policy. Fail on broken assets/codecs, never silently fall back."""

    has_waveform_augmentation = True
    supports_audio_types = True

    def __init__(self, policy, legacy):
        if not isinstance(policy, dict) or set(policy) - {
            "clean_probability",
            "max_effects",
            "effects",
        }:
            raise ValueError("Unknown augmentation policy setting")
        self.policy = policy
        self.base = build_augmenter(legacy, SR)
        self.clean_probability = _number(
            policy.get("clean_probability", 0.2), "clean_probability", 0, 1
        )
        self.max_effects = policy.get("max_effects", 2)
        if type(self.max_effects) is not int or self.max_effects < 1:
            raise ValueError("max_effects must be a positive integer")
        self.effects = policy.get("effects", {})
        supported = {
            "gain": {"p", "db"},
            "shift": {"p", "max_ms", "boundary_peak"},
            "rir": {"p", "manifest"},
            "time_stretch": {"p", "rate"},
            "white_noise": {"p", "db"},
            "transcode": {"p", "codecs"},
            "speed": {"p"},
            "noise": {"p"},
            "codec_approx": {"p"},
        }
        if not isinstance(self.effects, dict) or set(self.effects) - set(supported):
            raise ValueError("Unknown effects or invalid effect mapping")
        self.stats = Counter()
        self.clean_rows = deque()
        self.rirs = []
        for name, settings in self.effects.items():
            if not isinstance(settings, dict) or set(settings) - supported[name]:
                raise ValueError(
                    f"Unknown/invalid settings for {name}; allowed: {sorted(supported[name])}"
                )
            _number(settings.get("p", 0), f"{name}.p", 0, 1)
            for key in ("db", "rate"):
                if key in settings:
                    bounds = settings[key]
                    if (
                        not isinstance(bounds, (list, tuple))
                        or len(bounds) != 2
                        or any(
                            isinstance(v, bool) or not isinstance(v, (int, float)) for v in bounds
                        )
                        or not np.isfinite(bounds).all()
                        or bounds[0] > bounds[1]
                        or (key == "rate" and bounds[0] <= 0)
                    ):
                        raise ValueError(f"Invalid {key} bounds for {name}")
            if name == "shift":
                _number(settings.get("max_ms", 5), "shift.max_ms", 0)
                _number(settings.get("boundary_peak", 0.001), "shift.boundary_peak", 0)
            if name == "transcode":
                codecs = settings.get("codecs", ["opus"])
                if (
                    not isinstance(codecs, list)
                    or not codecs
                    or any(not isinstance(c, str) for c in codecs)
                ):
                    raise ValueError("transcode.codecs must be a nonempty list")
                for codec in codecs:
                    self.codec(codec)
            if not settings.get("p", 0):
                continue
            if name in ("speed", "noise", "codec_approx"):
                attr = {
                    "speed": "speed_perturb",
                    "noise": "noise_injection",
                    "codec_approx": "codec_augment",
                }[name]
                obj = getattr(self.base, attr, None)
                if obj is None:
                    raise ValueError(f"Enable legacy {attr} before selecting {name}")
                obj.p = 1.0
                if name == "noise" and not obj._noise_files:
                    raise ValueError(
                        "Real-noise policy requires a nonempty readable noise directory"
                    )
            if name == "rir":
                if (
                    not isinstance(settings.get("manifest"), str)
                    or not settings["manifest"].strip()
                ):
                    raise ValueError("Enabled RIR augmentation requires a manifest path")
                manifest = Path(settings["manifest"]).resolve()
                for line in manifest.read_text().splitlines():
                    if line.strip():
                        p = Path(json.loads(line)["audio_filepath"])
                        p = p if p.is_absolute() else manifest.parent / p
                        if not p.is_file():
                            raise FileNotFoundError(p)
                        self.rirs.append(p)
                if not self.rirs:
                    raise ValueError("RIR manifest is empty")
            if name == "transcode":
                if not shutil.which("ffmpeg"):
                    raise RuntimeError("Transcoding requires ffmpeg on PATH")
                encoders = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                ).stdout
                available = {
                    line.split()[1] for line in encoders.splitlines() if len(line.split()) >= 2
                }
                for codec in settings.get("codecs", ["opus"]):
                    encoder, _, _ = self.codec(codec)
                    if encoder not in available:
                        raise RuntimeError(f"FFmpeg encoder unavailable: {encoder}")
        self.has_spec_augmentation = bool(self.base and self.base.has_spec_augmentation)

    def scan_assets(self):
        """Decode selected assets and exercise every configured codec, before training."""
        for path in self.rirs:
            _rir(path)
        noise_files = []
        if self.effects.get("noise", {}).get("p", 0):
            noise_files = sorted(
                {p for paths in self.base.noise_injection._noise_files.values() for p in paths}
            )
            for path in noise_files:
                wave = load_mono_audio(path, SR)
                if not np.any(wave):
                    raise ValueError(f"Real-noise asset is all-zero: {path}")
        codecs = []
        if self.effects.get("transcode", {}).get("p", 0):
            wave = (0.05 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
            for codec in self.effects["transcode"].get("codecs", ["opus"]):
                self.transcode(wave, {"codecs": [codec]})
                codecs.append(codec)
        return dict(rirs=len(self.rirs), noise_files=len(noise_files), codec_round_trips=codecs)

    @staticmethod
    def codec(name):
        codecs = {
            "opus": ("libopus", "ogg", "24k"),
            "aac": ("aac", "adts", "32k"),
            "mp3": ("libmp3lame", "mp3", "32k"),
            "g711": ("pcm_mulaw", "wav", None),
            "amr_nb": ("libopencore_amrnb", "amr", "7.95k"),
        }
        if name not in codecs:
            raise ValueError(f"Unsupported codec: {name}")
        return codecs[name]

    def transcode(self, wave, cfg):
        name = random.choice(cfg.get("codecs", ["opus"]))
        encoder, fmt, rate = self.codec(name)
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(SR),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-c:a",
            encoder,
            "-threads",
            "1",
        ]
        if name in ("g711", "amr_nb"):
            cmd += ["-ar", "8000"]
        if rate:
            cmd += ["-b:a", rate]
        cmd += ["-f", fmt, "pipe:1"]
        encoded = subprocess.run(
            cmd, input=wave.astype("<f4").tobytes(), capture_output=True, timeout=30, check=True
        ).stdout
        decoded = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-f",
                "f32le",
                "-ar",
                str(SR),
                "-ac",
                "1",
                "-threads",
                "1",
                "pipe:1",
            ],
            input=encoded,
            capture_output=True,
            timeout=30,
            check=True,
        ).stdout
        result = _waveform(np.frombuffer(decoded, dtype="<f4").copy())
        # Preserve decoder output, including codec delay/tail; never truncate speech.
        self.stats[f"codec:{name}"] += 1
        return result

    def apply(self, name, wave, cfg):
        if name == "gain":
            return wave * 10 ** (random.uniform(*cfg.get("db", [-6, 6])) / 20)
        if name == "white_noise":
            level = random.uniform(*cfg.get("db", [-70, -46]))
            return wave + np.random.normal(0, 10 ** (level / 20), len(wave))
        if name == "shift":
            n = round(random.uniform(-cfg.get("max_ms", 5), cfg.get("max_ms", 5)) * SR / 1000)
            if n == 0 or abs(n) >= len(wave):
                return wave
            lost = wave[-n:] if n > 0 else wave[:-n]
            # Require near-silent boundaries; do not crop phonemes or wrap samples.
            if np.max(np.abs(lost)) > cfg.get("boundary_peak", 0.001):
                self.stats["shift:boundary_skip"] += 1
                return wave
            return (
                np.concatenate((np.zeros(n), wave[:-n]))
                if n > 0
                else np.concatenate((wave[-n:], np.zeros(-n)))
            )
        if name == "rir":
            rir = _rir(random.choice(self.rirs))
            result = signal.fftconvolve(wave, rir)
            # Keep full tail, but preserve RMS to separate room and gain effects.
            rms = np.sqrt(np.mean(result**2))
            return result * np.sqrt(np.mean(wave**2)) / max(rms, 1e-8)
        if name == "time_stretch":
            import librosa

            return librosa.effects.time_stretch(
                wave, rate=random.uniform(*cfg.get("rate", [0.9, 1.1]))
            )
        if name == "transcode":
            return self.transcode(wave, cfg)
        obj = getattr(
            self.base,
            {"speed": "speed_perturb", "noise": "noise_injection", "codec_approx": "codec_augment"}[
                name
            ],
        )
        result = obj(wave, SR)
        if name == "noise" and getattr(obj, "last_noise_source", None) == "gaussian":
            raise RuntimeError("Configured real-noise file failed; refusing Gaussian fallback")
        return result

    def augment_waveform(
        self, wave, min_audio_seconds=None, max_audio_seconds=None, audio_type=None
    ):
        wave = _waveform(wave)
        if audio_type not in (None, "speech", "music", "noise", "silence", "non_speech"):
            raise ValueError(f"Invalid augmentation audio_type: {audio_type}")
        if min_audio_seconds is not None:
            _number(min_audio_seconds, "min_audio_seconds", 0)
        if max_audio_seconds is not None:
            _number(max_audio_seconds, "max_audio_seconds", min_audio_seconds or 0)
        self.stats["samples"] += 1
        # Silence must not become labeled silence with added audible noise/music.
        clean = audio_type == "silence" or random.random() < self.clean_probability
        if clean:
            if self.has_spec_augmentation:
                self.clean_rows.append(True)
            self.stats["clean"] += 1
            return wave
        choices = [
            k
            for k, v in self.effects.items()
            if random.random() < v.get("p", 0)
            and not (audio_type in ("music", "noise", "non_speech") and k == "noise")
            and not (audio_type == "music" and k == "white_noise")
        ]
        # The configured real-noise corpus may contain speech/music. Do not mix it
        # into typed negatives and silently invalidate their category labels.
        random.shuffle(choices)
        chosen = []
        for name in choices:
            if name in ("speed", "time_stretch") and any(
                n in chosen for n in ("speed", "time_stretch")
            ):
                continue
            if name in ("transcode", "codec_approx") and any(
                n in chosen for n in ("transcode", "codec_approx")
            ):
                continue
            chosen.append(name)
            if len(chosen) == self.max_effects:
                break
        order = [
            "speed",
            "time_stretch",
            "shift",
            "rir",
            "noise",
            "gain",
            "white_noise",
            "codec_approx",
            "transcode",
        ]
        for name in sorted(chosen, key=order.index):
            self.stats[f"{name}:attempted"] += 1
            result = np.asarray(self.apply(name, wave, self.effects[name]), dtype=np.float32)
            if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
                raise ValueError(f"{name} produced invalid waveform")
            duration = len(result) / SR
            if (min_audio_seconds is not None and duration < min_audio_seconds) or (
                max_audio_seconds is not None and duration > max_audio_seconds
            ):
                self.stats[f"{name}:duration_skip"] += 1
                continue
            changed = result.shape != wave.shape or not np.array_equal(result, wave)
            self.stats[f"{name}:changed"] += int(changed)
            wave = result
        if self.has_spec_augmentation:
            self.clean_rows.append(False)
        if self.stats["samples"] % 1000 == 0:
            LOG.info("Worker-local augmentation counts: %s", dict(self.stats))
        return wave

    def augment_features(self, features, mask):
        if not self.has_spec_augmentation:
            return features
        if len(self.clean_rows) < len(features):
            raise ValueError("SpecAugment requires one waveform policy decision per example")
        flags = [self.clean_rows.popleft() for _ in range(len(features))]
        rows = []
        for i, clean in enumerate(flags):
            row = features[i : i + 1]
            if not clean:
                row = self.base.augment_features(row, mask[i : i + 1])
            rows.append(row)
        import torch

        return torch.cat(rows)


class NonSpeechPool:
    """Replace a controlled fraction of training slots; retain each slot's language tag."""

    def __init__(self, cfg, minimum, maximum, require_audio_type=False):
        if not isinstance(cfg, dict) or set(cfg) - {"probability", "manifests"}:
            raise ValueError("Unknown non-speech pool setting")
        self.probability = _number(cfg.get("probability", 0), "non_speech.probability", 0, 1)
        if not 0 < self.probability < 1:
            raise ValueError("Non-speech probability must be between 0 and 1")
        manifests = cfg.get("manifests", [])
        if (
            not isinstance(manifests, list)
            or not manifests
            or any(not isinstance(p, str) for p in manifests)
        ):
            raise ValueError("non_speech.manifests must be a nonempty list of paths")
        if len({str(Path(p).resolve()) for p in manifests}) != len(manifests):
            raise ValueError("Duplicate non-speech manifest")
        self.minimum, self.maximum = minimum, maximum
        self.categories = []
        for path in manifests:
            categories = {}
            self.categories.append(categories)
            for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
                if line.strip():
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number}: expected a JSON object")
                    kind = validate_negative_record(row, require_audio_type)
                    categories[line_number] = kind
        self.parts = [
            JsonlSpeechDataset(
                p, min_duration_seconds=minimum, max_duration_seconds=maximum, empty_target_ok=True
            )
            for p in cfg.get("manifests", [])
        ]
        if not self.parts or not sum(map(len, self.parts)):
            raise ValueError("Non-speech manifests contain no eligible records")
        self.ends = np.cumsum([len(p) for p in self.parts])
        self.stats = Counter()
        # Never turn accidentally supplied speech transcripts into empty labels.
        for part in self.parts:
            for i in range(len(part)):
                if part[i]["text"].strip():
                    raise ValueError("Non-speech manifests must contain only empty transcripts")

    def record(self, part, index):
        row = self.parts[part][index]
        return dict(row, audio_type=self.categories[part][row["line_number"]])

    def scan_audio(self):
        """Decode eligible negatives before launch, with no silent dropping/replacement."""
        counts, seconds = Counter(), 0.0
        for part_number, part in enumerate(self.parts):
            for index in range(len(part)):
                row = self.record(part_number, index)
                wave = load_mono_audio(row["audio_path"], SR)
                validate_audio_duration(
                    wave,
                    path=row["audio_path"],
                    sampling_rate=SR,
                    min_audio_seconds=self.minimum,
                    max_audio_seconds=self.maximum,
                )
                counts[row["audio_type"] or "untyped"] += 1
                seconds += len(wave) / SR
        return dict(
            records=sum(counts.values()),
            categories=dict(counts),
            hours=seconds / 3600,
            manifest_records=sum(p.stats.total_records for p in self.parts),
            skipped_duration=sum(p.stats.skipped_by_duration for p in self.parts),
        )

    def replace(self, records):
        output = []
        for record in records:
            self.stats["slots"] += 1
            if random.random() < self.probability:
                index = random.randrange(int(self.ends[-1]))
                part = int(np.searchsorted(self.ends, index, side="right"))
                row = self.record(part, index - (int(self.ends[part - 1]) if part else 0))
                output.append(dict(row, language=record.get("language"), text=""))
                self.stats["replaced"] += 1
            else:
                output.append(record)
        if self.stats["slots"] // 1000 != (self.stats["slots"] - len(records)) // 1000:
            LOG.info("Worker-local non-speech sampling: %s", dict(self.stats))
        return output
