"""Opt-in, bounded waveform augmentation and explicit empty-target replacement."""

import json
import logging
import random
import shutil
import subprocess
from collections import Counter, deque
from pathlib import Path

import numpy as np
from qasr.audio import load_mono_audio
from qasr.augmentation import build_augmenter
from qasr.data import JsonlSpeechDataset
from scipy import signal

LOG = logging.getLogger("audm.robustness")
SR = 16000


class RobustAugmenter:
    """Worker-local policy. Fail on broken assets/codecs, never silently fall back."""

    has_waveform_augmentation = True

    def __init__(self, policy, legacy):
        if set(policy) - {"clean_probability", "max_effects", "effects"}:
            raise ValueError("Unknown augmentation policy setting")
        self.policy = policy
        self.base = build_augmenter(legacy, SR)
        self.clean_probability = float(policy.get("clean_probability", 0.2))
        self.max_effects = int(policy.get("max_effects", 2))
        if not 0 <= self.clean_probability <= 1 or self.max_effects < 1:
            raise ValueError("Invalid clean probability or effect budget")
        self.effects = policy.get("effects", {})
        supported = {
            "gain",
            "shift",
            "rir",
            "time_stretch",
            "white_noise",
            "transcode",
            "speed",
            "noise",
            "codec_approx",
        }
        if set(self.effects) - supported:
            raise ValueError(f"Unknown effects: {set(self.effects) - supported}")
        self.stats = Counter()
        self.clean_rows = deque()
        self.rirs = []
        for name, settings in self.effects.items():
            if not 0 <= settings.get("p", 0) <= 1:
                raise ValueError(f"Invalid probability for {name}")
            if not settings.get("p", 0):
                continue
            for key in ("db", "rate"):
                if key in settings:
                    bounds = settings[key]
                    if (
                        len(bounds) != 2
                        or not np.isfinite(bounds).all()
                        or bounds[0] > bounds[1]
                        or (key == "rate" and bounds[0] <= 0)
                    ):
                        raise ValueError(f"Invalid {key} bounds for {name}")
            if name == "shift" and (
                settings.get("max_ms", 5) < 0 or settings.get("boundary_peak", 0.001) < 0
            ):
                raise ValueError("Shift limits must be nonnegative")
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
                ).stdout
                for codec in settings.get("codecs", ["opus"]):
                    encoder, _, _ = self.codec(codec)
                    if encoder not in encoders:
                        raise RuntimeError(f"FFmpeg encoder unavailable: {encoder}")
        self.has_spec_augmentation = bool(self.base and self.base.has_spec_augmentation)

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
                "pipe:1",
            ],
            input=encoded,
            capture_output=True,
            timeout=30,
            check=True,
        ).stdout
        result = np.frombuffer(decoded, dtype="<f4").copy()
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
            rir = load_mono_audio(str(random.choice(self.rirs)), SR)
            if not np.isfinite(rir).all() or np.max(np.abs(rir)) == 0:
                raise ValueError("Invalid/zero RIR")
            rir = rir[np.argmax(np.abs(rir)) :]
            rir = rir / max(np.linalg.norm(rir), 1e-8)
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

    def augment_waveform(self, wave, min_audio_seconds=None, max_audio_seconds=None):
        self.stats["samples"] += 1
        clean = random.random() < self.clean_probability
        if self.has_spec_augmentation:
            self.clean_rows.append(clean)
        if clean:
            self.stats["clean"] += 1
            return wave
        choices = [k for k, v in self.effects.items() if random.random() < v.get("p", 0)]
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
        if self.stats["samples"] % 1000 == 0:
            LOG.info("Worker-local augmentation counts: %s", dict(self.stats))
        return wave

    def augment_features(self, features, mask):
        flags = [self.clean_rows.popleft() for _ in range(len(features))]
        if not self.has_spec_augmentation:
            return features
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

    def __init__(self, cfg, minimum, maximum):
        if set(cfg) - {"probability", "manifests"}:
            raise ValueError("Unknown non-speech pool setting")
        self.probability = cfg.get("probability", 0)
        if not 0 < self.probability < 1:
            raise ValueError("Non-speech probability must be between 0 and 1")
        for path in cfg.get("manifests", []):
            for line in Path(path).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    if any(str(row.get(k) or "").strip() for k in ("text", "transcript")):
                        raise ValueError("Non-speech manifests require explicitly empty text")
                    if "text" not in row and "transcript" not in row:
                        raise ValueError("Non-speech records require an empty text field")
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

    def replace(self, records):
        output = []
        for record in records:
            self.stats["slots"] += 1
            if random.random() < self.probability:
                index = random.randrange(int(self.ends[-1]))
                part = int(np.searchsorted(self.ends, index, side="right"))
                row = self.parts[part][index - (int(self.ends[part - 1]) if part else 0)]
                output.append(dict(row, language=record["language"], text=""))
                self.stats["replaced"] += 1
            else:
                output.append(record)
        if self.stats["slots"] // 1000 != (self.stats["slots"] - len(records)) // 1000:
            LOG.info("Worker-local non-speech sampling: %s", dict(self.stats))
        return output
