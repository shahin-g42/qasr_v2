"""Additional waveform effects; optional distortions are disabled in shipped recipes."""

import math
import random
import subprocess

import numpy as np

SR = 16000
SETTINGS = {
    "silence": {"p", "start_ms", "end_ms"},
    "pitch": {"p", "semitones"},
    "clipping": {"p", "peak_fraction"},
    "dropout": {"p", "duration_ms", "max_drops", "max_fraction"},
    "compression": {"p", "threshold_db", "ratio", "attack_ms", "release_ms"},
}
BOUNDS = {
    "silence": {"start_ms": ([0, 100], 0, 1000), "end_ms": ([0, 100], 0, 1000)},
    "pitch": {"semitones": ([-1, 1], -4, 4)},
    "clipping": {"peak_fraction": ([0.7, 0.95], 0.1, 1)},
    "dropout": {"duration_ms": ([5, 20], 1, 100)},
    "compression": {"threshold_db": ([-24, -12], -60, 0), "ratio": ([2, 4], 1, 20)},
}


def validate(name, cfg):
    for key, (default, lower, upper) in BOUNDS[name].items():
        values = cfg.get(key, default)
        if (
            not isinstance(values, (list, tuple)) or len(values) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not lower <= v <= upper for v in values)
            or values[0] > values[1]
        ):
            raise ValueError(f"{name}.{key} requires ordered bounds within [{lower}, {upper}]")
    scalars = {
        "dropout": {"max_fraction": (0.01, 0, 0.05)},
        "compression": {"attack_ms": (5, 0.01, 2000), "release_ms": (50, 0.01, 9000)},
    }
    for key, (default, lower, upper) in scalars.get(name, {}).items():
        value = cfg.get(key, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not lower <= value <= upper):
            raise ValueError(f"Invalid {name}.{key}")
    if name == "dropout":
        count = cfg.get("max_drops", 2)
        if type(count) is not int or not 1 <= count <= 10:
            raise ValueError("dropout.max_drops must be an integer in [1, 10]")


def draw(name, cfg, key):
    return random.uniform(*cfg.get(key, BOUNDS[name][key][0]))


def apply(name, wave, cfg, max_audio_seconds=None):
    if name == "silence":
        start = round(draw(name, cfg, "start_ms") * SR / 1000)
        end = round(draw(name, cfg, "end_ms") * SR / 1000)
        if max_audio_seconds is not None:
            budget = max(0, math.floor(max_audio_seconds * SR) - len(wave))
            if start + end > budget:
                start = math.floor(start * budget / max(start + end, 1))
                end = budget - start
        # All original samples survive. Only optional added silence is shortened.
        return np.pad(wave, (start, end))
    if name == "pitch":
        import librosa

        steps = draw(name, cfg, "semitones")
        if steps == 0:
            return wave
        return librosa.effects.pitch_shift(wave, sr=SR, n_steps=steps)
    if name == "clipping":
        threshold = float(np.abs(wave).max()) * draw(name, cfg, "peak_fraction")
        return np.clip(wave, -threshold, threshold)
    if name == "dropout":
        result = wave.copy()
        remaining = math.floor(len(wave) * cfg.get("max_fraction", 0.01))
        for _ in range(random.randint(1, cfg.get("max_drops", 2))):
            count = min(round(draw(name, cfg, "duration_ms") * SR / 1000), remaining)
            if count <= 0:
                break
            start = random.randint(0, len(wave) - count)
            result[start : start + count] = 0
            remaining -= count
        return result
    if name == "compression":
        threshold = 10 ** (draw(name, cfg, "threshold_db") / 20)
        ratio = draw(name, cfg, "ratio")
        spec = (
            f"acompressor=threshold={threshold}:ratio={ratio}:"
            f"attack={cfg.get('attack_ms', 5)}:release={cfg.get('release_ms', 50)}:"
            "makeup=1:detection=rms"
        )
        command = [
            "ffmpeg", "-nostdin", "-v", "error", "-f", "f32le", "-ar", str(SR),
            "-ac", "1", "-i", "pipe:0", "-filter_threads", "1", "-af", spec,
            "-threads", "1", "-f", "f32le", "pipe:1",
        ]
        output = subprocess.run(
            command, input=wave.astype("<f4").tobytes(), capture_output=True,
            timeout=30, check=True,
        ).stdout
        result = np.frombuffer(output, dtype="<f4").copy()
        if len(result) != len(wave):
            raise ValueError("Compression unexpectedly changed audio duration")
        return result
    raise ValueError(f"Unknown additional effect: {name}")
