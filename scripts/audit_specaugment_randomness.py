"""Read-only audit of how well the SpecAugment randomization actually behaves.

Measures the production recipe shipped verbatim by every training YAML
(p=0.37, 2 time masks at <=5% of the valid frames, 2 freq masks at <=15% of
the 128 mel bins) and reports:

1. **Strength** — what share of a spectrogram is really masked, in relative and
   absolute terms, bucketed over the real duration distribution sampled from
   ``training_manifests/v7.0``.
2. **Granularity** — the ``p`` gate is drawn once per *batch*, not per sample:
   how much that inflates the step-to-step variance of the augmented share of
   the global batch.
3. **Rank correlation** — every rank calls ``set_seed(42)``, the DataLoader
   derives ``base_seed`` from the *global* torch RNG and each worker then does
   ``random.seed(base_seed + worker_id)``. Whether all 64 ranks therefore
   replay one identical mask stream, verified analytically (3a) and end to end
   with real DataLoader worker processes (3b).
4. **Coupling** — a single ``intensity`` draw scales both the time and the freq
   mask width, so the two axes are not independent.
5. **Waveform augmentations** — the same rank-shared stream also drives the
   speed / noise / codec draws, so those parameters are identical across ranks.
6. **Budget** — closed-form expectation of the masked share versus measured, to
   show which draws spend the nominal masking budget.

Usage::

    PYTHONPATH=src .venv-test/bin/python scripts/audit_specaugment_randomness.py
"""

from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from qasr.augmentation import SpecAugment

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_GLOB = "training_manifests/v7.0/*/train_*.jsonl"
PROBE_CONFIG = "configs/train_full_8node_filtered.yaml"

# The recipe shared verbatim by all seven training YAMLs.
PROD = {
    "num_time_masks": 2,
    "max_time_mask_ratio": 0.05,
    "num_freq_masks": 2,
    "max_freq_mask_ratio": 0.15,
    "p": 0.37,
}
FRAME_RATE = 100  # hop_length=160 @ 16 kHz
NUM_BINS = 128
WORLD_SIZE = 64
NUM_WORKERS = 8

# Fixed geometry for the cross-process DataLoader probe.
_PROBE_TIME = 400
_PROBE_BATCH = 4
_PROBE_STEPS = 40


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _spec() -> SpecAugment:
    return SpecAugment(**PROD)


def _read_durations(per_file: int = 4_000, total: int = 40_000) -> list[float]:
    """Sample the real training duration distribution from the v7.0 manifests.

    Every corpus contributes at most ``per_file`` records so the profile is not
    dominated by whichever manifest sorts first. Rejected piles are skipped.
    """
    durations: list[float] = []
    for path in sorted(REPO_ROOT.glob(MANIFEST_GLOB)):
        if "still_rejected" in path.name:
            continue
        taken = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if taken >= per_file or len(durations) >= total:
                    break
                try:
                    value = json.loads(line).get("duration")
                except json.JSONDecodeError:
                    continue
                if isinstance(value, (int, float)) and value > 0:
                    durations.append(float(value))
                    taken += 1
        if len(durations) >= total:
            break
    return durations


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _count_runs(flags: list[bool]) -> int:
    runs = 0
    previous = False
    for flag in flags:
        if flag and not previous:
            runs += 1
        previous = flag
    return runs


def _signature(features: torch.Tensor, attention_mask: torch.Tensor) -> tuple[str, bool]:
    """Hash the masked layout so two runs can be compared byte for byte.

    Feeding an all-ones spectrogram makes every masked cell exactly 0.0, so the
    mask geometry can be recovered from the output alone.
    """
    augmented = _spec()(features, attention_mask)
    applied = not torch.equal(augmented, features)
    return hashlib.sha1(augmented.numpy().tobytes()).hexdigest()[:16], applied


# ---------------------------------------------------------------------------
# 1. strength profile
# ---------------------------------------------------------------------------


def strength_profile(durations: list[float], n_samples: int = 40_000) -> tuple[float, float]:
    rng = random.Random(7)
    time_frac: list[float] = []
    freq_frac: list[float] = []
    cell_frac: list[float] = []
    masked_seconds: list[float] = []
    n_time_runs: list[int] = []
    n_freq_runs: list[int] = []
    pairs: list[tuple[float, float]] = []
    buckets: dict[str, list[float]] = {
        "<1s": [], "1-3s": [], "3-10s": [], "10-20s": [], ">20s": [],
    }

    for _ in range(n_samples):
        seconds = rng.choice(durations)
        valid = max(2, int(seconds * FRAME_RATE))
        features = torch.ones(1, valid, NUM_BINS)
        mask = torch.ones(1, valid)
        augmented = _spec()(features, mask)
        zeroed = augmented[0] == 0.0
        frame_masked = zeroed.all(dim=-1)  # a time mask zeroes every bin
        bin_masked = zeroed.all(dim=0)  # a freq mask zeroes every valid frame
        tf = float(frame_masked.sum()) / valid
        ff = float(bin_masked.sum()) / NUM_BINS
        time_frac.append(tf)
        freq_frac.append(ff)
        cell_frac.append(float(zeroed.sum()) / (valid * NUM_BINS))
        masked_seconds.append(tf * valid / FRAME_RATE)
        pairs.append((tf, ff))
        n_time_runs.append(_count_runs(frame_masked.tolist()))
        n_freq_runs.append(_count_runs(bin_masked.tolist()))
        key = (
            "<1s" if seconds < 1 else
            "1-3s" if seconds < 3 else
            "3-10s" if seconds < 10 else
            "10-20s" if seconds < 20 else ">20s"
        )
        buckets[key].append(tf)

    fired_seconds = [value for value in masked_seconds if value > 0]
    print("\n=== 1. Masking strength (single sample, p gate included) ===")
    print(f"samples drawn: {len(time_frac)}   real durations: {len(durations)}")
    print(
        f"time-frame share masked : mean={statistics.fmean(time_frac):.4f} "
        f"p50={_quantile(time_frac, 0.5):.4f} p95={_quantile(time_frac, 0.95):.4f} "
        f"max={max(time_frac):.4f}"
    )
    print(
        f"freq-bin share masked   : mean={statistics.fmean(freq_frac):.4f} "
        f"p50={_quantile(freq_frac, 0.5):.4f} p95={_quantile(freq_frac, 0.95):.4f} "
        f"max={max(freq_frac):.4f}"
    )
    print(
        f"spectrogram cells zeroed: mean={statistics.fmean(cell_frac):.4f} "
        f"p95={_quantile(cell_frac, 0.95):.4f} max={max(cell_frac):.4f}"
    )
    print(
        f"speech zeroed per FIRED sample (s): mean={statistics.fmean(fired_seconds):.2f} "
        f"p95={_quantile(fired_seconds, 0.95):.2f} p99={_quantile(fired_seconds, 0.99):.2f} "
        f"max={max(fired_seconds):.2f}"
    )
    no_time = sum(1 for runs in n_time_runs if runs == 0) / len(n_time_runs)
    no_freq = sum(1 for runs in n_freq_runs if runs == 0) / len(n_freq_runs)
    fired = sum(1 for t, f in pairs if t > 0 or f > 0) / len(pairs)
    print(f"P(no time mask at all)  = {no_time:.4f}")
    print(f"P(no freq mask at all)  = {no_freq:.4f}")
    print(f"P(batch gate fired)     = {fired:.4f}  (config p={PROD['p']:.2f})")
    print(
        f"time masks applied (of 2): mean={statistics.fmean(n_time_runs):.3f}   "
        f"freq masks applied: mean={statistics.fmean(n_freq_runs):.3f}"
    )

    print("\n--- time-mask share by utterance length ---")
    for key, values in buckets.items():
        if not values:
            print(f"{key:>7}: (no samples)")
            continue
        fired_values = [value for value in values if value > 0]
        mean_fired = statistics.fmean(fired_values) if fired_values else 0.0
        print(
            f"{key:>7}: n={len(values):6d}  mean={statistics.fmean(values):.4f}  "
            f"mean|fired={mean_fired:.4f}  "
            f"P(zero time mask)={1 - len(fired_values) / len(values):.3f}"
        )

    # Coupling between the time and the freq strength (shared `intensity` draw).
    both = [(t, f) for t, f in pairs if t > 0 and f > 0]
    if len(both) > 10:
        ts = [t for t, _ in both]
        fs = [f for _, f in both]
        mean_t, mean_f = statistics.fmean(ts), statistics.fmean(fs)
        cov = statistics.fmean((t - mean_t) * (f - mean_f) for t, f in both)
        corr = cov / (statistics.pstdev(ts) * statistics.pstdev(fs))
        print(f"\n=== 4. intensity coupling (masked samples only, n={len(both)}) ===")
        print(
            f"Pearson corr(time share, freq share) = {corr:.3f}  "
            "(1.0 = one draw drives both axes, 0.0 = independent)"
        )
    return statistics.fmean(time_frac), statistics.fmean(freq_frac)


# ---------------------------------------------------------------------------
# 2. gate granularity
# ---------------------------------------------------------------------------


def gate_granularity(steps: int = 4000) -> None:
    print("\n=== 2. Gate granularity: share of the GLOBAL batch masked per step ===")
    rng = random.Random(11)
    p = PROD["p"]
    for per_device in (2, 4, 8, 32):
        global_batch = per_device * WORLD_SIZE
        # (a) today: one draw per batch, and every rank replays the same draw.
        shared = [1.0 if rng.random() <= p else 0.0 for _ in range(steps)]
        # (b) one draw per batch, ranks seeded independently.
        independent = [
            sum(1.0 if rng.random() <= p else 0.0 for _ in range(WORLD_SIZE)) / WORLD_SIZE
            for _ in range(steps)
        ]
        # (c) one draw per sample (reference SpecAugment behaviour).
        per_sample = [
            sum(1.0 if rng.random() <= p else 0.0 for _ in range(global_batch)) / global_batch
            for _ in range(steps)
        ]
        print(
            f"per_device_bs={per_device:2d} (global {global_batch:4d}): std  "
            f"shared-batch-gate={statistics.pstdev(shared):.4f}  "
            f"indep-batch-gate={statistics.pstdev(independent):.4f}  "
            f"per-sample-gate={statistics.pstdev(per_sample):.4f}"
        )
    print(f"All three average to p={p:.2f}; only the step-to-step variance differs.")


# ---------------------------------------------------------------------------
# 3. rank correlation
# ---------------------------------------------------------------------------


def analytic_rank_correlation(ranks: int = 8) -> None:
    """Replay torch's worker seeding for several ranks that all call set_seed(42)."""
    print("\n=== 3a. Analytic: base_seed / worker streams per rank ===")
    streams: list[list[float]] = []
    base_seeds: list[int] = []
    for _ in range(ranks):
        # transformers.set_seed(42): random / numpy / torch / cuda all seeded alike.
        random.seed(42)
        torch.manual_seed(42)
        # torch DataLoader._BaseDataLoaderIter.__init__ with generator=None:
        base_seed = torch.empty((), dtype=torch.int64).random_(generator=None).item()
        base_seeds.append(base_seed)
        # torch _worker_loop: random.seed(base_seed + worker_id)
        worker0 = random.Random(base_seed)
        streams.append([round(worker0.random(), 9) for _ in range(6)])
    print(f"base_seed per rank: {base_seeds}")
    print(f"all ranks identical: {len(set(base_seeds)) == 1}")
    print(f"worker-0 gate draws, rank 0: {streams[0]}")
    print(f"worker-0 streams identical across ranks: {len({tuple(s) for s in streams}) == 1}")


class _ProbeDataset(Dataset):
    def __len__(self) -> int:
        return _PROBE_BATCH * _PROBE_STEPS * 2

    def __getitem__(self, index: int) -> dict:
        return {"index": index}


def probe_collate(features: list[dict]) -> dict:
    """Runs inside a DataLoader worker; mirrors the training collator's call site."""
    batch = len(features)
    probe = torch.ones(batch, _PROBE_TIME, NUM_BINS)
    mask = torch.ones(batch, _PROBE_TIME)
    digest, applied = _signature(probe, mask)
    return {"sig": digest, "applied": applied}


def _rank_probe(rank: int, out_path: Path, persistent: bool) -> None:
    from transformers import set_seed

    set_seed(42)  # exactly what Trainer.train() does on every rank
    loader = DataLoader(
        _ProbeDataset(),
        batch_size=_PROBE_BATCH,
        shuffle=False,
        num_workers=2,
        persistent_workers=persistent,
        collate_fn=probe_collate,
    )
    epochs = []
    for _ in range(2):
        batches = list(loader)[:_PROBE_STEPS]
        epochs.append({
            "sigs": [batch["sig"] for batch in batches],
            "applied": [bool(batch["applied"]) for batch in batches],
        })
    out_path.write_text(json.dumps({"rank": rank, "persistent": persistent, "epochs": epochs}))


def dataloader_probe(tmp_dir: Path) -> None:
    import multiprocessing as mp

    print("\n=== 3b. End-to-end: two 'ranks' with real DataLoader workers ===")
    ctx = mp.get_context("spawn")
    for persistent in (True, False):
        outputs = []
        procs = []
        for rank in range(2):
            out_path = tmp_dir / f"probe_p{int(persistent)}_rank{rank}.json"
            proc = ctx.Process(target=_rank_probe, args=(rank, out_path, persistent))
            proc.start()
            procs.append((proc, out_path))
            outputs.append(out_path)
        for proc, _ in procs:
            proc.join(180)
        payloads = [json.loads(path.read_text()) for path in outputs if path.exists()]
        if len(payloads) != 2:
            print(f"persistent_workers={persistent}: probe failed")
            continue
        epoch0, epoch1 = payloads[0]["epochs"]
        sig0, sig1 = epoch0["sigs"], payloads[1]["epochs"][0]["sigs"]
        applied0, applied1 = epoch0["applied"], payloads[1]["epochs"][0]["applied"]
        cross = sum(1 for a, b in zip(sig0, sig1, strict=True) if a == b)
        fired = sum(applied0)
        masked_sigs = {s for s, a in zip(sig0, applied0, strict=True) if a}
        print(f"persistent_workers={persistent}:")
        print(
            f"  gate fired on {fired}/{len(sig0)} batches "
            f"({fired / len(sig0):.2f}, config p={PROD['p']:.2f})"
        )
        print(f"  rank0 vs rank1, epoch 1: {cross}/{len(sig0)} batches share the mask layout")
        print(f"  rank0 gate decisions identical to rank1: {applied0 == applied1}")
        print(f"  rank0 epoch1 vs epoch2 identical: {epoch0['sigs'] == epoch1['sigs']}")
        print(f"  distinct mask layouts among the {fired} augmented batches: {len(masked_sigs)}")


# ---------------------------------------------------------------------------
# 5. waveform augmentations across ranks
# ---------------------------------------------------------------------------


def waveform_rank_identity(ranks: int = 8) -> None:
    """Run the real waveform augmentations under each rank's worker seeding.

    At step k every rank is served by worker ``k % NUM_WORKERS`` and that
    worker's RNG stream sits at the same position on every rank, so the speed
    factor, the noise clip, the SNR and the codec parameters are a function of
    (step, worker_id) alone. This hashes the augmented waveform for several
    simulated ranks, once with today's seeding and once with a rank-offset
    seeding, to size the missing entropy.
    """
    import numpy as np
    import yaml

    from qasr.augmentation import build_augmenter

    print("\n=== 5. Waveform augmentations across ranks (real classes) ===")
    recipe = yaml.safe_load((REPO_ROOT / PROBE_CONFIG).read_text())["augmentation"]
    augmenter = build_augmenter(recipe, sampling_rate=16_000)
    print(f"noise_dir={recipe['noise_injection'].get('noise_dir')}")

    seconds = np.arange(4 * 16_000, dtype=np.float32) / 16_000
    waveform = (0.2 * np.sin(2 * np.pi * 220.0 * seconds)).astype(np.float32)
    clean = hashlib.sha1(np.ascontiguousarray(waveform).tobytes()).hexdigest()[:16]

    def rank_digest(rank: int, *, offset_by_rank: bool) -> str:
        # transformers.set_seed(42) runs identically on every rank.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        base_seed = torch.empty((), dtype=torch.int64).random_(generator=None).item()
        worker_seed = base_seed + (rank * 1_000_003 if offset_by_rank else 0)
        # torch _worker_loop: random.seed(seed) + a SeedSequence-derived numpy seed.
        random.seed(worker_seed)
        entropy = [0, worker_seed & 0xFFFFFFFF, worker_seed >> 32, 0]
        np.random.seed(np.random.SeedSequence(entropy).generate_state(4, dtype=np.uint32))
        out = augmenter.augment_waveform(waveform.copy())
        return hashlib.sha1(np.ascontiguousarray(out).tobytes()).hexdigest()[:16]

    current = [rank_digest(rank, offset_by_rank=False) for rank in range(ranks)]
    offset = [rank_digest(rank, offset_by_rank=True) for rank in range(ranks)]
    untouched = sum(1 for digest in offset if digest == clean)
    print(f"distinct augmented waveforms over {ranks} ranks, today's seeding : "
          f"{len(set(current))}")
    print(f"distinct augmented waveforms over {ranks} ranks, rank-offset seed  : "
          f"{len(set(offset))} ({untouched} rank(s) had all three gates off, so they "
          "equal the clean input)")
    print("=> every rank applies byte-identical speed/noise/codec parameters to its")
    print("   own (different) audio at a given step.")


# ---------------------------------------------------------------------------
# 6. closed-form budget
# ---------------------------------------------------------------------------


def analytic_decomposition(measured_time: float, measured_freq: float) -> None:
    """Closed-form masked share, to show which draws spend the budget.

    E[share] = P(gate) x E[#masks] x E[width] / extent, with
    E[#masks] = (1 + num_masks) / 2 for ``randint(1, num_masks)`` and
    E[width] = m / 2 for ``randint(0, m)`` where m = extent x ratio x intensity.
    """
    p = PROD["p"]
    e_intensity = 0.65  # uniform(0.3, 1.0)
    e_masks_t = (1 + PROD["num_time_masks"]) / 2
    e_masks_f = (1 + PROD["num_freq_masks"]) / 2
    pred_t = p * e_masks_t * PROD["max_time_mask_ratio"] * e_intensity / 2
    pred_f = p * e_masks_f * PROD["max_freq_mask_ratio"] * e_intensity / 2
    ceiling = 2 * PROD["max_time_mask_ratio"]
    print("\n=== 6. Where the masking budget goes (closed form vs measured) ===")
    print(f"P(gate)={p:.2f}  E[#time masks]={e_masks_t:.2f}  "
          f"E[#freq masks]={e_masks_f:.2f}  E[intensity]={e_intensity:.2f}")
    print(f"time share: predicted={pred_t:.4f}  measured={measured_time:.4f}  "
          "(gap = mask overlaps + int() truncation)")
    print(f"freq share: predicted={pred_f:.4f}  measured={measured_freq:.4f}")
    print(f"multiplicative discount: gate x{p:.2f}, randint(0, m) x0.50, "
          f"intensity x{e_intensity:.2f}, E[#masks]/max x0.75")
    print(f"=> {measured_time / ceiling * 100:.1f}% of the nominal "
          f"2 x {PROD['max_time_mask_ratio'] * 100:.0f}% time budget reaches the model.")


def main() -> int:
    durations = _read_durations()
    if not durations:
        print(f"no durations found under {MANIFEST_GLOB}", file=sys.stderr)
        return 1
    print(f"SpecAugment audit - recipe: {PROD}")
    print(f"duration sample: n={len(durations)}  mean={statistics.fmean(durations):.2f}s  "
          f"p50={_quantile(durations, 0.5):.2f}s  p99={_quantile(durations, 0.99):.2f}s")
    measured = strength_profile(durations)
    gate_granularity()
    analytic_rank_correlation()
    tmp_dir = REPO_ROOT / "logs" / "specaug_probe"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dataloader_probe(tmp_dir)
    waveform_rank_identity()
    analytic_decomposition(*measured)
    print("\n--- nominal ceilings for reference ---")
    print(f"time: 2 masks x {PROD['max_time_mask_ratio'] * 100:.0f}% = "
          f"{2 * PROD['max_time_mask_ratio'] * 100:.0f}% of the valid frames; "
          f"freq: 2 masks x {PROD['max_freq_mask_ratio'] * 100:.0f}% = "
          f"{2 * PROD['max_freq_mask_ratio'] * 100:.0f}% of {NUM_BINS} bins")
    print("randint(0, m) has E[width] = m/2, so the expected share is half the ceiling")
    print("before overlaps and the intensity draw are taken into account.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
