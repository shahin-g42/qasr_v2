# QASR v7.6 training configs

Four-phase pipeline for the **v7.6** data generation, on 8 nodes × 8 GPUs (64 GPUs).
Run them in order; each phase reads the previous phase's `output_dir`.

Every phase runs **exactly one full-coverage epoch** under
`sampling_strategy: stratified` (`StratifiedLanguageSampler` in
`src/qasr/sampling.py`):

- **Every record of every language is seen at least once per epoch.** Arabic —
  always the largest — is seen *exactly* once; smaller languages cycle
  (duplicate) to fill their share. Coverage is a bijective per-rank partition,
  not a statistical estimate, and `max_steps` in each YAML equals the exact
  number of batches in one epoch.
- **Every per-device mini-batch holds an equal number of samples per language**
  (2 per language at batch 10; 6 per language at EAGLE's batch 30). This is why
  per-device batches are 10/30, not 8/32 — they must divide by 5 languages.
- Repeat views of small-language records are differentiated by the augmentation
  stack (speed 0.85–1.15×, MUSAN noise 5–20 dB, codec, SpecAugment, each p=0.37).

| # | Config | Trains | Steps (=1 epoch) | Global batch | LR | Repeats (per epoch) |
|---|--------|--------|-----------------:|-------------:|-----|---------------------|
| 1 | `01_projector_8node.yaml` | projector (~4.3M) | 903,217 | 640 | 1e-4 cosine | ar×1.0 en×2.4 hi×10 ml×94 zh×213 |
| 2 | `02_full_8node.yaml` | everything (3.62B) | 903,217 | 640 | 4e-5 WSD | ar×1.0 en×2.4 hi×10 ml×94 zh×213 |
| 3 | `03_hq_8node.yaml` | everything, curated pool | 161,639 | 640 | 4e-6 cosine | ar×1.0 en×1.1 hi×3.6 ml×47 zh×132 |
| 4 | `04_eagle_8node.yaml` | EAGLE head (~8.4M) | 103,194 | 1,920 | 3e-4 cosine | ar×1.0 en×1.2 hi×6.8 ml×180 zh×253 |

Chain: `output/initial` → **P1** → `output/v76/projector` → **P2** →
`output/v76/full` → **P3** → `output/v76/hq` → **P4** → `output/v76/eagle`.

## Data sources

Two manifest trees, both under
`/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/`. They contain
different material (the SFT tree is not a copy of v7.6's q3asr) and both are
used in full.

**`training_manifests/v7.6/`** — LLM-cleaned (ITN, punctuation, Arabic dialect
preservation + diacritic restoration). 46 corpora, 115,414,904 records. Clean
snapshot: 0 leaked `*_rejected` piles, no `<<<>>>` fences.

**`q3asr_sft_manifests/`** — q3asr envelopes filtered by
`scripts/prepare_q3asr_filter.py` (LID, duration probe, 512-token gate).
61,098,352 train + 213,485 eval records, fully duration-complete.

### Train pools per phase

| Phase | Pool | Records | Per-language (ar / en / hi / ml / zh) |
|---|---|---:|---|
| 1, 2 | everything | 176,454,622 | 115.6M / 47.7M / 11.4M / 1.23M / 0.54M |
| 3 | curated + cleaned hi | 45,785,614 | 20.7M / 18.7M / 5.8M / 0.44M / 0.16M |
| 4 | deployed distribution | 78,928,385 | 39.6M / 33.1M / 5.8M / 0.22M / 0.16M |

Phases 1 and 2 use the identical full pool: under stratified sampling the epoch
layout is driven by record counts, not hours, so the old reason to restrict
Phase 1 to duration-complete corpora (hours-weighted sampling) no longer applies.

### Eval sets (all phases)

All 11 eval manifests that exist, 272,021 records: v7.6 cleaned evals for
ar/en/ml/zh plus the SFT evals for all five languages. The two reference styles
differ (cleaned vs verbatim) — compare a language's WER against itself over
time, not across styles. `eval_ml_inworld_as_hi.jsonl` is excluded everywhere:
it is Malayalam audio filed under Hindi.

## Steering notes

- **Phase 1 early exit.** The projector is a 4.3M-param MLP and typically
  converges within tens of thousands of steps. The full-epoch budget is the
  guarantee, not an obligation: if eval loss is flat for 3 consecutive evals,
  stop and hand the checkpoint to Phase 2.
- **Phase 2 WSD.** Warmup 9k, stable plateau, 90k decay tail. If per-language
  WER flattens for 2 consecutive evals in the stable region, stop and fire the
  decay branch from the best checkpoint.
- **VRAM.** 10/device is 25% above the 8/device that the ~50 GB/GPU Phase-2
  estimate was made for. Watch the first steps; fall back to 5/device
  (gb 320 — still divisible by 5; double `max_steps` to keep the full epoch).
- **EAGLE binding.** The head is bound to exact target weights — train it on
  the final Phase 3 checkpoint, and retrain it if the target ever changes.

## Before launching

```bash
PYTHONPATH=src python -m pytest tests/test_sampling.py tests/test_augmentation.py -q
```

Confirm the manifests exist on the cluster (the local v7.6 copy was still in
flight when these configs were written):

```bash
PYTHONPATH=src python -c "
import yaml,os
for p in sorted(os.listdir('configs/v7.6')):
    if not p.endswith('.yaml'): continue
    d=yaml.safe_load(open('configs/v7.6/'+p))
    miss=[q for k in ('train_manifest','eval_manifest') for v in [d.get(k) or {}]
          for ps in (v.values() if isinstance(v,dict) else []) for q in ps if not os.path.exists(q)]
    print(f'{p}: {len(miss)} missing'); [print('   ',q) for q in miss[:10]]
"
```

Smoke-test each phase before committing GPU-days:

```bash
sbatch --export=ALL,CONFIG=configs/v7.6/01_projector_8node.yaml scripts/qasr_train_8node.slurm --smoke-test
```

> **Launcher caveat.** `scripts/qasr_train_8node.slurm` calls `torchrun` in the
> script body with no `srun`, so under `sbatch` it runs on the first allocated
> node only — the other 56 ranks never join. Fix that (or launch per-node
> manually) **before** launching: every `max_steps` above is computed for
> world_size 64. If fewer ranks actually start, one epoch needs proportionally
> more steps than `max_steps` allows, training stops early, and the
> full-coverage guarantee is silently truncated (at 8 ranks you would see only
> ~1/8 of each language). The rank-0 log line
> `Stratified sampling: 5 languages, ...` prints the actual epoch layout —
> check `draws/language/epoch × languages ÷ global batch == max_steps` at
> launch.
