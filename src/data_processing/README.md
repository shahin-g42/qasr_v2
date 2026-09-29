# Sequential Corpus Bundles — staged build

Builds sequentially numbered training **bundles** for the QASR hybrid ASR model.
A **bundle N** = `{ar,en,zh,hi,ml}/bN`, each language exactly **100 000** samples
→ **500 000 samples (~1 111 h)** per bundle. Bundles are numbered `0..K-1` where
`K = min` batches produced across the five languages, so a bundle ships only when
all five languages have a full `bN`.

Samples are filtered for transcript **quality and richness**, drawn from both
**internal** sources (the v7.6 manifest tree, the raw q3asr SFT tree,
already-extracted Emilia-ZH — paths kept as-is) and **external** HuggingFace
sources (audio downloaded to fixed
16 kHz mono FLAC paths). The LLM `corrector` runs **co-located** on each node at
`http://localhost:8010/v1` — there is no load balancer and no multi-URL pool.

---

## Pipeline at a glance

| Stage | Module | Runs on | Needs GPU / corrector | What it does |
|------:|--------|---------|:---------------------:|--------------|
| 0 Preflight | `data_processing.datasets.preflight` | 1 node | no | Resolve repos, probe column names, measure duration coverage |
| 1 Prepare | `data_processing.prepare` | **all 9 nodes** (sharded by source) | no | Stream metadata → normalize → gate → accent-tag → score → **sorted candidate pools** |
| 2 Assemble | `data_processing.assemble` | **per language; hash-sliced across nodes** | **yes** (local corrector) | Best-first merge → LLM-correct → **inline-materialize external audio** → sequential `bNNNN` batches |
| 3 Materialize | `data_processing.materialize` | any node | no | Verify/backfill every external clip is 16 kHz mono FLAC (idempotent) |
| 4 Bundle + audit | `data_processing.bundle` (dispatcher: `bundle`) | 1 node | no | Group `bN` across langs → `MANIFEST.json`; audit counts / overlap / eval-leak / audio — **exit 1 = not shippable** |

All five stages are implemented and tested. The dispatcher is the recommended
entry point; each stage module can still be invoked directly as shown below.

> **Two entry points.** `python3 -m data_processing.build_corpus <stage>` is the
> **dispatcher** and the recommended entry: `preflight`, `prepare`, `assemble`,
> `materialize`, `bundle`, or no subcommand for the **legacy single-process
> build** (`--config configs/corpus.yaml`; it streams, gates, corrects and
> distributes one language in one process — it cannot shard ingest across
> nodes and re-streams each source's metadata per language). The dispatcher
> feeds each stage its section of `configs/corpus.yaml` as defaults (CLI
> flags still win) and delegates by literal `main(argv)` passthrough, so every
> stage keeps its own argparse, `--report` and exit codes. The staged path is
> the 9-node build: Stage 1 fans ingest out across all nodes into shared
> pools, Stage 2 assembles each language beside its own co-located corrector
> — whole-language, or as N hash slices spread across the ranks.

### Why staged

- The corrector is `localhost:8010` on **every** node → the LLM stage must run
  where the model is (Stage 2 is per-language, co-located).
- `SeenLedger` is SQLite-WAL, which corrupts under concurrent writers → **one
  ledger file per language slice, one writer ever** (slices have disjoint,
  hash-defined path sets; languages have disjoint audio pools); ledgers live
  in the workspace scratch area (`$QASR/scratch/corpus/ledgers`) by default,
  with node-local disk preferred. Cross-language overlap is audited to zero
  afterwards.
- Source caps (`max_per_source_fraction=0.40`) and the diversity floor
  (`min_distinct_text_fraction=0.50`, the "zh guard") need a whole-language pool
  view → assembly is per-language, not per-source.
- The dominant cost is streaming ingest + audio download (I/O), not the triaged
  LLM (~18% of samples) → Stages 1/3 shard across all 9 nodes.

---

## On-disk layout

```
$POOL_DIR/<lang>/<source>/part-00000.jsonl[.gz]     # Stage 1: sorted candidate pools (shared)
$AUDIO_ROOT/<lang>/<source>/<blake2b16>.flac        # Stage 2/3: materialized external audio, 16 kHz mono (shared)
$OUT_DIR/<lang>/train_<lang>_b0000_p0000.jsonl      # Stage 2: batch manifest (canonical 4-key)
$OUT_DIR/<lang>/train_<lang>_b0000_p0000.meta.jsonl # Stage 2: sidecar (provenance, quality, richness)
$OUT_DIR/MANIFEST.json                              # Stage 4: shipping doc (green audits only)
$LEDGER_DIR/<lang>[_p<k>].sqlite3                   # per-language slice ledger — one writer per file
```

- `<blake2b16>` = `blake2b(native_id, digest_size=8).hexdigest()` (16 hex chars).
  This one string is the identity: the ledger key, the manifest path, and the
  audio file all derive from it, so they cannot disagree.
- **Internal/local audio is never copied or resampled** — a batch manifest row
  for a local clip (a `LOCAL_JSONL` v7.6 tree, or the `LOCAL_AUDIO` Emilia-ZH
  extract, which is a **44 kHz** tree) points at its original absolute path and
  keeps its native sample rate. Only **external** Hub clips are fetched and
  rewritten to **16 kHz mono FLAC** under `$AUDIO_ROOT`
  (`DatasetSpec.needs_materialization` is true for exactly the HF/CUSTOM kinds,
  false for `LOCAL_JSONL`/`LOCAL_AUDIO`). The corpus is therefore mixed-rate by
  design, exactly as the v7.6 tree already is; final resampling is the training
  feature extractor's job, not the build's.
- Batch manifests carry exactly `{audio_filepath, duration, text, lang}` and are
  consumable by training as-is.
- `_p<k>` is the **part index within the batch label**: a whole-language
  assemble writes part 0; a hash-sliced multi-node assemble writes one part
  per slice, and the parts of one label compose one whole batch (each slice
  writes `batch_size / N` rows per part — see *Multi-node slicing* below).

---

## Registry — what each node pulls

`data_processing.datasets.registry` holds **29 sources**: 5 internal v7.6
trees + the 5-language q3asr SFT tree (raw, verbatim — same audio as the v7.6
q3asr shards by design; the ledger's path claims keep one transcript per clip)
plus 19 external, all `verified=True` against the live Hub API. **7 are gated**
(need `HF_TOKEN` + accepted Hub terms). The two Common Voice 17 entries
(`cv17_ar`, `cv17_ml`) were dropped 2025-10: Mozilla withdrew CV from the Hub
(now the Mozilla Data Collective), so the repos resolve to zero data files and
streaming raises `EmptyDatasetError`. What "internal" means is defined by
`configs/corpus/internal_ingest.yaml` (`train_manifest`); a test pins the
registry to that file. Both internal trees' eval sets feed the leak gates:
`--root` (v7.6) plus `--eval-root` (SFT) in stages 2 and 4.

| lang | internal v7.6 (train/eval shards) | q3asr SFT (raw) | external sources |
|------|-----------------------------------|-----------------|------------------|
| `ar` | ✓ (29 / 2) | ✓ | `masc_ar`, `fleurs_ar_eg`, `arabic_speech_corpus` |
| `zh` | ✓ (1 / 1) + `emilia_zh_local` (44 kHz) | ✓ | `aishell1`, `aishell3`, `wenetspeech`\*, `fleurs_cmn_hans` |
| `en` | ✓ (6 / 2) | ✓ | `peoples_speech`, `gigaspeech`\*, `librispeech`, `voxpopuli_en` |
| `hi` | ✓ (1 / 0) | ✓ (its eval sets) | `shrutilipi_hi`\*, `indicvoices_hi`\*, `kathbath_hi`\*, `fleurs_hi_in` |
| `ml` | ✓ (2 / 1) | ✓ | `shrutilipi_ml`\*, `indicvoices_ml`\*, `fleurs_ml_in` |

`\*` = gated. Ungated supply is the binding constraint: **`hi` has only
`fleurs_hi_in` (~12 h) without a token**, so it cannot fill a 100k batch unless
the gated ai4bharat sources are accepted; `ml` is likewise thin (v7.6 tree +
~9 h `fleurs_ml_in`). Add `--no-gated` for a first tokenless smoke run, but
then expect only `ar`/`zh`/`en` to fill a batch.

---

## Prerequisites

```bash
export QASR=/lustrefs/shared/shahin.konadath/workspace/train/qasr   # repo root on Lustre
cd "$QASR"
export PYTHONPATH="$QASR/src"
export TOKENIZERS_PARALLELISM=false
export HF_TOKEN=hf_xxx            # MANDATORY for hi & ml (all large sources are gated)

# Shared build paths (all nodes must mount these)
export INTERNAL_ROOT="$QASR/training_manifests/v7.6"   # or $QASR_INTERNAL_ROOT
export POOL_DIR="$QASR/corpus/pool"
export AUDIO_ROOT="$QASR/corpus/audio"
export OUT_DIR="$QASR/training_manifests/v8.0"
export LOGS="$QASR/logs/corpus"; mkdir -p "$LOGS"
# Per-language ledgers: one writer per file, ever. Workspace scratch by
# default; point at node-local disk when the node has it.
export LEDGER_DIR="$QASR/scratch/corpus/ledgers"; mkdir -p "$LEDGER_DIR"
```

- Build nodes need `pip install 'datasets>=3.0' 'huggingface-hub[hf_transfer]'`.
- Gated terms must be accepted on the Hub for `shrutilipi_hi/ml`, `indicvoices_hi/ml`,
  `kathbath_hi`, `wenetspeech` (zh), `gigaspeech` (en). Without them, **hi cannot
  fill a 100k batch** (only ~12 h of fleurs is ungated) and ml is thin.
- Storage under `$AUDIO_ROOT`: ~16 GB per 100k external clips at 16 kHz.
- The `corrector` vLLM server is already up on all 9 nodes (port `8010:8000`,
  `--served-model-name corrector`). Health-check it before Stage 2:
  `curl -sf http://localhost:8010/health`.

---

## Quickstart — one node, tiny, no GPU

Smoke-test the whole deterministic path locally before touching the cluster:

```bash
# 1) prepare a couple of internal sources, 500 rows each
python3 -m data_processing.prepare \
  --pool-dir /tmp/pool --audio-root /tmp/audio --root "$INTERNAL_ROOT" \
  --langs ar --only internal_v76_ar --limit 500 --report /tmp/prepare.json -v

# 2) assemble ONE batch of 1 000, no LLM, no external download
python3 -m data_processing.assemble \
  --lang ar --pool-dir /tmp/pool --out-dir /tmp/out --audio-root /tmp/audio \
  --ledger /tmp/ar.sqlite3 --root "$INTERNAL_ROOT" \
  --batch-size 1000 --batches 1 --no-materialize --report /tmp/assemble.json -v

# inspect
head -n1 /tmp/out/ar/train_ar_b0000_p0000.jsonl
python3 -c "import json;print(json.load(open('/tmp/assemble.json'))['batch_hours'])"
```

Add `--dry-run` to `assemble` to plan without writing manifests or claiming paths
(in-memory ledger). Add `--llm-url http://localhost:8010/v1` to enable correction.

---

## Stage 0 — Preflight (once, any node)

```bash
python3 -m data_processing.datasets.preflight --probe-fields --json > "$LOGS/preflight.json"
```

Fails loudly (exit 1) if a repo does not resolve, a `FieldMap` column is wrong, or
a local source has no duration coverage. **Do not skip this** — sources rot: the
Hub withdrew Common Voice outright (2025-10), and preflight is the only cheap
proof the registry still matches reality before a multi-hour, 9-node run depends
on it.

---

## Stage 1 — Prepare (all 9 nodes, GPU-free)

Each node takes a disjoint, deterministic slice of the registry
(`index % num_nodes == node_rank` over the sorted `(lang, source)` list), so nodes
write **disjoint files** into the shared `$POOL_DIR` and never coordinate.

```bash
python3 -m data_processing.prepare \
  --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \
  --node-rank "$RANK" --num-nodes 9 \
  --min-richness 0.0 \
  --report "$LOGS/prepare_rank${RANK}.json" -v
```

Key flags: `--jobs N` (spawned worker processes over this node's sources;
default 1, drivers set 4 — one crashed source never kills the node) ·
`--langs ar,zh` (default: all five) · `--only <name>` (repeatable) ·
`--no-gated` (skip gated sources; first tokenless run) · `--no-probe` (don't
`soundfile.info` local clips missing a duration) · `--shard-size` · `--gzip` ·
`--limit N` (smoke test) · `--min-richness` · `--gate-duration` · `--diacritic-policy`.

What it does per row: derive the final `audio_filepath` identity, `normalize`,
run the **quality gate**, score **richness**, `accent.detect`, and — for a local
clip whose metadata omits duration — probe it from the audio header. External clips
with no metadata duration are kept with `duration=None` (deferred to Stage 2);
local clips whose header can't be read are dropped (unusable for training).
Survivors are buffered into per-source shards **sorted by `composite = quality × richness`
descending**, which is what lets Stage 2 merge best-first.

---

## Stage 2 — Assemble + correct (per language, hash-sliced across nodes)

One process per (language, slice). It merges that language's pool shards
**best-first**, LLM-corrects the triaged ~18%, **materializes each external clip
inline at the moment it is selected** (after the distributor confirms it will
actually land, so nothing is over-downloaded), and fills sequential batches
`b0000`, `b0001`, …

**Multi-node slicing.** A language can be assembled by several nodes at once:
`--pool-part K/N` admits only the pool rows with
`blake2b(audio_filepath) % N == K`, and `--batch-part K` writes part `K` of
every batch (`train_<lang>_bNNNN_p<KKKK>` on disk). The hash keys on
`audio_filepath`, so a clip's duplicate rows (the v7.6 cleaned and raw q3asr
SFT copies share the path) always land in one slice, where the slice's own
ledger still dedups them. Each slice keeps its own ledger
(`<lang>_p<k>.sqlite3`); `N` must divide `--batch-size`, and with `N` slices
each writing `batch_size / N` rows per part the parts of one label compose
one whole batch — the 100k-per-`bN` bundle contract is unchanged. Keep
`N = 1` (the whole-language, one-writer default) for a language that cannot
fill half a batch per slice. `--max-per-text` applies per slice, so a
transcript can appear up to `2N` times per language (the diversity floor is
still enforced per slice). `scripts/corpus/plan_distribution.py` and the
multi-node `run_internal_only.sh` flow pick `N` and the rank→slice
assignment automatically from the finished pools.

```bash
curl -sf http://localhost:8010/health >/dev/null || { echo "corrector down"; exit 1; }

python3 -m data_processing.assemble \
  --lang "$L" \
  --pool-dir "$POOL_DIR" --out-dir "$OUT_DIR" --audio-root "$AUDIO_ROOT" \
  --ledger "$LEDGER_DIR/${L}.sqlite3" --root "$INTERNAL_ROOT" \
  --eval-root "$QASR_SFT_ROOT" \
  --llm-url http://localhost:8010/v1 --llm-model corrector \
  --exclude-eval \
  --batches 0 \
  --report "$LOGS/assemble_${L}.json" -v
```

Key flags:
- `--ledger` — **per-language or per-slice**, one writer per file. Never
  point two processes at one ledger.
- `--pool-part K/N`, `--batch-part K` — assemble one hash slice of a language
  across N nodes (see *Multi-node slicing*). `K/N` outside `0 <= K < N` is
  rejected, `N > 1` must divide `--batch-size`, and `--batch-part` defaults
  to the pool-part index.
- `--llm-prompt rich|compact` — `rich` (default) selects the language-
  specialized cleaner prompts (Arabic dialect/diacritics-aware; en/zh/hi/ml
  conventions); `compact` is the old minimal prompt.
- `--fetch-workers N` — windowed parallel prefetch of external audio ahead
  of placement (default 16; 0 = serial inline). A clip fetched while
  admission changed is simply left on disk — idempotent, never wasted.
- `--batches 0` — drain the pool (writes a short trailing batch). Use `--batches N`
  to cap at N full batches. Batch index **continues across runs** via the ledger,
  so a re-run appends `b0001`, `b0002`, … — this is how you add bundles later.
- `--exclude-eval` — load `eval_*.jsonl` (under `--root`) as ledger exclusions
  first, making the measured ar eval/train leak structurally impossible.
- `--eval-root PATH` — a **second** internal tree whose `eval_*.jsonl` also join
  the exclusions (repeatable). Pass `$QASR_SFT_ROOT` (the raw q3asr SFT
  manifests): that tree ships its own `eval_<lang>_q3asr.jsonl` sets.
- `--no-materialize` — skip external download (internal-only build).
- `--llm-all` — correct every sample, not just the triaged ~18%.
- `--min-richness`, `--gate-duration` — quality knobs (see below).
- `--max-per-source-fraction 0.40`, `--max-per-text 2`,
  `--min-distinct-text-fraction 0.50` — distribution/diversity policy.
- `--dry-run` — in-memory ledger, write nothing.

Output: `$OUT_DIR/<lang>/train_<lang>_bNNNN_p0000.jsonl` + `.meta.jsonl` (a
sliced run writes its own part index instead of `p0000`).

---

## Stage 3 — Materialize / verify (any node, idempotent)

Confirms every external clip the batches reference exists at 16 kHz mono, and
re-fetches the missing ones in parallel from the pool's `native_id`. Safe to run
on a different node than Stage 2 (it reads shared `$OUT_DIR`/`$POOL_DIR`).

```bash
python3 -m data_processing.materialize \
  --lang "$L" --out-dir "$OUT_DIR" --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" \
  --workers 16 --report "$LOGS/materialize_${L}.json"
```

Stage 2 already materializes inline; Stage 3 is the parallel **backfill/verify**
pass that closes any gap (interrupted run, cleaned `$AUDIO_ROOT`, refused batch).

`--prewarm --budget N` (no `--out-dir`) skips batch verification and instead
walks the language's pool in composite order, materializing the top-N external
candidates ahead of Stage 2 selection (N ≈ 1.3 × 100k) — the Stage 2b
accelerator for the four nodes assemble does not use.

---

## Stage 4 — Bundle + audit

`data_processing.bundle` (dispatcher: `build_corpus bundle`; the
`scripts/assemble_bundles.py` shim routes here) groups the five languages' `bN`
into `bundle_NNNN` and writes `$OUT_DIR/MANIFEST.json` — per-language per-batch
paths, rows, hours, source/accent composition, external/internal split,
distinct-text fraction and quality/richness/diacritics means, aggregated from
the sidecars (no re-gating). A bundle ships only when all five languages have
a full 100k `bN`; `K = min`, surplus full batches are reported and ship in
later runs once the laggards catch up.

The audit hard-gates the ship (**exit 1 on any failure**): count contract,
strict canonical record shape, no duplicate `audio_filepath` within or across
languages, zero eval leak against the v7.6 **and** q3asr-SFT eval sets
(`--root` + `--eval-root "$QASR_SFT_ROOT"`), every external clip
present at 16 kHz mono (`--spot-check N` for a sampled fast pass), the ≥ 0.50
diversity floor recomputed from the shipped texts, positive durations. A
failed audit never overwrites an existing good `MANIFEST.json`; `--dry-run`
audits and reports without writing anything.

```bash
python3 -m data_processing.build_corpus bundle \
  --out-dir "$OUT_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \
  --eval-root "$QASR_SFT_ROOT" \
  --report "$LOGS/bundle.json"
```

The older tools remain useful for ad-hoc checks:

```bash
python3 scripts/audit_training_manifests.py --manifest-dir "$OUT_DIR"   # counts, durations, out-of-band
python3 scripts/check_manifest_overlap.py  ...                          # eval-leak + cross-language path overlap
```

---

## Running on multiple machines (9 nodes)

Nodes 0–3 serve `Qwen3.8-Flash-Next-FP8`; nodes 4–8 serve `Qwen3.8-27B-FP8`; all
expose it as `corrector` on `localhost:8010`. Stage 1 uses **all 9**; the
campaign's Stage 2 maps **5** (one language each), and any language can be
hash-sliced across more nodes with `--pool-part`/`--batch-part` (see
*Multi-node slicing* in Stage 2).

**Production drivers exist**: `scripts/corpus/*.slurm` (+ `env.sh` and the
runbook in `scripts/corpus/README.md`) wrap everything below with a shared
environment, per-node JSON reports and a health-gated Stage 2 —
`sbatch scripts/corpus/stage1_prepare.slurm` etc. The inline examples in this
section are the manual fallback.

### Stage 1 — SLURM, all 9 nodes at once

```bash
#!/bin/bash
#SBATCH --job-name=corpus-prepare
#SBATCH --nodes=9
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/corpus/prepare-%N.out
set -euo pipefail
cd "$QASR"; export PYTHONPATH="$QASR/src"
srun bash -c '
  python3 -m data_processing.prepare \
    --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \
    --node-rank "$SLURM_NODEID" --num-nodes "$SLURM_NNODES" \
    --report "$LOGS/prepare_rank${SLURM_NODEID}.json" -v
'
```

`$QASR`, `$POOL_DIR`, `$AUDIO_ROOT`, `$INTERNAL_ROOT`, `$LOGS`, `$HF_TOKEN` (all
exported in Prerequisites) are forwarded by `sbatch`/`srun` to every task, so
export them before submitting. No GPU and no corrector are needed for Stage 1.

### Stage 1 — manual (parallel-ssh / per-node)

```bash
# on each node, set R to that node's rank (0..8), then run:
R=0   # <- this node's rank
python3 -m data_processing.prepare \
  --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \
  --node-rank "$R" --num-nodes 9 --report "$LOGS/prepare_rank${R}.json" -v
```

### Stage 2 — 5 languages across 5 nodes

Pin one language per node; each uses **its own** local corrector and **its own**
ledger. Put the highest-triage languages on the Flash-Next nodes (0–3): zh (~37%
triaged) and hi are the LLM-heaviest.

| node | lang | note |
|-----:|------|------|
| 0 | zh | highest triage → Flash-Next |
| 1 | hi | gated + ASR-labelled → Flash-Next |
| 2 | ar | internal-seeded, ample |
| 3 | en | external-only, download-heavy |
| 4 | ml | thin; needs gated |

SLURM (5 nodes, language chosen by node id). **Assemble is a CPU-only client** —
the persistent `corrector` container already owns the GPUs on each node, so this
job requests **no `--gres`**; it only needs to land on a node where that
container is already up (the `curl` health-check below enforces exactly that):

```bash
#!/bin/bash
#SBATCH --job-name=corpus-assemble
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/corpus/assemble-%N.out
set -euo pipefail
cd "$QASR"; export PYTHONPATH="$QASR/src"
srun bash -c '
  LANGS=(zh hi ar en ml)
  L=${LANGS[$SLURM_NODEID]}
  curl -sf http://localhost:8010/health >/dev/null || { echo "no corrector for $L"; exit 1; }
  python3 -m data_processing.assemble \
    --lang "$L" --pool-dir "$POOL_DIR" --out-dir "$OUT_DIR" --audio-root "$AUDIO_ROOT" \
    --ledger "$LEDGER_DIR/${L}.sqlite3" --root "$INTERNAL_ROOT" \
    --llm-url http://localhost:8010/v1 --llm-model corrector --exclude-eval \
    --batches 0 --report "$LOGS/assemble_${L}.json" -v
'
```

Manual (run the Stage-2 command from that stage on each of the 5 nodes, with `L`
set to the node's language from the table). Because ledgers are per-language and
each file has exactly one writer, the five processes never contend. **One writer
per ledger file, ever** — never share a ledger between processes (SQLite-WAL
corrupts under concurrent writers). Ledgers default to the workspace scratch
area (`$QASR/scratch/corpus/ledgers`), and Stage 2 additionally needs `$OUT_DIR`
forwarded.

### Stage 3 — verify (any nodes, in parallel by language)

```bash
# fan out one per language (same node or different — it is pure I/O on shared storage)
for L in ar en zh hi ml; do
  python3 -m data_processing.materialize --lang "$L" \
    --out-dir "$OUT_DIR" --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" \
    --workers 16 --report "$LOGS/materialize_${L}.json" &
done; wait
```

### Full 9-node walkthrough

```bash
# 0) once, any node
python3 -m data_processing.datasets.preflight --probe-fields --json > "$LOGS/preflight.json"

# 1) all 9 nodes  -> $POOL_DIR filled with sorted shards
sbatch scripts/corpus/stage1_prepare.slurm     # or the sbatch block above

# 2) 5 nodes, one per language -> $OUT_DIR/<lang>/train_<lang>_bNNNN_*
#    (optionally concurrent on the rest: sbatch scripts/corpus/stage2b_prewarm.slurm)
sbatch scripts/corpus/stage2_assemble.slurm

# 3) verify/backfill external audio, per language
for L in ar en zh hi ml; do python3 -m data_processing.materialize --lang "$L" \
  --out-dir "$OUT_DIR" --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" \
  --workers 32 --report "$LOGS/materialize_${L}.json" & done; wait

# 4) bundle + audit: K = min full-batch count; the job's exit code is the ship gate
sbatch scripts/corpus/stage4_bundle.slurm
```

---

## Quality gates & knobs

The gate (`data_processing/quality.py`) is arithmetic only — no model judgement.
Verdicts: `PASS` / `FIXABLE` / `REJECT`. Ranking key is `composite = quality × richness`.

**Duration gates are OFF by default** (`QualityConfig.gate_duration = False`).
Nothing is rejected for being too short, too long, or spoken at an odd rate;
duration is still *measured* (it fills the manifest and informs ranking), it just
never rejects. Pass **`--gate-duration`** to `prepare`/`assemble` to re-arm the
`min_duration`/`max_duration` band and the speech-rate band.

**Richness floor is OFF by default** (`min_richness = 0.0`). Set
**`--min-richness 0.3`** (roughly) to drop the trivial tail ("yes", one repeated
word) beyond what `too_short` catches. Richness = lexical type-token ratio +
distinct content-unit count + length-in-band bonus, computed from text alone, so
Stage 1 can rank external clips before their audio is fetched.

Always-on text gates: script purity (`min_script_ratio 0.80`), contamination
(URLs, `[noise]`, speaker labels, timestamps), Arabic over-vocalization
(`max_diacritics_ratio 0.40`), degenerate repetition (`max_char_run 4`,
`max_word_fraction 0.50`), length (`min_text_chars 2`, `max_text_chars 2000`).

Distribution policy (`distribute.py`): identity is `audio_filepath` alone (two
rooms = two samples); `max_per_text 2` caps renditions of one transcript;
`min_distinct_text_fraction 0.50` refuses a batch that is mostly copies (the zh
guard); `max_per_source_fraction 0.40` keeps a batch mixed across ≥3 sources.

---

## Resuming, idempotency & ledger safety

- **Prepare** is authoritative per source: a re-run clears that source's stale
  `part-*.jsonl*` first, so a shorter second run can't leave high-index shards
  behind for Stage 2 to double-count.
- **Assemble** resumes from the ledger: already-claimed paths are skipped and the
  batch index continues (`b0000` → `b0001` …). A short overflow tail is *released*,
  not stranded, so the next run can reuse it. A sliced run resumes per slice —
  each slice's ledger continues its own label sequence.
- **Materialize** is idempotent: existing correct-duration files are skipped.
- **Ledger**: one per language (or per slice), one writer each. The default
  lives under the workspace scratch area, so any node can resume — but never
  share one ledger file across processes (WAL corrupts under concurrent
  writers).

---

## Verification checklist before shipping a bundle

- Exactly 100 000 rows per language per `bN` (Stage 4 enforces; `audit_training_manifests.py` counts).
- Zero eval-path leak (`--exclude-eval` + `check_manifest_overlap.py`).
- Zero cross-language / cross-node `audio_filepath` overlap.
- Diversity floor met per batch (no `refused_low_diversity` in the assemble report).
- Every external clip present under `$AUDIO_ROOT` at 16 kHz mono (Stage 3 report: `errors` empty).
- Richness / duration distributions sane in the sidecar `.meta.jsonl`.
