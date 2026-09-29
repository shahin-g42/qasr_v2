# Corpus production runbook — 9× H100 nodes

How to **drive** the staged corpus build on the cluster. The modules and their
flags are documented in `src/data_processing/README.md`; this file is the
operational guide: what to run, in what order, what to watch, and when the
output is shippable.

The build produces versioned **bundles**: bundle `N` = `{ar,en,zh,hi,ml}/bN`,
exactly **100 000 samples per language** (500k samples, ~1 111 h per bundle).
`K = min` full-batch count across the five languages; bundle `N` ships only
when all five languages have a full `bN`. Batch indices continue across runs
via the per-language ledgers, so later runs append `b0001, b0002, …` and grow
`K` — nothing is ever renumbered.

---

## Cluster layout

- 9 nodes × 8 H100, 96 CPU cores each.
- Every node runs the corrector as a **persistent vLLM container** that owns
  the GPUs, served as `corrector` at `http://localhost:8010/v1`. Nodes 0–3 are
  the faster Flash-Next FP8; nodes 4–8 are 27B FP8. Corpus jobs therefore
  request **no `--gres`** and never launch the corrector — where a stage needs
  it, the driver gates on `curl -sf http://localhost:8010/health`.

Node assignment during Stage 2 (picked from `SLURM_PROCID` inside the job):

| task rank | role | why |
|---:|---|---|
| 0 | assemble **zh** | highest triage (~37%) → Flash-Next |
| 1 | assemble **hi** | gated + ASR-labelled → Flash-Next |
| 2 | assemble **ar** | internal-seeded, ample |
| 3 | assemble **en** | external-only, download-heavy → Flash-Next |
| 4 | assemble **ml** | thin; needs gated sources |
| 5–8 | stage 2b prewarm, then free | pure I/O — no corrector needed |

`SLURM_PROCID` is the task rank *within the job*, so if your scheduler does not
hand out nodes in hostname order, pin the node lists explicitly so the
LLM-heavy languages land on the Flash-Next nodes:

```bash
sbatch --nodelist=gpu0,gpu1,gpu2,gpu3,gpu4 scripts/corpus/stage2_assemble.slurm
sbatch --nodelist=gpu5,gpu6,gpu7,gpu8 scripts/corpus/stage2b_prewarm.slurm
```

## Prerequisites (once per campaign)

1. **Environment** — every driver sources `scripts/corpus/env.sh`, which
   cds to the repo root (`$QASR`), sets `PYTHONPATH`, activates the conda env
   (`$CONDA_ENV`), exports the shared paths (`$INTERNAL_ROOT`, `$QASR_SFT_ROOT`,
   `$POOL_DIR`, `$AUDIO_ROOT`, `$OUT_DIR`, `$LOGS`) and the node-local
   `$LEDGER_DIR`, and enables `HF_HUB_ENABLE_HF_TRANSFER=1`. Every value is
   environment-overridable (`POOL_DIR=/tmp/pool sbatch …`); check them before
   the first run. `env.sh` mirrors the `paths:` section of
   `configs/corpus.yaml` — edit both together (the test suite pins the YAML
   side). There are TWO internal trees, both in the training checkout
   (`train/stt/qasr`) and both named file-for-file in
   `configs/v7.6/internal_ds_sources.yaml`: the v7.6 cleaned manifests
   (`$INTERNAL_ROOT`) and the raw q3asr SFT envelopes (`$QASR_SFT_ROOT`, the
   registry's `internal_sft_*` specs). Stage 2 excludes — and stage 4 audits
   against — the eval sets of BOTH trees (`--eval-root`).
2. **`HF_TOKEN`** with accepted gated terms for `ai4bharat/Shrutilipi`,
   `ai4bharat/IndicVoices`, `ai4bharat/Kathbath` (**hi/ml mandatory** — `hi`
   has only ~12 h of ungated fleurs and cannot fill a 100k batch without
   them), `wenet-e2e/wenetspeech` (zh) and `speechcolab/gigaspeech` (en,
   optional). `env.sh` warns if the token is unset; stage 0 fails on it.
3. **Python deps** on every node: the `qasr` conda env (`datasets>=3.0`,
   `huggingface-hub[hf_transfer]`, pyyaml, numpy/scipy, soundfile). If the
   training conda env cannot take the `datasets` stack (it fights the pinned
   torch/transformers), use the docker twin instead: run
   `scripts/corpus/docker_build_env.sh` once on any node (builds a shared venv
   at `$QASR/.hfenv` inside `python:3.12-slim`), then drive every stage with
   `scripts/corpus/docker.sh -m …` — see *Docker path* under *Manual path*.
4. **Storage**: ~16 GB per 100k external clips under `$AUDIO_ROOT` (16 kHz
   mono FLAC) — `en` is external-only, so ~16 GB per bundle for en alone.
5. **Ledgers node-local**: `$LEDGER_DIR` (default `/scratch/corpus/ledgers`)
   must be node-local disk, **never Lustre/NFS** — SQLite-WAL corrupts under
   concurrent writers. One ledger per language, one writer (the assembling
   node). If you must move a language to another node, copy its
   `<lang>.sqlite3` first.
6. **Corrector containers up** on the nodes stage 2 will use
   (`curl -sf http://localhost:8010/health` on each — stage 2 gates on this
   itself and fails in 5 minutes, not mid-run).

## Order of play

```
stage0 preflight (1 node)
        │
stage1 prepare (9 nodes, --jobs 4)
        │
        ├────────────────────────────┐
stage2 assemble (5 nodes)     stage2b prewarm (4 nodes)   ← concurrent
        │                              │
stage3 materialize (5 nodes) ←─────────┘
        │
stage4 bundle + audit (1 node) — exit code = ship gate
```

One paste, from the repo root on a submit node:

```bash
J0=$(sbatch --parsable scripts/corpus/stage0_preflight.slurm)
J1=$(sbatch --parsable --dependency=afterok:$J0 scripts/corpus/stage1_prepare.slurm)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/corpus/stage2_assemble.slurm)
J2B=$(sbatch --parsable --dependency=afterok:$J1 scripts/corpus/stage2b_prewarm.slurm)
J3=$(sbatch --parsable --dependency=afterok:$J2:$J2B scripts/corpus/stage3_materialize.slurm)
J4=$(sbatch --parsable --dependency=afterok:$J3 scripts/corpus/stage4_bundle.slurm)
echo "stage0=$J0 stage1=$J1 stage2=$J2 stage2b=$J2B stage3=$J3 stage4=$J4"
```

- **stage0 — preflight** (1 node, ≤4 h). Resolves every registry source: repo
  access, column names, duration coverage. Exit 1 if anything fails, so a bad
  repo_id or missing gated term surfaces *here*, not three hours into
  stage 1. Re-run before every campaign.
- **stage1 — prepare** (9 nodes, ≤48 h). GPU-free, `--jobs 4` spawned workers
  per node: stream metadata → normalize → gate → accent-tag → score → sorted
  candidate pools in `$POOL_DIR`. Ranks shard the registry
  (`index % 9 == node_rank`) so nodes write disjoint directories and never
  coordinate; a crashed source shows up in that node's JSON, never kills the
  node. 4 HF streams/node × 9 = 36 concurrent — safely under rate limits.
- **stage2 — assemble** (5 nodes, ≤72 h). One language per node per the table
  above, health-gated on the co-located corrector. Best-first merge of the
  pools → rich-prompt LLM correction of the triaged ~18% → inline
  materialization of external audio through a windowed `--fetch-workers 16`
  prefetch → sequential 100k batches `b0000, b0001, …` in `$OUT_DIR`. The LLM
  knobs (`--llm-prompt rich`, `--llm-batch 16`, `--llm-concurrency 64`) come
  from the `assemble:` section of `configs/corpus.yaml` via the dispatcher.
- **stage2b — prewarm** (4 nodes, ≤72 h, *optional but recommended*). Run
  concurrently with stage 2 on the four nodes it does not use: walk each
  language's pool in composite order and materialize the top ~130k external
  candidates ahead of selection, so assemble verify-and-skips instead of
  downloading on its critical path. Concurrent writers to the same
  deterministic path are safe (idempotent, atomic rename, same bytes from the
  same source).
- **stage3 — materialize** (5 nodes, ≤48 h). One node per language: verify
  every external clip the written batches reference exists at 16 kHz mono
  FLAC, backfill the missing ones with 32 workers. Closes any gap from a
  crashed stage 2 cheaply — instead of failing the stage-4 audit.
- **stage4 — bundle + audit** (1 node, ≤4 h). Group `bN` across the five
  languages, write `$OUT_DIR/MANIFEST.json`, and audit every ship gate.
  **The exit code is the verdict**: 0 = audited and manifested (shippable),
  1 = audit failed or nothing is shippable yet (and an existing good
  `MANIFEST.json` is left untouched). Wire it into your pipeline as the ship
  gate.

## Monitoring

Every stage writes a JSON report to `$LOGS` (`logs/corpus/`); SLURM
stdout/stderr land beside them as `stageN_<jobid>.log/.err`.

| Report | What to watch |
|---|---|
| `preflight_<jobid>.json` | every source `ok` — missing tokens/configs stop here |
| `prepare_node<r>_<jobid>.json` | per-source `read`/`emitted`, `counters.reject_gate` (why rows dropped), `parallel.ok/failed` |
| `assemble_<lang>_<jobid>.json` | `batches_written`, `batch_hours`, `counters.dist_*` placement outcomes, `counters.materialize_failed`, `llm.samples_failed`, `counters.erasure_reverted` |
| `prewarm_<lang>_<jobid>.json` | `written`/`skipped`/`errors` vs `budget` |
| `materialize_<lang>_<jobid>.json` | `written`/`skipped`, `errors` (want empty before stage 4) |
| `bundle_<jobid>.json` | `k`, `full_batches`, `surplus_full_batches`, `audit.ok` + `audit.failures` |

Live: `squeue -u $USER`, `tail -f $LOGS/stage2_<jobid>.log` (per-batch progress
lines), `sacct -j <jobid>` for exit codes.

## Resume rules

- **stage0 / stage1** — re-run at will. Prepare is per-source authoritative:
  a re-run clears that source's stale pool shards first, so a shorter second
  run cannot leave high-index shards behind for stage 2 to double-count.
- **stage2** — resumable from the ledger: already-claimed paths are skipped,
  the batch index continues (`b0000` → `b0001` …), and a short overflow tail
  is *released*, not stranded, so the next run reuses it. Re-run the same
  command.
- **stage2b / stage3** — idempotent: files already correct are skipped.
- **stage4** — re-run any time; a green audit (re)writes `MANIFEST.json`, a
  failed audit never overwrites a good one.

**Adding bundles later**: just re-run stage 2 (same ledgers) — each language
appends its next `bN`; when the laggard languages catch up, `K` grows and the
next stage-4 run ships the new bundles. The stage-4 report lists each
language's `surplus_full_batches` so you can see who is holding `K` back.

## Manual path (no SLURM)

The machines can be driven directly — same commands, ranks pinned by hand. On
each node, from the repo root, `source scripts/corpus/env.sh` first, then:

```bash
# stage 1 — one command per node, R = that node's rank (0..8):
R=3
nohup python3 -m data_processing.build_corpus prepare \
    --pool-dir "$POOL_DIR" --audio-root "$AUDIO_ROOT" --root "$INTERNAL_ROOT" \
    --node-rank "$R" --num-nodes 9 --jobs 4 \
    --report "$LOGS/prepare_node${R}_manual.json" \
    > "$LOGS/prepare_node${R}_manual.out" 2>&1 &

# stage 2 — on the node pinned to language L (corrector must be healthy):
L=ar
curl -sf http://localhost:8010/health || { echo "corrector down"; exit 1; }
nohup python3 -m data_processing.build_corpus assemble \
    --lang "$L" --pool-dir "$POOL_DIR" --out-dir "$OUT_DIR" --audio-root "$AUDIO_ROOT" \
    --ledger "$LEDGER_DIR/${L}.sqlite3" --root "$INTERNAL_ROOT" \
    --eval-root "$QASR_SFT_ROOT" \
    --llm-url "$CORRECTOR_URL" --batches 0 --exclude-eval \
    --report "$LOGS/assemble_${L}_manual.json" \
    > "$LOGS/assemble_${L}_manual.out" 2>&1 &
```

Stages 2b/3/4 mirror their `.slurm` files one-for-one: the
`python3 -m data_processing.build_corpus …` line inside each driver is the
entire command (stage 2b drops `--out-dir` and uses `--prewarm`; stage 3 adds
`--out-dir` and drops the corrector gate; stage 4 is a single foreground
command whose exit code you check — keep its `--eval-root "$QASR_SFT_ROOT"`
so the audit sees the SFT eval sets too).

### Docker path (training env without `datasets`)

When the conda env can't host the HF `datasets` stack, run the stages in a
plain python container — every command below stays identical, only
`python3 -m` becomes `scripts/corpus/docker.sh -m`. Source `env.sh` on the
host as usual first (its conda activation is irrelevant — python runs in the
container; the path vars it exports are forwarded by the wrapper):

```bash
# once per campaign, any node: builds the shared venv at $QASR/.hfenv
scripts/corpus/docker_build_env.sh

# then every stage, e.g. stage 0:
scripts/corpus/docker.sh -m data_processing.build_corpus preflight \
    --probe-fields --json --root "$INTERNAL_ROOT" \
    > "$LOGS/preflight_manual.json"
```

`docker.sh` is `env.sh`'s docker twin (same path defaults — edit together).
It mounts `$QASR`, `/scratch` (ledgers + **node-local** HF cache, so nine
nodes never race one cache) and `/vast` (Emilia), runs as your uid so nothing
on the shared filesystem ends up root-owned, and uses `--network host` so
`localhost:8010` (the co-located corrector) stays reachable for stage 2. The
venv's interpreter symlinks the image, so `IMAGE` in both scripts is a
matched pair — change them together and rebuild. If the cluster needs a
proxy for PyPI/Docker Hub, add the usual `-e https_proxy=…` passthroughs to
both scripts.

### Internal-only path (no Hub at all)

When the external side is what is blocked — gated terms, the `datasets`
dependency, a repo withdrawn from the Hub — the internal trees alone can be
processed and corrected on **one node, in the plain conda env**: no `datasets`,
no `HF_TOKEN`, no docker twin. `scripts/corpus/run_internal_only.sh` runs
stage 1 over the ten internal specs (`internal_v76_*` + `internal_sft_*`)
only, then stage 2 per language with the co-located corrector, health-gated
like the campaign driver:

```bash
scripts/corpus/run_internal_only.sh                  # all five languages
LIMIT=500 scripts/corpus/run_internal_only.sh        # smoke run first
LANGS="ar ml" scripts/corpus/run_internal_only.sh    # subset
```

**Using more than one machine** — two shapes:

*Simplest: one node per language* — five nodes, each prepares and assembles
only its language (no barrier, no ordering):

```bash
# per node, following the campaign's rank map (0=zh 1=hi 2=ar 3=en 4=ml):
LANGS="zh" nohup scripts/corpus/run_internal_only.sh \
    > "$LOGS/internal_zh.out" 2>&1 &
```

*All nine nodes: split stage 1, then assemble on five.* Phase A shards the
ten internal `(lang, source)` pairs round-robin across nine ranks
(`--node-rank`/`--num-nodes`), which can roughly halve the lane of a single
dominating source (e.g. `internal_sft_ar`, whose 58.6% duration-less rows
force header probes); phase B still assembles on five nodes, one language
each:

```bash
# phase A -- on EVERY node, R = that node's rank 0..8 (keep the default LANGS
# so every rank slices the same (lang, source) list -- and don't reuse a rank):
PREPARE_ONLY=1 NODE_RANK=$R NUM_NODES=9 nohup scripts/corpus/run_internal_only.sh \
    > "$LOGS/internal_prepare_node${R}.out" 2>&1 &

# barrier: all nine prepare reports must exist before phase B starts:
ls "$LOGS"/prepare_internal_ar_en_zh_hi_ml_node*.json   # expect 9 files

# phase B -- on the five assemble nodes, after the barrier:
SKIP_PREPARE=1 LANGS="zh" nohup scripts/corpus/run_internal_only.sh \
    > "$LOGS/internal_zh.out" 2>&1 &
```

Stage 2 is the ceiling either way — one ledger, one writer per language; the
spare nodes have nothing to take there. **Never point two nodes at the same
language** (and never reuse a rank in phase A): the ledgers are node-local
and cannot see each other, so both nodes would claim the same paths and
overwrite each other's batches. Report names carry language and node tags
(`prepare_internal_<langs>[_nodeR].json`, `bundle_internal_<langs>.json`), so
per-node reports never collide in the shared `$LOGS`. Phase B warns per
language when a pool source has no shards — the sign that phase A has not
finished everywhere yet.

It writes to **separate roots** (`corpus/pool_internal`,
`training_manifests/v8.0_internal`, `/scratch/corpus/ledgers_internal`) so it
cannot touch the campaign's pools, claims or batches; the ledger stays
node-local — resume on the same node or copy the `<lang>.sqlite3` files first.
Stage 3 is not needed (`--no-materialize`: internal audio is read in place),
and `--batches 0` drains, so each language gets one **partial** batch of
whatever the gates admit. Every language has exactly two internal sources,
so the script uses `--max-per-source-fraction 0.5` — at the campaign's 0.40
two sources can fill at most 80% of a batch and nothing could ever close.
Expect the diversity floor to bite **zh** (7.8k transcripts behind 153k
paths — it may come out empty *by design*); ar/en/hi/ml land partial batches.
`BUNDLE=1` additionally runs the stage-4 audit — expect exit 1 (reported,
not fatal) while any language lacks a full 100k batch; the internal-only
deliverable is the corrected manifests plus reports, not campaign bundles.

## Ship checklist

Per bundle — all automated by stage 4's audit, so **the job's exit code is
the verdict** (0 means every check below passed):

- exactly 100 000 rows per language per bundle;
- every manifest line parses as the strict canonical 4-key `Sample`;
- no duplicate `audio_filepath` within a language; zero cross-language overlap;
- zero eval leak (no shipped path in any v7.6 or q3asr-SFT `eval_*.jsonl`);
- every external clip present at 16 kHz mono FLAC under `$AUDIO_ROOT`;
- per-batch diversity floor (`distinct_text_fraction >= 0.50`) recomputed
  from the shipped texts;
- every row `duration > 0` (p50/p95/max + out-of-band counts in the report).

Then: `$OUT_DIR/MANIFEST.json` exists, `bundles: K` matches expectation, and
spot-check a row: `head -n1 $OUT_DIR/ar/train_ar_b0000_p0000.jsonl`.

## Knobs

Behavior knobs live in the staged sections of `configs/corpus.yaml`
(`prepare.jobs`, `assemble.llm_*` / `fetch_workers`, `materialize.workers`,
`bundle.langs`): the dispatcher feeds each section to its stage as defaults,
and **CLI flags still win**. Identity (language, ledger), paths and drain mode
are deliberately explicit in the drivers, so a YAML accident cannot change
*what* gets built — only how fast. The test suite enforces that every YAML
key is a flag its stage actually reads.
