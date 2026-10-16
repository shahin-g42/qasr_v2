# GQASR

GQASR is the ASR package inside the `gemma_speech` project. Install the parent project once; `gqasr`, `gqtts` and `gemma-speech` use the same PyTorch 2.11.0 / Transformers 5.19.0 environment. TTS remains a separate model.

The cluster creates `models/gqasr-base/` by loading the pinned pretrained audio encoder and ASR text decoder, then exporting GQASR configuration, tensor names, tokenizer, frontend and provenance. Its public model class is `GQASRForSpeechRecognition`; state keys begin with `gqasr_acoustic`, `gqasr_decoder`, `gqasr_bridge` or `gqasr_prompt`. Local subclasses retain the upstream computations. This is a derived initialization, not a new architecture created by renaming. Upstream Apache notices and attribution remain in the artifact; tokenizer vocabulary is preserved exactly.

The base has approximately 901.9M parameters. The acoustic encoder and text decoder are pretrained; the 0.99M alignment bridge is random and must be trained. Exporting the base does not establish recognition quality. All model downloads, exports, verification and training are performed by the cluster scripts in [the cluster runbook](../cluster/README.md). No generated model weights are included in the code.

## Training recipe

`configs/01_bridge_12node.yaml` trains the bridge for 1,000 optimizer steps with the encoder and decoder frozen. `configs/02_full_b40_12node.yaml` (the default full-stage config) starts from the bridge stage's `final/` artifact and keeps the encoder frozen and adapts the bridge, decoder and embeddings for 31,446 optimizer steps. All recipes expect 96 ranks. `configs/02_full_12node.yaml` retains batch 20 with accumulation 1 and 62,891 updates.

The configs use the **22 training and 6 evaluation manifests** from QASR's `configs/v8.1/02_full_12node.yaml`. Their paths remain cluster paths. The source SHA and detailed mapping are in `configs/source_mapping.yaml`.

| Setting | GQASR |
|---|---|
| Machines / GPUs | 12 × 8 |
| Arabic / English / Hindi / Malayalam / Chinese | 50% / 30% / 10% / 5% / 5% |
| Microbatch / accumulation per GPU | Bridge: 20 / 1; next stage: 40 / 1 |
| Effective global batch | Bridge: 1,920; next stage: 3,840 utterances per optimizer step |
| Data workers | 4 per GPU, 32 per machine |
| Precision | BF16 with FP32 parameters, TF32 enabled |
| Distributed strategy | Native PyTorch DDP |
| Audio | Mono 16 kHz, duration 0.1–35 seconds |
| Targets | At most 512 tokens; overlong targets filtered, never truncated |
| Full-stage learning rates | Bridge 2e-5; encoder frozen; decoder 1e-5; embeddings 5e-6 |

Bridge quotas are 10 Arabic, 6 English, 2 Hindi, 1 Malayalam, 1 Chinese per rank. Next-stage batch 40 doubles these quotas to 20, 12, 4, 2, 2. Sampling uses replacement with deterministic rank/update seeds, so resume does not require a corpus-sized permutation. This is not an epoch traversal.

The full stage now uses QASR's online code-switching, SpecAugment, speed perturbation, noise and codec simulation. The bridge warm-up remains unaugmented. Logging is JSONL and distributed training uses native DDP; this is not an exact reproduction of QASR's optimizer/runtime. H100 throughput and scaling still require cluster validation.


## Launch on 12 nodes

From the repository root, run rank 0 on node 051 and a unique rank 1–11 on each other node:

```bash
# Node inception-H100-hpc-051
bash scripts/run_gqasr_12node.sh 0

# Other nodes
bash scripts/run_gqasr_12node.sh YOUR_RANK
```

This starts the bridge stage using the shared gemma_speech environment and completed SQLite index; it does not rescan. All nodes must have matching code, configs and shared artifacts. The launcher checks the environment and eight BF16-capable GPUs per node, then waits for all 96 processes to rendezvous. The default master is `inception-H100-hpc-051`; override it with a second positional argument. Bridge uses port 29611 and full uses 29612; any `MASTER_PORT` override must match across nodes.

Preview without loading weights or touching GPUs with `DRY_RUN=1 bash scripts/run_gqasr_12node.sh 0`. Check the existing index with `bash scripts/run_gqasr_12node.sh 0 --preflight`. Microbatch 40 still requires memory/throughput validation on the cluster with real audio.

After the bridge finishes successfully and publishes its `final/` artifact, launch the full stage on all nodes:

```bash
bash scripts/run_gqasr_12node.sh YOUR_RANK --stage full
```

The revised next stage writes to outputs/full_b40_frozen. Its warmup/decay
steps are 417/3125, evaluation and saves occur every 1042 updates, and learning
rates are unchanged. The halved schedule preserves approximately the previous
example budget. Start from bridge/final, not an old batch-20 optimizer checkpoint.

To use the retained batch-20 recipe, set the same override on every node:

```bash
GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/02_full_12node.yaml" bash scripts/run_gqasr_12node.sh YOUR_RANK --stage full
```

It writes to outputs/full_b20_frozen, with 833 warmup steps, 6250 decay steps,
and evaluation/saves every 2083 updates. Both recipes freeze the encoder and
use the same learning rates, data index and augmentation settings.

## Data preparation

`gqasr preflight --config CONFIG --scan --verify-audio` builds one shared SQLite index before distributed training. It streams JSONL manifests rather than copying millions of rows into each rank's Python memory. Accepted audio fields are `audio_filepath`, `wav_path`, or `audio`; text is the first usable `text` or `transcript`. Relative paths use `audio_root` or the manifest directory. `/vast/...` maps to `/lustrefs/taiga/vast40/...`.

The scan uses all CPUs available to the CLI process by default. Worker processes validate audio and tokenize targets; a single ordered writer preserves deterministic sample IDs and duplicate/leakage checks. The scan reads audio headers, filters duration/target limits, rejects conflicting duplicate transcripts and train/eval path or recording-ID leakage, and checks every language has usable train and evaluation data. Identical duplicate paths within a split are counted and skipped. Blank or punctuation-only transcripts are skipped. Training preserves Arabic diacritics. Corpus WER/CER include strict and Arabic-normalized views.

Treat manifests, tokenizer and audio as immutable after scanning. The index records source hashes and tokenizer identity. Rebuilding requires a new `index_path` or explicitly removing the old index yourself; the tool does not overwrite one. Missing data and malformed rows fail with a source filename and line number.

## Artifacts and resume

`models/gqasr-base/` contains config, safetensors, tokenizer, frontend, checksums and attribution. Training needs this local artifact, not upstream model directories. Intermediate `checkpoint-NNNNNNNN/` and `final/` exports also include optimizer state, per-rank RNG state and the training config. Checkpoints use tensors/configs rather than pickled model classes.

For resume, copy the same stage config alongside the original, change **only `output_dir`** to a fresh directory, set `GQASR_CONFIG` to that file and pass `--resume /shared/path/checkpoint-NNNNNNNN`. Workers may also change; the model, batch, world-size, step budget and indexed dataset must match. Starting the full stage from bridge `final/` resets the optimizer intentionally.

The bridge stage always ends at `gqasr/outputs/bridge/final/` under the default config. If you change its output location, update the full-stage `base_model` accordingly.

## Validation scope

The package includes tests for tensor round trips, acoustic/decoder gradient checkpointing, exact multilingual sampling, data leakage, bridge-to-full training, and exact CPU checkpoint resume. Run them on the cluster with the `test` stage. `verify` additionally reloads the actual exported base offline, checks artifact hashes and GQASR names, runs a synthetic forward/backward pass and generates a few tokens. These are integration checks, not a speech-quality benchmark. Evaluate real Arabic dialects and all required languages before accepting the trained model.

## Parallel scan and progress

Stop an older serial scan with Ctrl+C and wait for it to exit. Restart on one allocated 96-core node:

```bash
bash gemma_speech/cluster/gqasr_stage.sh scan \
  --scan-workers 96 --scan-chunk-size 128 --progress-seconds 5 --scan-temp-dir /tmp
```

`--scan-workers 0` (default) detects available CPUs, respecting CPU affinity and `SLURM_CPUS_PER_TASK`. Workers are lightweight spawned processes with independent native tokenizers and single-threaded numerical libraries; they load no neural models. Work queues are bounded to two batches per worker. Results are indexed in source order regardless of worker completion order.

The database is built in node-local scratch (`TMPDIR` by default; explicitly use `/tmp` or your allocated local SSD directory). Ensure that directory has enough space for the completed corpus index. This avoids per-row SQLite writes over Lustre. Only the completed index is copied to a temporary file beside the destination and atomically published without overwriting an existing index.

Progress is emitted every five seconds, including while waiting for a worker batch: worker count, active manifest, checked/kept/skipped rows, rows/sec, elapsed time, processed manifest-byte percentage and an approximate ETA. The percentage tracks manifest bytes, not audio bytes; ETA will vary with file sizes and storage latency. Publishing the completed database also reports copy progress.

A `.scan.lock` file prevents concurrent new scans against the same index. Ctrl+C terminates workers and removes the current temporary files and lock. After an uncatchable kill or node crash, remove a stale lock only after confirming the recorded host/PID is no longer running. A failed or interrupted run never publishes an incomplete index. An existing completed index is preserved; select a fresh `index_path` to intentionally rebuild it.

## Training augmentation

`configs/02_full_12node.yaml` points to the existing QASR
`code_switch_indexes/cs_v1/assembled/index.json`. No new audio index or base-model
export is needed. Keep its frozen `_state/plan.json`, SQLite pools and indexed
audio at their original locations. Indexed clip text is frozen: correcting the
training manifests does **not** update code-switch partner transcripts. Rebuild
affected QASR pools if those source transcripts require correction.

Order and source settings:
1. Code-switch gate per example: 0.4; whole clips, original anchor first, frozen
   pair weights/segment counts/gaps/language policy from the existing index.
2. Speed perturbation: 0.85–1.15, probability 0.37. Out-of-bounds speed changes
   revert to the pre-speed waveform rather than truncating speech.
3. MUSAN noise: SNR 5–20 dB, probability 0.37. Missing configured noise files
   fail preflight rather than silently substituting synthetic noise.
4. QASR's bandpass/quantization codec **simulation**: probability 0.37.
5. SpecAugment: per-utterance per-frequency mean fill over original valid frames,
   two maximum time/frequency masks, ratios 0.05/0.15, batch gate
   0.37. Padding is preserved; masking follows the actual frontend dimensions.

The packaged runtime needs no QASR model import or second environment. Code-switch
candidates are retokenized using GQASR and checked against the complete target
(language header + transcript + EOS); unsuccessful budget attempts retain the
original example. Source pool selection still reflects the original index's
eligibility rules. Source language quotas remain anchor quotas; mixed-label
language assignment follows the index policy.

Evaluation never receives augmentation. Per-update/rank/example seeds make
augmentation reproducible across worker counts and resume points. Metrics include
rank-0 last-microbatch gate/fallback and waveform/SpecAugment change information;
these are diagnostics, not globally aggregated application rates.

After copying corrected manifests back to the cluster, **rebuild the GQASR
manifest SQLite index** (it stores transcript copies). Select a new `index_path`
in BOTH stage configs, or move the old index aside while no scan/training is
running. Then run:

```bash
bash gemma_speech/cluster/gqasr_stage.sh test -q
bash gemma_speech/cluster/gqasr_stage.sh scan
bash gemma_speech/cluster/gqasr_stage.sh preflight
bash gemma_speech/cluster/gqasr_stage.sh augmentation-smoke --batches 50
# Launch bridge through the existing 12-node scheduler/torchrun setup.
# Once bridge/final exists, validate the full-stage augmentation assets:
GQASR_CONFIG=gemma_speech/gqasr/configs/02_full_12node.yaml bash gemma_speech/cluster/gqasr_stage.sh preflight
# Launch full through the same 12-node setup after successful preflight.
```

The Malayalam corrected manifest still contains flagged damaged transcripts;
resolve or explicitly quarantine them before treating it as clean training data.

### Malayalam quarantine (ml_training_v2)

Both stage configs use the completed cluster index `full_v81.sqlite` and the
original manifest path `asr_cleaned_manifests/full_v1/ml/data/train_ml_itn_punct.jsonl`.
The cluster's successful scan confirms that this file was replaced with the
filtered Malayalam manifest (SHA-256
`1570b758575849e161408db1e0e0ddcc74553cb96947eebbb4988473d3aaea26`).
Do not rescan this completed index merely to update the augmentation code.

The exclusion file must also be available at
`asr_cleaned_manifests/ml_training_v2/quarantine/ml_damaged_transcripts.jsonl`.
Copy it from the local partition output if it is not already on the cluster.
It is **not** a training manifest. The code-switch runtime reads it only as an
exclusion list, matched by audio basename so mount aliases cannot bypass it.
Missing exclusion data fails rather than silently reintroducing quarantined
recordings. Other heuristic review flags remain in the training manifest.

### Rescan Malayalam only

Once an earlier full scan has completed, replace only Malayalam train/evaluation
records while retaining every other language. The destination defaults to the
selected config's `index_path` and MUST NOT already exist:

```bash
bash gemma_speech/cluster/gqasr_stage.sh rescan-ml \
  --source-index gemma_speech/gqasr/indexes/full_v81.sqlite \
  --output gemma_speech/gqasr/indexes/full_v81_ml_rescanned.sqlite \
  --workers 0
```

This scans the Malayalam manifests currently configured in the bridge YAML,
including audio-header and token-length validation. It checks for leakage against
retained records, copies the old SQLite database to local scratch, replaces the
Malayalam rows, and publishes a new index atomically. No existing index is edited.
`--workers 0` uses available CPUs; `--temp-dir /local/scratch` selects local scratch.
Progress is printed during scanning, copying and publication. Space is required
for the copied full index plus the Malayalam subset and the destination index.

If the config's destination already exists, use `--output /absolute/new.sqlite`
and then set BOTH bridge/full `index_path` values to that new path before training.
Supply the actual completed index as `--source-index`; never use a scan in progress.
Other languages' files are assumed unchanged and are NOT read or revalidated.
Tokenizer, duration limits, path rules, or other-language manifest changes require
a full scan. The original full-scan skip totals are replaced by explicit rescan
metadata because the old totals were not recorded per language.

Run `preflight` and `augmentation-smoke` after the rescan. If the scan currently
running already uses `ml_training_v2`, another Malayalam rescan is unnecessary
unless the manifest changes again. This prepares a new training run; changing
the index is intentionally incompatible with exact resume of an old-data run.

### Scan storage exhaustion

`database or disk is full` is a storage/quota failure, not a bad transcript.
The scan builds its SQLite database in `--scan-temp-dir` (system temporary
storage by default), and SQLite indexes consume additional space. Inspect the
`scan_scratch` log entry for the actual directory. Check both filesystem space
and user/project quota on that volume; `df` alone does not show Lustre quotas.
The current scanner discards incomplete databases and cannot resume a failed
partial scan. A language rescan requires a previously completed source index.

Choose a scratch volume with sufficient capacity, preferably local NVMe for
random-write performance. Shared Lustre scratch also works but can be slower:

```bash
GQASR_SCAN_SCRATCH=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/gqasr_scan_scratch
mkdir -p "$GQASR_SCAN_SCRATCH"
df -h "$GQASR_SCAN_SCRATCH" gemma_speech/gqasr/indexes
# Check your user/project quota using the cluster's quota tooling as well.
TMPDIR="$GQASR_SCAN_SCRATCH" SQLITE_TMPDIR="$GQASR_SCAN_SCRATCH" \
  bash gemma_speech/cluster/gqasr_stage.sh scan --scan-temp-dir "$GQASR_SCAN_SCRATCH"
```

Progress now reports scratch/output free space and working SQLite size. These
are filesystem values, not quota guarantees. On filesystems supporting hard
links, scratch on the same filesystem as the output is published without a
second full database copy. Different filesystems still require space for both
the working index and the destination copy. No existing index is overwritten.
