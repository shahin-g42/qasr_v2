# Cluster workflow

Run these commands **on the cluster**, from the repository root after copying/pulling the code. All model execution belongs on your machines; generated weights, indexes, logs and outputs are ignored by Git. `gqasr` and `gqtts` live inside `gemma_speech` and install together into one environment.

Use shared storage at the same absolute path on all 12 nodes for code, model artifacts, indexes and outputs. Use allocated compute resources for verification, tests and training; do not run heavy model work on a login node. Downloads require network access; later stages set `HF_HUB_OFFLINE=1`.

## 1. Install the common environment

```bash
bash gemma_speech/cluster/setup_env.sh
```

The script installs the CUDA 12.8 wheel of PyTorch 2.11.0 and Transformers 5.19.0. Your CUDA 12.9 toolkit can coexist with that bundled runtime; the NVIDIA driver must support the wheel. `GEMMA_SPEECH_ENV` optionally selects a different environment directory. Use the same environment on all nodes, or reproduce `gemma_speech/logs/environment.lock.txt` on each node. Do not install the upstream `qwen-tts` SDK into this environment: the compatible private port is included.

On every allocated GPU node, check `nvidia-smi` and run:

```bash
bash gemma_speech/cluster/gqasr_stage.sh doctor
```

This requires eight visible CUDA GPUs with BF16 support. Set `NPROC_PER_NODE` only when deliberately testing a smaller allocation; the training config's `expected_world_size` must also match.

## 2. Download and initialize GQASR once

```bash
bash gemma_speech/cluster/gqasr_stage.sh download
bash gemma_speech/cluster/gqasr_stage.sh export
bash gemma_speech/cluster/gqasr_stage.sh verify
bash gemma_speech/cluster/gqasr_stage.sh test -q
```

`download` fetches pinned upstream revisions into `gemma_speech/models/`. `export` creates `gemma_speech/gqasr/models/gqasr-base/` with pretrained core weights and a random bridge. It refuses to overwrite an existing artifact. `verify` is an offline synthetic integration check. The GQASR `test` stage runs only the GQASR tests with tiny fixtures and needs no pretrained download. TTS tests have their own stage.

The public core names and tensor prefixes are GQASR. Provenance and license notices retain the upstream source names. The export is a derived base requiring adaptation, not a trained recognizer.

## 3. Validate the actual manifests once

```bash
bash gemma_speech/cluster/gqasr_stage.sh preflight
bash gemma_speech/cluster/gqasr_stage.sh scan
```

The default configs copy 22 training and 6 evaluation paths from QASR v8.1. Update those paths if the cluster mount differs. `preflight` reports readiness; `scan` reads the manifests and audio headers, filters unusable samples and builds the shared index. A successful scan reports `ready_to_train: true`. It uses all allocated CPUs by default and reports progress every five seconds. For your 96-core node, use `gqasr_stage.sh scan --scan-workers 96 --scan-temp-dir /tmp`. SQLite is built on local scratch before atomic publication to the shared index path. The full corpus still requires opening every audio header, so storage throughput affects runtime. Do not scan independently from all ranks or mutate the corpus afterward.

## 4. Train the bridge on 96 GPUs

With Slurm, submit **from the repository root**:

```bash
sbatch gemma_speech/cluster/submit_12node.sh gqasr bridge
```

Without Slurm, use the rank-based launcher from the repository root:

```bash
# Master inception-H100-hpc-051
bash scripts/run_gqasr_12node.sh 0

# Other nodes: unique ranks 1 through 11
bash scripts/run_gqasr_12node.sh YOUR_RANK
```

This uses the unified gemma_speech environment and existing completed index without rescanning. Preview with `DRY_RUN=1`; use `--preflight` for an index/config check. An optional second positional argument overrides the master hostname. Bridge defaults to port 29611; full defaults to 29612.

If needed, configure `NCCL_SOCKET_IFNAME` and other fabric settings for your network. The scripts do not guess interface names. All nodes must use matching rank/world-size and rendezvous settings.

## 5. Train the full ASR model

Wait for a successful bridge run and its `gemma_speech/gqasr/outputs/bridge/final/` artifact, then:

```bash
sbatch gemma_speech/cluster/submit_12node.sh gqasr full
```

For manual launch, run `bash scripts/run_gqasr_12node.sh YOUR_RANK --stage full` on every node using the same ranks. The full stage uses the pretrained bridge-stage artifact and resets the optimizer. Defaults are microbatch 40 × accumulation 1 × 96 GPUs, yielding 3,840 utterances per optimizer update with the source's exact language quotas. The full stage enables QASR online code-switching and acoustic augmentations; bridge warm-up is unaugmented. Validate microbatch 40 memory usage on real cluster audio; see `gqasr/configs/source_mapping.yaml`.

## 6. Resume or transcribe

For resume, copy the stage config **within its existing configs directory** and edit only `output_dir` to a new path. Keep the other settings unchanged. For example:

```bash
export GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/02_full_resume.yaml"
sbatch gemma_speech/cluster/submit_12node.sh gqasr full \
  --resume /shared/path/to/checkpoint-00002083
```

This config file must be created and edited first. Resume uses the saved optimizer, step and rank RNG states. Output directories are never silently overwritten. For inference after training:

```bash
bash gemma_speech/cluster/gqasr_stage.sh transcribe \
  --model "$PWD/gemma_speech/gqasr/outputs/full_b40_frozen/final" \
  --audio /shared/path/sample.wav --device cuda
```

## GQTTS stages in the same environment

GQTTS contains the existing speaker-adapter and talker training implementation. It retains the Qwen codec. The ASR training manifest does not supply the explicit speaker identities and voice requirements needed to automatically prepare a TTS corpus. Its configs remain templates; prepare paired multilingual TTS manifests with speaker IDs and disjoint validation recordings/speakers before running:

```bash
bash gemma_speech/cluster/gqtts_stage.sh download
bash gemma_speech/cluster/gqtts_stage.sh cache \
  --manifest /shared/path/tts-train.jsonl \
  --output "$PWD/gemma_speech/data/speaker-train-cache" --include-codes
bash gemma_speech/cluster/gqtts_stage.sh cache \
  --manifest /shared/path/tts-validation.jsonl \
  --output "$PWD/gemma_speech/data/speaker-val-cache" --include-codes
sbatch gemma_speech/cluster/submit_12node.sh gqtts speaker-train
# After speaker adaptation completes:
sbatch gemma_speech/cluster/submit_12node.sh gqtts train
```

Cache creation supports `--shard` / `--shards`; give each shard a separate output directory and assemble cache manifests before training. Default cache commands run one process on CUDA. `GQTTS_CONFIG` selects a custom stage config. For inference, `gqtts_stage.sh synthesize` accepts `--checkpoint`, `--audio`, `--text`, `--language`, `--output` and optional `--talker-checkpoint`; run `gqtts synthesize --help` for details. The standalone GQTTS base export is described below. Existing TTS training commands still use the teacher-based initialization workflow until the later TTS adaptation work integrates the new base.

## Logs to send back

Stage console logs are saved under `gemma_speech/logs/`; GQASR verification reports are under `gemma_speech/gqasr/verification/`. Training writes `metrics.jsonl`, prediction files and checkpoints under the configured output directory. Send the stage log and any traceback first; for training include the initial parameter/global-batch report, recent metrics and node/GPU diagnostics. Slurm also captures scheduler output in its normal job log.

The scripts and package structure have been checked locally. Cluster data access, real-audio quality, H100 memory consumption, NCCL connectivity and 96-GPU scaling still require your runs.

## Prepare both bases, then fine-tune ASR first

Keep the already exported and verified `gemma_speech/gqasr/models/gqasr-base/`. Do not rerun its export into the same directory. On one allocated cluster node:

```bash
bash gemma_speech/cluster/gqtts_stage.sh download
bash gemma_speech/cluster/gqtts_stage.sh export
bash gemma_speech/cluster/gqtts_stage.sh verify --device cuda
bash gemma_speech/cluster/gqtts_stage.sh test -q
```

GQTTS writes its independent artifact to `gemma_speech/gqtts/models/gqtts-base/`; existing GQASR weights are untouched. Both replacement adapters are initially untrained. Keep source TTS assets available for later teacher caching. Do not start TTS fine-tuning yet.

After both bases verify, prepare ASR data and start bridge training:

```bash
bash gemma_speech/cluster/gqasr_stage.sh test -q
bash gemma_speech/cluster/gqasr_stage.sh preflight
bash gemma_speech/cluster/gqasr_stage.sh scan
sbatch gemma_speech/cluster/submit_12node.sh gqasr bridge
```

If the index was already built successfully, use `preflight` and skip `scan`. Start the full ASR stage only after bridge training completes. For machines without Slurm, use the manual per-node launch instructions above. These model operations run on the cluster; local validation of newly added export code is limited to source checks and test collection.


## Revised next-stage recipe (October 16)

The full launcher now starts a frozen-encoder stage from bridge/final, writing to
outputs/full_b40_frozen. Only the bridge, decoder and embeddings train.
The default config is gqasr/configs/02_full_b40_12node.yaml.
Batch 40 with accumulation 1 across 96 GPUs gives global batch 3840.
The retained 02_full_12node.yaml uses batch 20, accumulation 1, global batch 1920
and writes to outputs/full_b20_frozen. Select it on every node with
`GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/02_full_12node.yaml"`
before the launcher command. Both recipes keep the encoder frozen.
Per-rank quotas are ar=20, en=12, hi=4, ml=2, zh=2.

The schedule preserves approximately the previous example budget: 31446 updates,
417 warmup, 3125 decay, evaluation/save every 1042 updates, logging every 5.
Ceiling rounding adds 1920 examples to the total budget. Learning rates are unchanged.
The completed bridge recipe is unchanged. Do not resume an old batch-20 optimizer
checkpoint into this recipe; start from the bridge final artifact.

SpecAugment now fills masked regions with per-frequency means from each utterance's
original valid frames. Padding is untouched. Gradient clipping uses standard
PyTorch norms again; nonfinite-gradient diagnostics remain. These changes do not
require a rescan or base-model re-export. Online streaming additions are not part
of this change.

## W&B curves for GQASR

All three recipes enable `report_to: wandb`, `wandb_project: qasr`, and a
stage/batch-specific `run_name`. Install the tracking extra in the shared environment
before a new launch: `gemma_speech/.venv/bin/python -m pip install 'wandb>=0.19,<1'`.
Authenticate using `gemma_speech/.venv/bin/wandb login` (or an existing
WANDB_API_KEY environment variable; do not put credentials in YAML).
WANDB_ENTITY and WANDB_DIR retain the SDK's standard environment behavior.
WANDB_MODE=offline records locally for later `wandb sync`; WANDB_MODE=disabled
turns tracking off. Only global rank zero starts a run.

Training loss is explicitly labeled rank0_last_microbatch_loss, not global loss.
New launches also record rank-0 pre-clipping gradient norm and each optimizer
group's learning rate. Validation includes strict and Arabic-normalized WER/CER
for each language/dialect, as available. WER/CER are ratios, not percentages.
Throughput is measured between logged updates and includes any intervening
evaluation/checkpoint delays. No audio, transcripts, or model artifacts are uploaded.

For the ALREADY RUNNING batch-40 job, do not restart training. After pulling this
code and installing/authenticating W&B, run this once in a separate terminal on
one node (keep it open, e.g. in tmux):

```bash
gemma_speech/.venv/bin/python -m gqasr.tracking \
  --output gemma_speech/gqasr/outputs/full_b40_frozen \
  --project qasr --name gqasr-full-b40 --follow
```

The follower imports complete existing metrics.jsonl records and tails new ones.
It cannot recover learning rates/gradient norms absent from the old JSONL.
Omit --follow to import a completed run. An exclusive lock prevents duplicate
writers. For online follower restarts, the saved run ID and last submitted step
avoid replaying old records. SDK submission is not a server durability guarantee;
the JSONL remains authoritative. Offline segments should each be synced;
offline SDK resume does not guarantee merging segments into one run.

A training checkpoint resumed into a fresh output directory gets a new W&B run
with absolute optimizer steps (avoids overlapping history after checkpoint rollback).
Logging-only config changes are allowed on resume; training/data changes remain
rejected. A W&B error warns and disables tracking for that process, without
aborting distributed training or stopping JSONL logging. The follower exits on
tracking errors so it can be corrected and restarted. Pulling code does not add
new telemetry to an already-running Python process.

## Readable training console

New launches print compact training summaries, code-switch outcome/pair counts,
a full validation table (all metric groups, counts, WER/CER and error totals),
and explicit validation-start/checkpoint-saved messages. WER/CER are displayed
as percentages in the console; JSONL and W&B retain ratios. Full per-example
augmentation details remain in metrics.jsonl. Set GQASR_CONSOLE_FORMAT=json to
retain the previous raw console format.

For an already-running job, view its existing and future records in a separate
terminal without restarting training:

```bash
gemma_speech/.venv/bin/python -m gqasr.console \
  --metrics gemma_speech/gqasr/outputs/full_b40_frozen/metrics.jsonl \
  --from-step 1042 --follow
```

The viewer is read-only and can run alongside the W&B follower. It shows only
events recorded in metrics.jsonl; it cannot reconstruct checkpoint completion
or validation-start messages from older runs. Omit --follow for a one-time report.

## Original Qwen3-ASR-1.7B Chinese comparison

Run on a spare cluster GPU, with the shared filesystem mounted at the same paths.
Uses Qwen's official `Qwen/Qwen3-ASR-1.7B-hf` Transformers-format checkpoint,
including its original audio encoder and decoder. Uses the existing Transformers
5.19 environment; do not install qwen-asr or downgrade Transformers.

```bash
git pull origin main
bash gemma_speech/cluster/qwen_asr_baseline.sh download
GPU_IDS=0 EVAL_STEP=7294 bash gemma_speech/cluster/qwen_asr_baseline.sh run
```

Download pins the resolved HF revision and records it locally. Inference is offline.
Default batch size is 1 to exclude batch-padding effects. Both automatic language
detection (the primary comparison) and forced Chinese (diagnostic) run with greedy
decoding and a 768-token limit. References are never supplied to the model.
Optional run arguments: `--batch-size 2`, `--modes auto`, `--device cpu`.
Select another saved evaluation with EVAL_STEP. GQASR_RUN_DIR, QWEN_BASELINE_MODEL,
GEMMA_SPEECH_ENV and BASELINE_OUTPUT can override paths.

Exactly the Chinese IDs from the chosen saved predictions are looked up in the
SQLite index, and reference text must match before loading the baseline.
No rescan, training or modification of model weights is performed.
Outputs go to a new gqasr/verification/qwen17b-zh-STEP-TIMESTAMP directory:

- samples.jsonl: IDs, audio paths, references and existing GQASR hypotheses.
- predictions-auto.jsonl / predictions-chinese.jsonl: baseline transcription,
  language, raw generated text, generation-limit flag and original comparison data.
- scores.json: GQASR and baseline strict WER/CER, punctuation-insensitive scores,
  no-Han output counts and generation-limit counts. Chinese WER uses whitespace
  splitting and is diagnostic; prioritize CER. No ITN normalization is applied.
- provenance.json: source revision when downloaded by the script, input/config
  hashes, library versions and inference settings.

Progress prints after each batch. Existing output directories are never overwritten.
After interruption, use a new output directory for a complete rerun.
A model agreeing with a reference is corroboration, not proof of label accuracy;
disagreements still need listening. The pretrained-model GPU run is cluster-side;
local tests cover sample matching, reference guards and scoring.
