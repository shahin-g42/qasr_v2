# GQASR encoder continuation

## Fresh optimizer round from checkpoint 7,500

Use `scripts/start_gqasr_encoder_4node.sh` for the requested weights-only restart.
This is a full **31,446 new updates**, starting at zero. It preserves the trained
model weights, tokenizer and frontend, but loads no `training_state.pt` and starts
new AdamW moments, RNG streams, LR warmup and W&B run.

The export is a standalone model artifact, not a declaration that the interrupted
training round finished or that speech quality was validated. It copies weights
without dtype conversion, verifies the copy's SHA-256 and finite weights on CPU,
retains attribution/licenses, and records the source checkpoint in
`checkpoint_export.json`. Progress is printed while copying and validating.
Only one tensor is materialized at a time; no GPU or full model is constructed.
Intentional scalar infinite clipping bounds are preserved and recorded separately;
nonfinite learned weights or invalid clipping bounds fail the export.
The source checkpoint is never overwritten. An existing export is refused.

| Setting | Fresh round |
|---|---|
| Config | `04_encoder12_b120_fresh_4node.yaml` |
| Source checkpoint | `outputs/encoder12_b80/checkpoint-00007500` |
| Exported model | `models/gqasr-encoder12-step7500` |
| New output | `outputs/encoder12_b120_4node_fresh7500` |
| Nodes / GPUs | 4 / 32; master 020, default port 29614 |
| Batch / accumulation / global batch | 120 / 1 / 3,840 |
| DataLoader workers / prefetch | 8 per GPU / 2 |
| Trainable model | 12 encoder blocks + projection, decoder, bridge and embeddings; frontend frozen |
| Base LR: encoder / bridge / decoder / embeddings | 1e-6 / 5e-6 / 2e-6 / 1e-6 |
| Schedule | 1,000-update warmup, then cosine decay over 30,446 updates to 10% |
| Validation / checkpoints | Initial validation, then every 500 updates and at completion |
| Sampling | New training seed 1000043; evaluation seed stays 42 |
| Code-switching | Disabled; waveform augmentation and SpecAugment retained |
| W&B | Project `gqasr`, new run `gqasr-encoder12-b120-4node-fresh7500` |

Run the export **once** in the shared cluster checkout:

```bash
git pull origin main
export GEMMA_SPEECH_ENV="$PWD/gemma_speech/.venv2"
bash scripts/start_gqasr_encoder_4node.sh export
```

Then on each of the four nodes, use the same checkout and environment:

```bash
export GEMMA_SPEECH_ENV="$PWD/gemma_speech/.venv2"
export MASTER_ADDR=inception-H100-hpc-020
unset MASTER_PORT GQASR_FRESH_CONFIG
# 020 runs rank 0; the other three nodes run unique ranks 1, 2, 3:
bash scripts/start_gqasr_encoder_4node.sh YOUR_RANK
```

The fresh launcher deliberately ignores inherited `GQASR_CONFIG` and
`GQASR_RESUME_CHECKPOINT`; it never forwards `--resume`. `GQASR_FRESH_CONFIG`
selects a custom fresh config, and `GQASR_EXPORT_CHECKPOINT` selects another export
source. The export always goes to that config's `base_model` directory.
The existing SQLite index is reused; no scan or upstream download is needed.
The previous resume launchers remain available with their original semantics.

This changes optimizer initialization, not DataLoader startup or systemd IPC
policy. On October 20, node 020 was verified to have `RemoveIPC=true`, and missing
DataLoader semaphores/NCCL shared-memory files followed a diagnostic SSH session
ending. The cluster administrator should address automatic IPC cleanup before
another run; a fresh optimizer does not protect running jobs against it.
[NVIDIA documents this failure and the `RemoveIPC=no` remedy](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2273/user-guide/docs/troubleshooting.html#systemd).
Batch-120 GPU memory and throughput still need validation on the cluster.

## Default recovery: four nodes, batch 120

`scripts/resume_gqasr_encoder.sh` now defaults to four nodes (32 GPUs), master
`inception-H100-hpc-020`, and `03_encoder12_b120_resume_4node.yaml`. It resumes
`encoder12_b80/checkpoint-00007500` into the fresh
`encoder12_b120_4node_resume7500` output directory. The explicit
`resume_gqasr_encoder_12node.sh` remains available for the earlier topology.

| Setting | Four-node recovery |
|---|---|
| GPUs / ranks | 32; four nodes with eight GPUs each |
| Batch / accumulation | 120 per GPU / 1 |
| Global batch | 3,840, same as 40 × 96 GPUs |
| Language quota per GPU/update | ar=60, en=36, hi=12, ml=6, zh=6 |
| Workers / prefetch | 8 per GPU / 2; 64 workers per node, 256 total |
| Learning rates and schedule | Unchanged from the batch-40 recovery plan |
| Next / final update | 7,501 / 31,446; 23,946 remain |
| Validation / saving | Every 500 updates and at completion |
| Code-switching / gradient checkpointing | Disabled / enabled |

The checkpoint itself used global batch 7,680. This plan therefore has half
that checkpoint's global batch, while matching the subsequently requested
batch-40 plan. No LR multiplication or schedule restart is applied. The remaining
sampled-anchor budget is 91,952,640; this is sampling with replacement, not epochs.
AdamW moments and update count are restored. `--allow-world-size-change` explicitly
permits the new topology; all saved rank RNG states must still be present.
New ranks receive deterministic RNG seeds derived from the training seed, saved
update and new rank/world size. Sampling and RNG streams change with topology;
`resume.json` records this. Index, trainable layers, augmentation, language
proportions and LR/schedule compatibility checks remain enforced.

Batch 120 has not been measured on the H100s. Checkpointing stays enabled.
Nominal prefetch capacity is 1,920 examples per GPU; actual shared-memory usage
depends on lengths. Fewer nodes reduce inter-node participants but do not guarantee
higher throughput, and larger padded batches can increase peak memory.

Run on each of the **four** selected nodes only:

```bash
git pull origin main
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT GQASR_LAUNCH_NNODES
export GEMMA_SPEECH_ENV="$PWD/gemma_speech/.venv2"
export MASTER_ADDR=inception-H100-hpc-020
```

On 020: `bash scripts/resume_gqasr_encoder.sh 0`.
On the other three nodes: the same command with unique ranks `1`, `2`, `3`.
The generic recovery wrapper overrides an inherited `NNODES` with four. All nodes
must use the same checkout and environment. An optional CPU checkpoint check is
`bash scripts/resume_gqasr_encoder.sh check`. Setup and code updates never launch
training automatically. No download, export or index rebuild is required.

## Previous recovery: batch 40 on 12 nodes

Use `scripts/resume_gqasr_encoder_12node.sh` for the interrupted encoder run.
It selects `configs/03_encoder12_b40_resume_12node.yaml`, restores
`outputs/encoder12_b80/checkpoint-00007500`, and writes to the fresh
`outputs/encoder12_b40_resume7500_master020` directory. Original checkpoints stay intact.
The existing environment, exported model and SQLite index are reused.

| Setting | Recovery |
|---|---|
| Per-GPU batch / accumulation | 40 / 1 |
| Global batch | 3,840 anchor utterances across 96 GPUs |
| DataLoader workers / prefetch | 8 per GPU / 2 batches per worker |
| Worker startup | Standard PyTorch `spawn` |
| DataLoader timeout | Default 0; no added batch-delivery timeout |
| Encoder | All 12 blocks and output projection trainable; convolutional frontend frozen |
| Decoder, bridge, tied embeddings | Trainable; 28 decoder layers |
| Code-switch augmentation | Disabled; waveform augmentation and SpecAugment unchanged |
| Encoder / bridge LR | Existing scheduled rates, with base rates 1e-6 / 5e-6 |
| Decoder / embedding LR | Existing scheduled rates, with base rates 2e-6 / 1e-6 |
| Optimizer / schedule | Restored AdamW moments and update counter; no restart of warmup |
| Updates | Next 7,501; stop at 31,446; 23,946 remain |
| Validation / saving | Every 500 updates and at completion |
| Tracking | W&B project `gqasr`, run `gqasr-encoder12-b40-resume7500-master020` |

The global batch is **half** the interrupted batch-80 run. The remaining
sampled-example budget is therefore halved at the same 23,946 remaining updates;
this is not an unchanged-effective-batch restart. Learning rates and the schedule
remain in optimizer-update units. Per-rank language quotas are ar=20, en=12,
hi=4, ml=2, zh=2, preserving the original proportions. Chinese remains included.
Sampling is with replacement, not an exact epoch traversal. Lower batch size
does not by itself guarantee better throughput or quality.

The recovery wrapper passes `--allow-batch-change` and `--allow-world-size-change`.
These permit batch / accumulation changes with proportional language quotas and
an explicitly configured world size; the legacy config still uses 96 ranks.
Changes to the index, tokenizer, trainable layers, seeds, augmentations and
schedule remain rejected. It records old/new global batches and all recipe changes in
`resume.json`. Optimizer state and per-rank RNG are restored. Different grouping
and quotas change subsequent draws, token weighting and augmentation seeds;
this is a continuation, not an identical replay of the batch-80 run.

## Simplified startup and retained metrics

Temporary CPU rendezvous readiness gates, prefetched warmup buffers, automatic
Python stack dumps, worker-ready logs and forkserver import preloading have been
removed. The wrapper no longer enables NCCL flight-recorder dumps automatically.
No standalone audit, checkpoint tensor scan, data scan or backward test is run by
the recovery command. The launcher's existing environment/GPU doctor remains;
checkpoint/index compatibility and finite-loss/gradient checks remain in training.
NCCL retains explicit LOCAL_RANK device binding and its standard timeout.

There are 64 DataLoader workers per 96-core node. Each worker prefetches two
batches: nominal queue capacity is 640 anchor examples per GPU. Audio lengths
and feature shapes determine actual RAM/shared-memory usage. The wrapper sets
OpenMP, BLAS and Numba thread counts to one to avoid nested thread oversubscription;
`GQASR_CPU_THREADS` overrides that limit. No custom worker-start method is used.

Global loss averaging and W&B are retained. The October 20 captured collective
history placed the stall at the first loss-finiteness reduction, before the first
average-loss reduction or W&B publication. This does not prove the cluster is
healthy, but it does not implicate loss logging as the initiating cause.

`train/loss` is the mean of detached, unscaled microbatch losses across all ranks
since the preceding log or resume. Every rank takes part in the same two-value
SUM reduction after optimizer updates, at logging boundaries. Each microbatch
loss is already a supervised-token mean; this is an equal-microbatch mean,
matching DDP/accumulation weighting, not a token-weighted corpus metric.
Only rank zero writes `metrics.jsonl`, console summaries and W&B. W&B failures
disable remote publication while retaining local metrics. A real two-rank CPU
regression checks collective ordering, arithmetic and rank-zero publication
with both accumulation 1 and accumulation 4. It does not validate H100/NCCL or
shared-storage health.

`train/rank0_last_microbatch_loss` remains a separately named diagnostic. LRs
appear only under `lr/bridge`, `lr/acoustic`, `lr/decoder` and `lr/embedding`.
Normal progress, validation, checkpoint and throughput summaries remain. The
original nonfinite-loss and gradient clipping protections are not disabled.
SpecAugment still runs before the encoder with per-frequency mean fill over
valid frames. No model architecture, audio preprocessing or dependency changes
are made by this recovery.

## Launch when ready

From the repository root, pull on the shared checkout. On master 020:

```bash
git pull origin main
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT
export MASTER_ADDR=inception-H100-hpc-020
bash scripts/resume_gqasr_encoder_12node.sh 0
```

On each remaining node, using its unique rank 1 through 11:

```bash
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT
export MASTER_ADDR=inception-H100-hpc-020
bash scripts/resume_gqasr_encoder_12node.sh YOUR_RANK
```

Node 020 must use rank 0. If node 051 stays in the allocation, give it the
previous nonzero rank of node 020; keep all ranks 0–11 unique.
The fresh output directory preserves the failed attempts.

All nodes must use the same code/configuration. The port is 29613; all eight GPUs
are visible by default. `GPU_IDS`, `MASTER_ADDR` and `MASTER_PORT` can override
allocation-specific settings. Existing W&B credentials are reused;
`WANDB_MODE=offline` is supported. No fresh environment, download, export or
index rebuild is needed. The code update itself does not launch any jobs.

The previously completed checkpoint check need not be repeated. If desired,
`bash scripts/resume_gqasr_encoder_12node.sh check` remains an explicit CPU-only
readability/finite-tensor check. It is not automatically run and is not a
cluster health test. After another interrupted attempt, use a fresh `output_dir`
and the appropriate `GQASR_RESUME_CHECKPOINT` / `GQASR_CONFIG` overrides.

Console files default to `/tmp/gqasr-$USER-logs` on each node; set `GQASR_LOG_DIR`
to another node-local directory if desired. Checkpoints, predictions,
`metrics.jsonl` and W&B state remain under the shared output directory.
Removing diagnostic machinery does not repair the previously reported RDMA
fatal event or Lustre I/O errors, nor prove the startup stall is resolved.

## Other retained recipes

The generic `run_gqasr_12node.sh --stage encoder` command still selects the
original fresh batch-80 stage (`03_encoder12_b80_12node.yaml`), initialized from
`outputs/full_b40_frozen/final` with a new optimizer. Use the **recovery wrapper
above** to continue checkpoint 7,500 at batch 40.

`03_encoder12_b40_12node.yaml` is the historical fresh-stage batch-40 recipe with
code-switch augmentation enabled. It is not the recovery configuration.
`03_encoder12_b20_acc4_12node.yaml` is retained as the previous recovery recipe.
The older `--allow-microbatch-change` option continues to require identical
effective batch and identical quotas. Neither option permits arbitrary recipe
drift during optimizer-state restoration.

## Observational rank startup status

Before GPU workers exist, each node prints `preflight_start` and
`launcher_preflight_complete`, then `torchrun_rendezvous` with the node rank,
master address and expected node count. `LOGLEVEL` defaults to `INFO` for
PyTorch's own rendezvous/start-worker messages (an explicit `LOGLEVEL` overrides
it). Static rendezvous launches GPU workers only after all node agents join;
there are no GPU rank counts during that earlier wait.

Each launched Python worker prints `rank_worker_started` before importing the
trainer, so a slow library import can be distinguished from rendezvous.
Each rank prints `rank_startup` with its rank, hostname and phase. Phases cover
joining, index/processor loading, model loading, optimizer restore, the initial
barrier, DDP initialization, augmentation setup, worker startup, waiting for the
first batch, forward, loss-finiteness collective, backward, clipping and the
first completed optimizer update. A phase names the operation being entered;
it does not prove completion of asynchronous GPU work.

After ordinary process-group initialization, a background observer publishes
small records through the existing CPU rendezvous store. Rank zero prints
`rank_startup_summary` every 15 seconds: reported/joined counts, ranks not yet
reported, first-batch/first-update counts, phase counts and pending rank/host
pairs. Before initialization completes, only individual joining messages are
available. Records are last-observed stages, not liveness heartbeats. Missing
reports do not prove failed ranks. Summaries stop once all first updates are
reported; per-rank phase logs stop after that rank's first update.

This adds no collective, training gate, CUDA synchronization, warmup batch or
stack dump. Store operations run in a daemon thread; observation failures are
reported without aborting training. Set `GQASR_RANK_STATUS=0` to disable these
messages. No training jobs are started by installing this update.

## Separate training-only `.venv2`

Run once in the shared repository (no sudo; not once per node):

```bash
bash gemma_speech/cluster/setup_training_env2.sh
```

This creates `gemma_speech/.venv2` from standard CPython 3.11–3.14. Set
`PYTHON=/path/to/python3.13` to select another installed interpreter. It refuses
to overwrite an existing environment; `GQASR_TRAIN_ENV` selects a fresh path.
The original `.venv` and its pinned unified ASR/TTS dependencies remain untouched.

The installer resolves newest stable `torch>=2.13,<3` from the official **cu129**
index, then constrains that exact build while installing `transformers>=5.20.2,<6`
and the remaining compatible training dependencies. It installs only the
`gqasr-training` project, not the combined `gemma-speech` distribution. This avoids
conflicting Torch/Transformers pins and omits TTS, datasets/PyArrow benchmark
support, torchvision, torchaudio, DeepSpeed and Accelerate. It retains W&B and
the audio/frontend dependencies used by training. Do not subsequently install
`gemma_speech[dev]` into this environment: that is the older combined profile.

CUDA 12.9 binaries use CUDA minor-version compatibility on driver 570.
Newer PTX or driver-dependent features may require a driver update; cluster
validation is still needed. The installer does not modify the host driver or
install a system CUDA toolkit. NCCL is supplied by the selected
Torch build. No Torch nightlies or CUDA 13 builds are selected. See the
[official PyTorch build matrix](https://pytorch.org/get-started/previous-versions/).
The exact compatible dependency set can change over time; successful setup saves
`requirements.lock.txt`, `pip-inspect.json` and `gqasr-training-profile.json`
inside `.venv2`. All nodes should use this single shared installation. Do not
upgrade it while processes are running.

Setup runs `pip check` plus a small CPU audio frontend, forward/backward,
AdamW and exact model save/reload test. No production weights are downloaded or
loaded, and no GPU/NCCL job is started. Failure stops setup before its success
message. The version stamp allows the regular doctor to recognize this new
runtime; subsequent Torch/Transformers changes invalidate the stamp. This smoke
is not full saved-checkpoint or 96-GPU compatibility validation.

To select `.venv2` on **every node**, before using the existing recovery launcher:

```bash
export GEMMA_SPEECH_ENV="$PWD/gemma_speech/.venv2"
export MASTER_ADDR=inception-H100-hpc-020
```

The existing doctor then checks GPU visibility/BF16 using the selected runtime.
Switch back by unsetting `GEMMA_SPEECH_ENV`. No launcher automatically switches
environments merely because `.venv2` exists. Use a fresh output directory for
another training attempt; do not overwrite an existing failed-run directory.
