# GQASR encoder continuation on 12 nodes

## Startup correction for the update-7,500 restart

Keep the existing environment and checkpoint. The recovery wrapper still resumes
`encoder12_b80/checkpoint-00007500` with batch 20, accumulation 4 and eight data
workers per GPU. Its new output is
`encoder12_b20_acc4_resume7500_startupfix`; prior attempts stay intact.

The October 20 reports show all 96 ranks completed the weight broadcast.
Nodes 009 and 056 enqueued the first loss check at 07:46 UTC; node 051 at
07:53, node 048 at 07:55 and node 020 at 07:56. Seven other nodes had not
enqueued it when the first ten-minute timeout fired. This is a startup
imbalance; the reports do not identify the instruction delaying those nodes.

The training loop now warms the data workers before wrapping the model in DDP.
It retains the first batches in order, without discarding or resampling them.
Startup temporarily retains up to eight prepared CPU microbatches per rank in
addition to the ordinary prefetch queue; this buffer drains during training.
All ranks announce readiness through the existing CPU rendezvous store before
DDP initialization, and again before the first accumulated loss reduction.
The second check drains first-update CUDA work before announcing readiness.
These are one-time startup checks, not additional per-update collectives.
The NCCL training timeout, objective, augmentation recipe, optimizer, schedule,
language quotas and effective batch remain unchanged.

The wrapper now explicitly sets OpenMP/BLAS/Numba thread budgets to one, even
if the invoking Conda environment exported larger values. Previously the shell
fallback preserved such values, allowing each of 64 data workers per node to
create its own large CPU thread pool. `GQASR_CPU_THREADS` is the explicit override.

Expect `training_startup_ready` for each rank, `training_startup_wait` progress
from rank zero, then `training_startup_all_ready` with 96 ranks at each stage,
followed by the usual training updates. Startup waits have a
separate 1,200-second bound (`GQASR_STARTUP_TIMEOUT`). Missing ranks and their
phases are reported; slow local startup work also dumps Python stacks after
180 seconds. This covers process spawning, which the normal DataLoader
batch-delivery timeout does not cover. A permanently stuck node is still an
error, not silently skipped.

Validation includes a real two-process Gloo regression: delaying one rank beyond
the unchanged collective timeout fails without readiness and succeeds with it.
The resumed training-loop test also uses spawned data workers, checks identical
collective order across ranks, and verifies global loss publication. These CPU
tests do not establish H100/NCCL throughput or hardware health.

## Inspect the October 20 resume timeout before relaunching

The supplied trace from node `inception-H100-hpc-056` (node rank 9) identifies
sequence 16 as `Runtime.all_finite` in the training loss check: a one-element
int32 MIN all-reduce, before the synchronized backward. This is different from
the average-loss logging operation: two float32 values reduced with SUM after
an optimizer update. With resume step 7,500 and logging every five updates, the
first average-loss publication is at 7,505. Its reduction runs on all ranks;
only publishing to W&B runs on rank zero.

The traceback establishes where those ranks waited, not the root cause.
Compare all ranks' recorder entries before deciding between collective-order
divergence, a rank stalled before entering the collective, or GPU/transport
failure. A missing dump is missing evidence. The reporting rank need not be
the rank causing the stall. These messages do not establish Python-environment
corruption or an OOM.

Collect existing evidence on **each of the 12 nodes**, using its original node
rank. For example, use `NODE_RANK=9` on node 056 and `NODE_RANK=0` on master 051:

```bash
git pull origin main
NODE_RANK=9  # replace with this node's original rank, 0 through 11
gemma_speech/.venv/bin/python -m gqasr.nccl_report \
  --node-rank "$NODE_RANK" --sequence 16 \
  --output "$PWD/gemma_speech/gqasr/verification/nccl-startup-20261020/node-$NODE_RANK.json"
```

This reads existing recorder files and a bounded tail of the latest encoder log.
It uses no GPU, loads no model, launches no collective and leaves the source files
intact. Keep the original dumps until diagnosis is complete. If logging was
redirected, pass `--log-dir /actual/node/local/logs`; a custom recorder prefix can
be supplied with `--trace-prefix /actual/path/prefix_`. Check report timestamps
against the failed attempt. Copy the report directory back for comparison.

NCCL initialization now receives `device_id=cuda:LOCAL_RANK` explicitly, removing
the device-inference warning. The wrapper uses PyTorch 2.11's `TORCH_FR_BUFFER_SIZE`
and `TORCH_FR_DUMP_TEMP_FILE`, while accepting old overrides as fallback. These
changes address the warnings; they are not proof of a fix for sequence 16.
See [PyTorch process-group initialization](https://docs.pytorch.org/docs/2.11/distributed.html#torch.distributed.init_process_group)
and [flight-recorder guidance](https://docs.pytorch.org/tutorials/unstable/flight_recorder_tutorial.html).
The model, augmentations, loss checks, gradient clipping and loss averaging are
unchanged. Retain the failed output directory; any later retry needs a fresh
output directory in its config.

## Recovery from update 7,500

Use `scripts/resume_gqasr_encoder_12node.sh` for the interrupted encoder run.
It selects `03_encoder12_b20_acc4_12node.yaml`, loads
`outputs/encoder12_b80/checkpoint-00007500`, and writes to the new
`outputs/encoder12_b20_acc4_resume7500_startupfix` directory. The original checkpoints stay
intact. Do not invoke this recipe as a fresh encoder stage without `--resume`.

| Setting | Restart |
|---|---|
| Microbatch / accumulation | 20 / 4 |
| Effective batch | 80 per rank, 7,680 across 96 GPUs; unchanged |
| Workers / prefetch | 8 per GPU / 1 batch per worker |
| DataLoader timeout | 180 seconds for delivery of a batch |
| Optimizer and learning rates | Restored AdamW state; existing schedule continues |
| Updates | Next 7,501; stop at 31,446; 23,946 remain |
| Code-switching | Disabled; waveform augmentation and SpecAugment unchanged |
| Output / W&B name | `encoder12_b20_acc4_resume7500_startupfix` / `gqasr-encoder12-b20-acc4-resume7500-startupfix` |

Accumulation 6 would increase the global batch to 11,520 and the remaining
sampled-example budget by 50%. This recovery uses 4 to retain the existing
budget and schedule. Smaller microbatches reduce peak activations and the likely
amount of padding; speed must be measured on the cluster. Four forward/backward
calls can also add overhead. This is not a promise of a fourfold speedup.

The prefetch queue capacity falls from 1,280 to 160 anchor examples per GPU
(80 × 8 × 2 versus 20 × 8 × 1). Actual host/shared-memory usage also depends
on audio duration and worker internals. There are 64 DataLoader workers per
96-core node. The wrapper sets OpenMP/BLAS thread counts to one, overriding
inherited values, to avoid nested CPU oversubscription.

The hot loop now checks all accumulated losses once at each optimizer boundary,
before the synchronized backward. Nonfinite losses still abort every rank without
updating the optimizer. DDP's `no_sync()` avoids gradient reduction on the first
three microbatches. Gradient clipping retains the ordinary PyTorch norm and scale;
per-parameter diagnostic scans only run if the norm check fails. No FP64 clipping
or nonfinite-value replacement is used. Scalar extraction and diagnostics are
limited to logged updates on rank zero.

The explicit `--allow-microbatch-change` flag allows only batch size and
accumulation to differ, and requires identical effective batch, language quotas,
world size, index metadata, seeds, augmentations, trainable layers and schedule.
It restores optimizer moments, step and RNG state. Microbatch grouping changes
token weighting, padding, floating-point reduction order and augmentation seeds:
this is a controlled continuation, not a bit-identical replay of batch 80.
Each resumed run records its source and changes in `resume.json`.

Before launching, have the cluster team check or replace the allocation containing
node `inception-H100-hpc-009` and HCA `mlx5_an0:1`: the reported fatal RDMA event
and Lustre I/O error are not repaired by changing batch size. The worker bus-error
message alone does not establish that `/dev/shm` was full. Keep 12 healthy nodes
with unique ranks 0–11.

Run once on a cluster machine, from the repository root:

```bash
git pull origin main
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT
bash scripts/resume_gqasr_encoder_12node.sh check
```

This CPU-only check loads the checkpoint and optimizer, verifies configuration,
index/tokenizer compatibility, finite weights and optimizer state, and a fresh
output directory. It does not rescan the corpus or write model weights. Require
`resume_check_complete` with `passed: true`; this is not an RDMA/storage health test.

On master 051:

```bash
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT
bash scripts/resume_gqasr_encoder_12node.sh 0
```

On each remaining node, with its rank 1–11:

```bash
unset GQASR_CONFIG GQASR_RESUME_CHECKPOINT
bash scripts/resume_gqasr_encoder_12node.sh YOUR_RANK
```

No download, export or index rebuild is needed. Normal `--stage encoder` still
selects the original batch-80 recipe; the recovery wrapper selects batch 20.
All nodes must use the same checkout/configuration. A later recovery can override
`GQASR_RESUME_CHECKPOINT` and `GQASR_CONFIG` (with another fresh output directory).

Console logs and NCCL flight-recorder dumps default to node-local
`/tmp/gqasr-$USER-logs`; set `GQASR_LOG_DIR` to a durable node-local directory if
available, and collect these files before the allocation is released.
Weights, `metrics.jsonl` and W&B tracking metadata remain in the shared output
directory. This keeps console diagnostics available during a shared-filesystem
failure; shared storage is still required for training data and checkpoints.
The wrapper enables a 2,000-event NCCL recorder and dumps on timeout, following
the [PyTorch flight-recorder guidance](https://docs.pytorch.org/tutorials/prototype/flight_recorder_tutorial.html).
It does not increase collective timeouts or disable InfiniBand.

Console and W&B expose recent seconds/update and global anchors/second.
The interval includes startup and any preceding validation/checkpoint work;
compare several steady-state windows after workers warm up. New
`performance/rank0_data_wait_seconds_per_update` and
`performance/rank0_data_wait_seconds_per_microbatch` measure host wall time
waiting for batch delivery, without additional GPU synchronization. They are
rank-zero diagnostics, not measurements of every rank or GPU compute time.
Global loss and separate LR curves remain available.

## Original encoder-stage setup

Run after the Arabic benchmark finishes. Pulling code during a benchmark changes
its implementation fingerprint and causes its final provenance check to fail.

`configs/03_encoder12_b80_12node.yaml` is the default encoder stage and starts from
`outputs/full_b40_frozen/final`, the completed 31,446-update model. It writes a
separate run to `outputs/encoder12_b80`, with `final/` and periodic checkpoints.
The first launch uses the saved weights with a **new optimizer and schedule**;
do not pass `--resume` from the previous frozen-encoder run.

## Training settings

| Setting | Value |
|---|---|
| Encoder | All 12 blocks, plus output projection |
| Convolutional frontend | Frozen, kept in evaluation mode |
| Bridge, decoder and tied embeddings | Trainable; 28 decoder layers |
| Per GPU batch / accumulation | 80 / 1 |
| Nodes / GPUs | 12 / 96 |
| DataLoader workers | 8 per GPU process, 64 per node, 768 across 12 nodes |
| Global batch | 7,680 anchor utterances per update |
| Synthetic code-switch augmentation | Disabled in batch 80; waveform augmentation and SpecAugment remain enabled |
| Updates | 31,446; sampling with replacement, not an exact epoch traversal |
| Encoder / bridge LR | 1e-6 / 5e-6 |
| Decoder / embeddings LR | 2e-6 / 1e-6 |
| Schedule | 1,000-update linear warmup, cosine decay over remaining 30,446 updates to 10% |
| Validation | Before the first update, then every 500 updates and at completion |
| Saving | Every 500 updates, retaining the latest eight periodic checkpoints, plus final |
| Tracking | W&B project `gqasr`, run `gqasr-encoder12-b80`; local `metrics.jsonl` |

Batch 80 keeps the requested **31,446 optimizer updates**, totaling 241,505,280
sampled anchors, twice the batch-40 recipe's budget. Learning rates, warmup,
decay and evaluation/save intervals remain the same in optimizer-update units.
Larger batches do not guarantee a faster complete run when the update count is
unchanged. If the goal is the same sampled-example budget as batch 40, use
15,723 updates with a separately adjusted schedule instead.

The unchanged `03_encoder12_b40_12node.yaml` remains available with synthetic
code-switch augmentation enabled. To select it,
set this on every node before launching or checking:

```bash
export GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/03_encoder12_b40_12node.yaml"
```

For the default batch-80 recipe, unset an earlier `GQASR_CONFIG` override, or set
it explicitly to `03_encoder12_b80_12node.yaml` on every node.

The exported encoder has **only 12 blocks**. This trains all of them, rather than
the last 12 of a deeper encoder. `train_acoustic_frontend: false` prevents the
previous all-blocks selection from also unfreezing its convolutional frontend.
Earlier recipes retain their behavior when this new option is absent.

The same indexed manifests and language proportions, with per-rank quotas
ar=40, en=24, hi=8, ml=4, zh=4, are used. Batch 80 sets
`online_code_switching: false` and `code_switch_probability: 0.0`: the code-switch
pool and its quarantine list are not loaded, and no synthetic multilingual
concatenation is performed. Existing multilingual recordings remain in the
indexed data. The retained pool paths allow a later comparison run. No rescan
or re-export is needed. Online dataset additions are not enabled in this recipe.
Training uses seed 43 for new draws, while validation retains seed 42 and the
same 460-utterance subset. This keeps the internal curves comparable. Chinese
remains in training/evaluation; encoder tuning alone is not a demonstrated fix
for its earlier recognition failures.

SpecAugment stays before the encoder, with the existing per-frequency mean fill
over valid frames. Speed, noise and codec augmentation also retain their settings.
The zero-fill failure reproducer remains in the tests. The
training optimizer still uses ordinary PyTorch gradient clipping; no FP64 norm
workaround was reintroduced. FP64 group norms in the optional backward audit
are diagnostic only; nonfinite FP32 total norms now fail that audit too.

## Cluster checks before launch

From the repository root, after benchmarking completes:

```bash
git pull origin main
export GQASR_CONFIG="$PWD/gemma_speech/gqasr/configs/03_encoder12_b80_12node.yaml"
bash gemma_speech/cluster/gqasr_stage.sh test -q
bash scripts/run_gqasr_12node.sh 0 --stage encoder --preflight

# On one free GPU: real saved weights, real augmented batches from eight sampler
# ranks, forward/backward, standard clipping, AdamW and optimizer-state memory.
# Updates are in memory only; this command never overwrites or saves model weights.
GQASR_CUDA_VISIBLE_DEVICES=0 \
  bash gemma_speech/cluster/gqasr_stage.sh backward-audit \
    --mode full --precision bf16 --optimizer-step --rank-count 8 --batches 2
```

Require `backward_audit_complete` with `passed: true`, finite norms and adequate
GPU memory headroom. The start record lists the exact trainable block indices,
frontend status and optimizer groups. This checks 16 representative batches; it
does not establish every possible batch's memory fit or 96-rank NCCL behavior.
The reported 30% peak came from the batch-40 encoder audit including AdamW.
Repeat it at batch 80: padding to the longest audio/target in a batch and DDP
buffers can make production peaks higher than this single-GPU sample.
Do not disable clipping checks or mask a failure to start the full run.

The local tests exercise a small 12-block model, real frontend features (including
silence), mean-filled SpecAugment, BF16/FP32 backward, frozen-weight invariance,
AdamW and exact checkpoint resume. Actual H100 memory and the full saved model
must still be checked with the command above.

## Launch

On master 051:

```bash
bash scripts/run_gqasr_12node.sh 0 --stage encoder
```

On the other nodes, with unique ranks 1 through 11:

```bash
bash scripts/run_gqasr_12node.sh YOUR_RANK --stage encoder
```

All nodes must use the same checkout, configuration and shared artifacts. The
encoder stage uses port **29613** and all eight GPUs by default. Set `GPU_IDS`,
`MASTER_ADDR` or `MASTER_PORT` consistently if your allocation requires overrides.
Ensure W&B authentication is already configured; `WANDB_MODE=offline` is supported.

Compare the step-zero validation with subsequent strict Arabic WER/CER and each
language's curves. Initial validation has no training loss and is labeled as such.
Learning rates, pre-clipping gradient norms and augmentation counts are logged.
W&B's `train/loss` averages the unscaled microbatch losses across all GPU ranks
and all accumulated microbatches since the preceding log (or resume). Each
microbatch loss is the label-smoothed mean over its supervised tokens; the
logged mean uses equal microbatch weighting, matching the training objective.
`metrics.jsonl` records the window's starting update, update count and total
microbatch count. `train/rank0_last_microbatch_loss` remains a separate diagnostic.
Historical records containing only the rank-zero loss cannot reconstruct this
global mean, and the follower does not relabel them as `train/loss`.

The LR curves appear only as `lr/bridge`, `lr/acoustic`, `lr/decoder` and
`lr/embedding`, plotted against `trainer/step`. These are the actual scheduled
optimizer rates. New records no longer duplicate them as `train/lr_*`; previously
uploaded duplicate panels may remain in an existing W&B workspace and can be
hidden there. This change does not delete existing W&B history.

Worker counts and logging changes take effect when the training processes start;
pulling code does not change an already-running job. Checkpoint resume permits
worker-count changes with the same training settings and a fresh output directory.
Review early checkpoints before committing to the whole budget; improvement is
not guaranteed by unfreezing.

For an unchanged-recipe recovery, use its own `checkpoint-NNNNNNNN`, a copy
of the same config with a fresh `output_dir`, and `--resume`. For microbatch
regrouping with the same effective batch, use the guarded recovery above.
Changing trainable layers, seeds, quotas or schedule remains disallowed on resume.
