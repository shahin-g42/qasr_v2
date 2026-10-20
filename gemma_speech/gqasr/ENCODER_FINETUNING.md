# GQASR encoder continuation on 12 nodes

## Current recovery: batch 40 from update 7,500

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

The recovery wrapper passes `--allow-batch-change`. This permits only a batch /
accumulation change with proportional language quotas; it still rejects changes
to the index, tokenizer, trainable layers, seeds, augmentations, world size and
schedule. It records old/new global batches and all recipe changes in
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
