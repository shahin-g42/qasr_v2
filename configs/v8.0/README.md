# QASR v8.0 Stage 2

Continue from `models/bck/full/v1` with a fresh optimizer and scheduler, using
only `asr_cleaned_manifests/full_v1`. Existing configs and launchers are unchanged.
The new config requires the accompanying shared-code changes; copying YAML alone
is insufficient. Sync this checkout to the cluster before running these commands.

## Recipe

10 nodes × 8 H100 80 GB; `qasr3`; BF16, SDPA, ZeRO-2, gradient checkpointing.
Per-GPU batch 20, accumulation 1, global batch 1,600. Per-batch language counts:
Arabic 10, English 6, Hindi 2, Malayalam 1, Chinese 1. Weighted sampling covers
all eligible records and repeats smaller languages. The reported counts imply
75,469 updates and 120,750,400 draws. Preflight verifies the post-filter counts.

Encoder/projector LR 2e-5, decoder 1e-5, embeddings 5e-6; warmup 1,000,
WSD decay 7,500 steps, final LR ratio 0.1. Save/evaluate every 2,500 updates,
retain six checkpoints. WER/CER preserve harakat, punctuation and case. The decode
callback uses up to 100 fixed samples per language across its manifests.
The empty Malayalam eval and 16-record Expresso diagnostic are excluded.

## Run

On each node, use the same checkout and cache:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export HF_HOME=/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/.cache
export TOKENIZERS_PARALLELISM=false
```

Run the input audit **once**, on any node:

```bash
bash scripts/run_v80_phase2_node.sh 0 localhost --preflight --scan
```

This reads all configured manifests and can take time. It validates the snapshot
counts after duration/text filtering, checkpoint files, processor limits and
MUSAN directory. It does not decode all audio or verify every target-token length.
If counts differ, review the report and recompute the step budget before training.

Run a two-update rehearsal on one node with all eight GPUs:

```bash
NNODES=1 bash scripts/run_v80_phase2_node.sh 0 localhost --smoke-test
```

Require a successful exit, finite loss, working WER/CER output and adequate GPU
memory headroom. A random smoke subset does not prove worst-case 35-second batch
memory safety. Check long-clip batches before leaving production unattended.
Smoke output is isolated under `output/v80/full/smoke-test`. Startup still indexes
the full manifests before subsetting, so it can spend considerable time on CPU/I/O.

Choose the actual hostname of node rank 0. Set the same `MASTER` on all ten nodes.
The following hostname is only an example from the previous run; use it only if
that machine is actually your rank-0 node:

```bash
MASTER=inception-H100-hpc-051
```

Launch on **each** machine with its unique rank, 0 through 9 (no repeated ranks).
First use the distributed smoke command on all ten machines:

```bash
RANK=0  # change to 1, 2, ... 9 on the other machines
bash scripts/run_v80_phase2_node.sh "$RANK" "$MASTER" --smoke-test
```

After every node exits successfully, run production on all ten:

```bash
bash scripts/run_v80_phase2_node.sh "$RANK" "$MASTER"
```

Use persistent terminal sessions or your cluster scheduler. The launcher runs in
the foreground and activates `qasr3`. Port 29580 must be reachable between nodes;
set `MASTER_PORT` identically everywhere if changing it. Authenticate W&B before
launching. Logs are in `output/v80/logs`; checkpoints in `output/v80/full`.
Do not run smoke and production simultaneously. Production rejects a world size
other than 80 and an output directory already containing checkpoints.

## Resume an interrupted v8.0 run

Use the same complete checkpoint on all ten nodes:

```bash
bash scripts/run_v80_phase2_node.sh "$RANK" "$MASTER" \
  --resume-from-checkpoint /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/v80/full/checkpoint-2500
```

Replace 2500 with the latest complete saved step. The backup model initializes the
first run; it is not a v8.0 optimizer-state resume checkpoint. Do not lower batch
size without adjusting language quotas and recomputing the coverage budget.

Evaluate the backup and new checkpoints on the same corrected reference panel.
The tiny Hindi/Malayalam evaluation sets and automatically corrected Arabic
references need a human-reviewed panel before claiming general quality gains.
