# AudPro: five-node cluster commands

Run on five available nodes with eight H100 GPUs each, separate from the GPUs
running AudTurbo. This launcher runs `torchrun` directly on each node. It ignores
Slurm environment settings and does not call `srun` or `scontrol`.

Recipe: `audm/configs/audpro_full_5node.yaml`. It reuses the same stage-02 full
training/evaluation manifests and augmentation settings as AudTurbo. Code-switch
mixing is disabled. Training uses three weighted coverage epochs, microbatch 10,
accumulation 4, and 40 GPUs: **1,600 samples per optimizer update**. There are
four DataLoader workers per GPU (32 per node, 160 total). Each GPU processes two
20-example language quota windows per update: 20 Arabic, 12 English, 4 Hindi,
2 Malayalam and 2 Chinese samples in total. BF16, ZeRO-2 and gradient
checkpointing remain enabled. Base learning rate is 2e-5, decoder factor 0.5,
embedding factor 0.25, warmup 500 updates, cosine schedule, save/eval every
1,000 updates. The scan determines updates per coverage epoch from the data.

AudPro retains the trained QASR encoder and decoder. It needs no new bridge
alignment stage. Export from the trained QASR checkpoint, not AudTurbo's adapter.
The export is a separate AUD model; the source checkpoint is not modified.

## 1. Shared setup — once

Use the already installed AUD environment. No environment reinstall is needed
while AudTurbo is training. Pull the new files once in the shared checkout:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
git pull --ff-only origin main

cat > audpro-cluster.env <<'ENV'
export AUD_REPO_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export AUD_CONDA_ENV=/lustrefs/shared/shahin.konadath/workspace/conda/envs/audm
export QASR_CHECKPOINT=/REPLACE/WITH/TRAINED/QASR/CHECKPOINT
export AUDPRO_MASTER_ADDR=inception-H100-hpc-031
export AUDPRO_MASTER_PORT=29601
export AUDPRO_MODEL="$AUD_REPO_ROOT/models/AudPro-v1"
export AUDPRO_RUN_ROOT="$AUD_REPO_ROOT/output/audpro-v1"
export AUDPRO_FULL_CONFIG="$AUD_REPO_ROOT/audm/configs/audpro_full_5node.yaml"
export TOKENIZERS_PARALLELISM=false
ENV

# Set the trained QASR checkpoint; rank 0 is inception-H100-hpc-031.
vi audpro-cluster.env
source audpro-cluster.env
eval "$(conda shell.bash hook)"
conda activate "$AUD_CONDA_ENV"
test -f "$QASR_CHECKPOINT/config.json"
```

AudPro uses port 29601 by default, separate from AudTurbo's default 29591.
The master must be reachable from all five nodes. Use the same environment file
on every node. Model, output, config and rendezvous settings use `AUDPRO_*`;
the launcher's defaults do not inherit AudTurbo's `AUD_MODEL`, `AUD_RUN_ROOT`,
`AUD_FULL_CONFIG`, `AUD_MASTER_ADDR` or `AUD_MASTER_PORT` values.
The launcher sets a node-local Triton cache under `/tmp/$USER/`; override it
with `AUDPRO_TRITON_CACHE_DIR` if needed.

## 2. Export, verify and scan — once on an available node

```bash
bash audm/scripts/audpro.sh export
bash audm/scripts/audpro.sh verify
bash audm/scripts/audpro.sh preflight full --scan
```

Export runs on CPU; verify uses `cuda:0`. Use an available GPU for verify, not a
GPU already running AudTurbo. If a valid AudPro export already exists at
`AUDPRO_MODEL`, skip export (existing destinations are protected). Preflight
reads shared data and can take time; it does not start distributed training.

## 3. Set up each node

Open one persistent shell/session on each of the five training nodes:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
source audpro-cluster.env
eval "$(conda shell.bash hook)"
conda activate "$AUD_CONDA_ENV"
export NODE_RANK=0  # inception-H100-hpc-031; assign 1, 2, 3, 4 on the other nodes
```

Optional dry run:

```bash
AUD_DRY_RUN=1 bash audm/scripts/audpro.sh full "$NODE_RANK"
```

## 4. Connectivity and rehearsal — run each action on all five nodes

```bash
bash audm/scripts/audpro.sh nccl "$NODE_RANK"
```

Start all five ranks before waiting for completion. Rank 0 should report
`"nccl": "passed"`, `"world_size": 40`, `"sum": 820.0`.
After it finishes on every node, run:

```bash
bash audm/scripts/audpro.sh smoke "$NODE_RANK"
```

This rehearses two optimizer updates with the actual model, data and evaluation.
It writes under `$AUDPRO_RUN_ROOT/full/smoke-test`. Check finite losses and GPU
memory on all nodes. A repeated smoke run needs a fresh run root.

## 5. Three-epoch full training — all five nodes

```bash
bash audm/scripts/audpro.sh full "$NODE_RANK"
```

Start one launch on each node, ranks 0–4. The first waits for the others to join;
do not wait for it to finish before launching the next node. Full training loads
`$AUDPRO_MODEL` and trains encoder, adapter/projector and decoder. The final model
is saved at `$AUDPRO_RUN_ROOT/full`; checkpoints are inside that directory.
W&B run name: `audpro-full-5node`. Launcher logs are in
`$AUDPRO_RUN_ROOT/launcher-logs`.

To resume an interrupted run, use the same config, paths and world size, and run
on all five nodes:

```bash
bash audm/scripts/audpro.sh full "$NODE_RANK" --resume auto
```

Local launcher/config tests do not validate cluster GPU memory or networking;
complete the connectivity and smoke checks before the full run.
