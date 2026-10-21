# AudTurbo: cluster commands

This follows the existing QASR **manual per-node launch** workflow. Use the
separate export node first, then the 8 training nodes, with 8 GPUs per node.
Run each distributed action once on **every** training node with a unique rank
0–7. The script launches `torchrun` directly on the current node, even when
`SLURM_*` variables are present; it does not call `srun` or `scontrol`. Start the
same action on all eight nodes without waiting for the first node to finish:
each launch waits for the other ranks to join. Once all nodes have started,
wait for that action to finish everywhere before starting the next action.

Copy the updated `audm/` directory into the shared cluster QASR checkout first.
The data recipe remains `configs/v8.1/02_full_12node.yaml`; its referenced data,
noise assets must remain accessible. Code-switching is disabled by default.

The default is 8 nodes × 8 GPUs = 64 ranks. Training uses microbatch 10 and
accumulation 4: **2,560 samples per optimizer update**, with 4 DataLoader workers
per GPU (32 per node). Adapter training remains 2,000 updates; full training
still covers one weighted epoch. Use a fresh `AUD_RUN_ROOT` when moving from a
12-node run; exact resume requires the original world size and recipe. Existing
12-node recipes remain available through the generic distributed launcher with
`AUD_NNODES=12`, or Slurm with `sbatch --nodes=12` and a 12-node recipe.

## 1. Set shared paths — once

Replace the checkout path if yours differs, the trained QASR checkpoint, and
the rank-0 hostname below. The checkpoint must be the **trained QASR model you
want to retain**, not an initial conversion or an upstream Qwen checkpoint.

```bash
export AUD_REPO_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
cd "$AUD_REPO_ROOT"

cat > audturbo-cluster.env <<'ENV'
export AUD_REPO_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export QASR_CHECKPOINT=/REPLACE/WITH/TRAINED/QASR/CHECKPOINT
export AUD_MASTER_ADDR=REPLACE_WITH_RANK0_TRAINING_HOST
export AUD_MASTER_PORT=29591
export AUD_MODEL="$AUD_REPO_ROOT/models/AudTurbo-v1"
export AUD_RUN_ROOT="$AUD_REPO_ROOT/output/audturbo-v1"
export AUD_CONDA_ENV=/lustrefs/shared/shahin.konadath/workspace/conda/envs/audm
export TOKENIZERS_PARALLELISM=false
ENV

# Edit the three site/checkpoint/host values before proceeding.
vi audturbo-cluster.env
source audturbo-cluster.env
test -f "$QASR_CHECKPOINT/config.json"
test -f audm/scripts/audturbo.sh
```

All nodes must use the same file and paths. The existing QASR launcher used
`inception-H100-hpc-051` as its master; use that only if it is actually rank 0
of this new allocation. The chosen port must be available on that host.

## 2. Prepare the shared environment — once

The QASR launcher uses the shared `qasr3` environment. Create a dedicated AUD
clone and install the package. If an AUD environment already exists, activate
it and run the installation/check instead of creating it again.

```bash
eval "$(conda shell.bash hook)"
conda create -y --prefix "$AUD_CONDA_ENV" \
  --clone /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3
conda activate "$AUD_CONDA_ENV"
python -m pip install -e './audm[train]'
python -c 'import torch, transformers, deepspeed; print("Torch", torch.__version__, "CUDA", torch.version.cuda, "Transformers", transformers.__version__, "DeepSpeed", deepspeed.__version__)'

# The supplied recipes use W&B. Authenticate once or select offline logging:
wandb login
# Alternative for a cluster without W&B connectivity:
# export WANDB_MODE=offline
# Add the same export to audturbo-cluster.env for every node.
```

Keep the known working QASR NCCL/network environment settings for these nodes.
The AUD launcher does not guess interface/HCA names. Run within an allocation
where each node already exposes its eight assigned GPUs; it does not overwrite
`CUDA_VISIBLE_DEVICES` or bypass scheduler GPU assignments.

## 3. Export and verify — export node only

The AUD recipes retain the stage-02 full datasets and disable code-switch
augmentation by default (`code_switch_index: null`, `code_switch_probability: 0.0`).
No code-switch index or Malayalam index refresh is needed for default preflight
or training. Speed perturbation, noise injection, codec augmentation and
SpecAugment retain their existing settings.

An already verified AudTurbo export does not need to be exported again. Continue
directly with preflight below. If changing an existing training run's recipe,
use a fresh output directory; exact resume requires the original recipe.

```bash
bash audm/scripts/audturbo.sh export
bash audm/scripts/audturbo.sh verify
bash audm/scripts/audturbo.sh preflight adapter --scan
```

Export uses CPU and downloads the Captioner encoder-containing shards. Verify
uses `cuda:0` and disables Hub/network model loading. No GQASR checkpoint is
needed. An existing model destination is rejected; if export already succeeded,
start with `verify` instead of exporting again. Preflight checks the stage-02
manifests (and the code-switch index only if explicitly enabled); `--scan` counts the corpus
and calculates coverage. This can take time on the full dataset.

## 4. Prepare each training node

On each of the 8 nodes, open a persistent shell/session, activate the shared
environment and assign a **unique** `NODE_RANK`. Keep those shells for the
subsequent commands.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
source audturbo-cluster.env
eval "$(conda shell.bash hook)"
conda activate "$AUD_CONDA_ENV"

export NODE_RANK=0  # use 0, 1, ..., 7 on the respective nodes
```

Optional command inspection without starting workers:

```bash
AUD_DRY_RUN=1 bash audm/scripts/audturbo.sh adapter "$NODE_RANK"
```

## 5. NCCL connectivity — all 8 nodes

```bash
bash audm/scripts/audturbo.sh nccl "$NODE_RANK"
```

Launch promptly on every node. Rank 0 must report `"nccl": "passed"`,
`"world_size": 64`, and `"sum": 2080.0`. Resolve connectivity failures before
loading full training models.

## 6. Two-update rehearsal — all 8 nodes

```bash
bash audm/scripts/audturbo.sh smoke "$NODE_RANK"
```

This exercises the actual export, data loading, augmentation,
distributed training and evaluation. It writes under
`$AUD_RUN_ROOT/adapter/smoke-test`. It still validates/indexes the production
manifests, so startup is not a synthetic two-batch test. Inspect finite losses,
successful completion, and GPU memory before proceeding. For a second rehearsal,
use a new `AUD_RUN_ROOT` on every node; existing output is protected.

## 7. Adapter alignment — all 8 nodes

```bash
bash audm/scripts/audturbo.sh adapter "$NODE_RANK"
```

Recipe: `audm/configs/audturbo_adapter_8node.yaml`.

- Encoder and trained QASR decoder frozen; new adapter trainable.
- 2,000 optimizer updates, learning rate `2e-4`, 100 warm-up updates.
- BF16, ZeRO-2, microbatch 10 × accumulation 4 × 64 GPUs = 2,560 samples/update.
- Stage-02 language quotas and other augmentations retained; code-switching disabled.
- Evaluation/checkpoint every 500 updates; final model at `$AUD_RUN_ROOT/adapter`.

Resume an interrupted adapter run on **all nodes** with:

```bash
bash audm/scripts/audturbo.sh adapter "$NODE_RANK" --resume auto
```

## 8. Full fine-tuning

After adapter training finishes on every node, review its loss and per-language
WER/CER. The 2,000-update recipe is an initial alignment budget, not an automatic
quality gate. Full training is intentionally not submitted automatically.

Run once on a node with the environment active:

```bash
bash audm/scripts/audturbo.sh preflight full --scan
```

Then run on **all 8 nodes**:

```bash
bash audm/scripts/audturbo.sh full "$NODE_RANK"
```

Recipe: `audm/configs/audturbo_full_8node.yaml`. It loads the **trained adapter
stage model**, unfreezes encoder/decoder/adapter, and runs one coverage epoch.
The base LR is `2e-5`, decoder LR `1e-5`, embedding LR `5e-6`, with 500 warm-up
updates and a cosine schedule. Evaluation/checkpoint every 1,000 updates.
The effective batch, language mix and augmentation settings remain unchanged.

Resume full training on **all nodes**:

```bash
bash audm/scripts/audturbo.sh full "$NODE_RANK" --resume auto
```

Do not change the recipe, world size or data/index in an optimizer-resume run.
Use a new output and a weights-only start for a changed fine-tuning recipe.

## Outputs and final verification

```text
$AUD_MODEL/                         Initial AudTurbo export
$AUD_RUN_ROOT/launcher-logs/        Per-action, per-node logs
$AUD_RUN_ROOT/adapter/smoke-test/   Rehearsal output
$AUD_RUN_ROOT/adapter/              Aligned model, metrics, numbered checkpoints
$AUD_RUN_ROOT/full/                 Fine-tuned model, metrics, numbered checkpoints
```

On a GPU node after full training:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m audm verify "$AUD_RUN_ROOT/full" --device cuda:0
```

This checks loading and numerical execution, not ASR quality. Use the held-out
stage-02 WER/CER and your external ASR benchmark before choosing a release.

For a scheduler-managed launch, use `audm/scripts/train.slurm` as described in
`README.md`; do not nest this manual per-node wrapper inside a Slurm job.
