# GQASR encoder continuation on 12 nodes

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

For recovery **within this new stage**, use its own `checkpoint-NNNNNNNN`, a copy
of the same config with a fresh `output_dir`, and `--resume`. Changing batch size,
trainable layers, seeds or schedule is a new experiment and must not restore an
incompatible optimizer state.
