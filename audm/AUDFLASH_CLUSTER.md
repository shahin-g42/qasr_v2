# AudFlash: single-node cluster commands

Use the remaining node with eight available GPUs. This follows the AudTurbo
workflow: export, verify, preflight, NCCL check, smoke, adapter alignment, then
full training. The launcher uses direct `torchrun`, ignores Slurm settings, and
never invokes `srun` or `scontrol`. Node rank is always 0; master defaults to
127.0.0.1:29611. The other AUD launchers and recipes are unchanged.

Both `audflash_adapter_1node.yaml` and `audflash_full_1node.yaml` use microbatch
20, accumulation 4 and eight GPUs: **640 samples per optimizer update**.
There are four DataLoader workers per GPU (32 total), with persistent workers
and pinned memory. BF16, ZeRO-2 and gradient checkpointing are enabled.
Evaluation batch size is one per GPU. The transcript limit is **512 tokens**,
inherited from the stage-02 data recipe; evaluation can generate up to 768 tokens
including language markers and EOS. Transcripts above 512 are replaced by the
resilient loader unless removed in a separately prepared filtered manifest.

Changes do not affect an already running process. Exact `--resume auto` requires
the original recipe and data; a run started with the temporary 1,024-token recipe
cannot resume with the restored recipe. The single-node YAML files are restored
to their pre-increase contents.

The same stage-02 training/evaluation manifests and augmentations are reused.
Code-switching is disabled; no code-switch index is needed. Each microbatch
contains the 20-example language quota (10 Arabic, 6 English, 2 Hindi,
1 Malayalam, 1 Chinese), repeated four times per optimizer update. Gemma SpecAugment operates on log-mel features. Recordings longer than 30 seconds
are filtered to respect the audio tower's context limit; manifest paths stay the same.

## 1. Shared setup — on the AudFlash node

Reuse the installed AUD environment; no reinstall is needed while other runs
are training. Set GQASR_CHECKPOINT to a trained GQASR checkpoint containing the
decoder you want to retain. This is not the QASR or AudTurbo checkpoint.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
git pull --ff-only origin main

cat > audflash-cluster.env <<'ENV'
export AUD_REPO_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
export AUD_CONDA_ENV=/lustrefs/shared/shahin.konadath/workspace/conda/envs/audm
export GQASR_CHECKPOINT=/REPLACE/WITH/TRAINED/GQASR/CHECKPOINT
export AUDFLASH_MODEL="$AUD_REPO_ROOT/models/AudFlash-Gemma4-v1"
export AUDFLASH_RUN_ROOT="$AUD_REPO_ROOT/output/audflash-gemma4-v1"
export AUDFLASH_MASTER_ADDR=127.0.0.1
export AUDFLASH_MASTER_PORT=29611
export AUDFLASH_ADAPTER_CONFIG="$AUD_REPO_ROOT/audm/configs/audflash_adapter_1node.yaml"
export AUDFLASH_FULL_CONFIG="$AUD_REPO_ROOT/audm/configs/audflash_full_1node.yaml"
export RUN_ID=cs_v1_ml_v2
export AUDFLASH_ENCODER_SOURCE=google/gemma-4-E2B-it
export TOKENIZERS_PARALLELISM=false
ENV

vi audflash-cluster.env  # set the trained GQASR checkpoint
source audflash-cluster.env
eval "$(conda shell.bash hook)"
conda activate "$AUD_CONDA_ENV"
export NODE_RANK=0
test -f "$GQASR_CHECKPOINT/config.json"
```

The launcher uses AUDFLASH_* model, output, config and rendezvous settings so
inherited AudTurbo/AudPro settings cannot redirect the run. It automatically
uses a node-local Triton cache under /tmp/$USER, overridable through
AUDFLASH_TRITON_CACHE_DIR. Use the existing W&B login or WANDB_MODE=offline.

## 2. Export, verify and preflight

```bash
bash audm/scripts/audflash.sh export
bash audm/scripts/audflash.sh verify
bash audm/scripts/audflash.sh preflight adapter --scan
```

Export combines the pretrained Gemma 4 E2B audio tower, the compatible trained
384-wide GQASR bridge, and the trained GQASR 0.6B decoder. **Stride 1** retains
approximately 25 audio embeddings/second. Every imported tensor is checked;
an incompatible bridge fails rather than partially loading. `AUDFLASH_BRIDGE_INIT=new`
explicitly requests a fresh bridge if needed.

Default source is `google/gemma-4-E2B-it`; set AUDFLASH_ENCODER_SOURCE to
`google/gemma-4-E4B-it` to use that model's audio weights instead. The E2B/E4B
labels describe the parent models, not the extracted tower's parameter count.
AUDFLASH_ENCODER_REVISION can pin a revision; the resolved commit is recorded.
Ensure your Hugging Face account has access to the selected Gemma weights.
Export runs on CPU; verification uses cuda:0 for strict reload and generation.

Use the new Gemma export/run paths above, including when replacing an old env
file. Do not resume a Moonshine checkpoint or reuse its adapter output. Existing
export destinations are protected; skip export only for an already verified
Gemma stride-1 artifact. The launcher rejects other architectures/strides.
Preflight scans the actual corpus and calculates coverage updates.

## 3. NCCL and two-update rehearsal

```bash
bash audm/scripts/audflash.sh nccl "$NODE_RANK"
bash audm/scripts/audflash.sh smoke "$NODE_RANK"
```

Expected NCCL result: world_size 8, sum 36.0, nccl passed. The adapter rehearsal
writes under `$AUDFLASH_RUN_ROOT/adapter/smoke-test`. Check finite losses and GPU
memory before continuing. A repeated rehearsal needs its previous smoke output
archived first. Adapter memory does not establish full-training memory usage.

## 4. Adapter alignment

```bash
bash audm/scripts/audflash.sh adapter "$NODE_RANK"
```

Encoder and decoder are frozen while the initialized bridge realigns for 2,000 updates,
with learning rate 2e-4, warmup 100, and save/eval every 500 updates. This is
1.28 million sample presentations at the configured global batch. Final adapter
model is saved at `$AUDFLASH_RUN_ROOT/adapter`. Wait for completion and inspect
per-language quality before moving to full training.

## 5. Three-epoch full training

```bash
bash audm/scripts/audflash.sh preflight full --scan
bash audm/scripts/audflash.sh full "$NODE_RANK"
```

Full training loads the saved adapter model and unfreezes encoder, adapter and
decoder. It uses three weighted coverage epochs, base LR 2e-5, decoder factor
0.5, embedding factor 0.25, warmup 500, cosine schedule and save/eval every
1,000 updates. Final model is saved at `$AUDFLASH_RUN_ROOT/full`. Launcher logs
are under `$AUDFLASH_RUN_ROOT/launcher-logs`; W&B names are
`audflash-adapter-1node` and `audflash-full-1node`.

To resume an interrupted stage with unchanged settings:

```bash
bash audm/scripts/audflash.sh adapter "$NODE_RANK" --resume auto
# Or, for an interrupted full run:
bash audm/scripts/audflash.sh full "$NODE_RANK" --resume auto
```

The 20-sample microbatch is the requested configuration, not a measured H100
memory guarantee. Local tests validate models, recipes and launcher arguments;
actual memory and throughput must be checked on the cluster.

## Audit Malayalam transcript lengths (CPU only)

Run once in the AUD environment; no GPU, audio decoding, or model weight loading
is needed. The wrapper selects the exported AudFlash tokenizer and the Malayalam
training manifests in the resolved adapter recipe. It streams batched tokenization
with eight tokenizer threads by default (`RAYON_NUM_THREADS` overrides this).

```bash
bash audm/scripts/audit_transcripts.sh --output output/ml-token-audit
```

The new output directory contains `report.json` (total/per-manifest counts and
percentages strictly above 512 and 1,024, source hashes, tokenizer fingerprint)
and `over_limit.jsonl` (original manifest/line, path, text and exact token count).
It reports both all usable transcripts and duration-eligible transcripts under
the recipe's 30-second limit. Unknown durations are included but counted separately;
audio validity and actual duration are not checked. These are manifest-row counts,
not unique recording counts. Blank/punctuation-only transcripts are counted separately.

To scan evaluation data, add `--split eval`. To select another export/checkpoint,
add `--model /path/to/export`. Counts exclude prompt markers and special tokens,
matching the trainer's transcript validation. Malformed JSON/records fail the scan;
only a completed scan writes `report.json`. Output directories are never overwritten.

After reviewing the counts, optionally generate new filtered manifest copies:

```bash
bash audm/scripts/audit_transcripts.sh \
  --output output/ml-token-filter-512 --filter-above 512
```

Filtering removes only tokenized transcripts above the threshold. It preserves
other metadata and normalizes retained usable records' audio paths so relocating
manifests does not break relative paths. Existing unusable-text records are retained
and will still be skipped by training. The originals and all training configs remain
unchanged; `filtered_path` in the report identifies each new copy. Do not replace
manifests under a running job. Using filtered data requires a deliberate new run/data
configuration, not exact resume of a run with different manifests.
