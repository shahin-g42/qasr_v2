# QASR — Cohere-Conformer × Qwen3 Hybrid ASR

A 3.62B-parameter speech recognition model combining Cohere's 48-layer Conformer
encoder (1.9B params, 1280-dim, 8× subsampling) with a Qwen3-1.7B decoder-only LLM
(1.72B params, 28 layers, 2048-dim, 151,936 vocab). Audio embeddings are injected
at `<|audio|>` token positions via a learned multimodal projector (4.3M params,
LLaVA-style).

```
┌─────────────────────────────────────────────────────────────────────┐
│  16 kHz waveform                                                    │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────┐    128-bin log-Mel, 8× subsample          │
│  │  Cohere Conformer    │───────────────────────────────┐           │
│  │  (48 layers, 1280d)  │                               │           │
│  └──────────────────────┘                               ▼           │
│                                              ┌──────────────────┐   │
│                                              │  MM Projector    │   │
│                                              │  1280→2048 (MLP) │   │
│                                              └────────┬─────────┘   │
│                                                       │             │
│       ┌───────────────────────────────────────────────┘             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Qwen3 Decoder (28 layers, 2048d, 151,936 vocab)            │   │
│  │  [<|audio|> × N audio frames] [prompt tokens] → transcript  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  Transcription (Arabic, English, Chinese, Hindi, Malayalam)         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline Overview

The pipeline is structured as **progressive validation stages** — each step
catches a specific class of failures before committing expensive GPU-hours:

```
Step 0  Environment Setup         ~10 min   Catches: missing deps, wrong versions
Step 1  Environment Validation    ~2 min    Catches: cluster, NCCL, FS issues
Step 2  Weight Conversion         ~10 min   Catches: architecture mismatch
Step 3  Dataset Validation        ~30 min   Catches: corrupt/missing audio, bad transcripts
Step 4  Smoke Test (2 steps)      ~5 min    Catches: data, config, OOM issues
Step 5  Phase 1: Projector        ~3 hrs    Catches: training dynamics
Step 6  Phase 2: Full Fine-tune   ~5 days   The main training run
Step 7  Phase 3: EAGLE Head       ~1-2 hrs  Inference acceleration (optional)
```

**Total GPU-hours:** ~2,500 (Phases 1+2) + ~100 (Phase 3)

---

## Upgrading After a Code Sync

If you already have a working environment and converted checkpoint, you do **NOT**
need to re-run from Step 0. Code changes (bug fixes, branding, new features) only
require a package reinstall:

```bash
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr

# Pull / rsync the latest code, then:
pip install -e ".[deepspeed,streaming,dev]"

# Verify the updated package:
python -c "
from qasr import QASRProcessor, QASRFeatureExtractor, CohereEncoderConfig
print('Package OK')
"
```

**When to re-run each step:**

| Scenario | Required Steps |
|----------|---------------|
| Code bug fix / feature update | Reinstall only → resume training |
| New training data added | Step 3 (validate) → Step 4 (smoke) → resume |
| Config YAML changes | Step 4 (smoke test) → resume |
| New encoder/decoder model | Step 2 (convert) → Step 4 → Step 5+ |
| Fresh cluster / new env | Step 0 → Step 1 → Step 2 → ... |

**Checkpoint compatibility:** All QASR checkpoints are forward-compatible.
Branding changes (Parakeet → Cohere) do not affect saved weights or configs.
The `model_type: "parakeet_encoder"` in saved configs is the transformers
library identifier and remains unchanged on disk.

---

## Step 0: Environment Setup & Dependencies

### Cluster Requirements

| Resource | Specification |
|----------|--------------|
| Nodes | 4 × 8 GPU (H100 80GB or A100 80GB) |
| Interconnect | InfiniBand (NCCL IB enabled) |
| Filesystem | Shared (Lustre/GPFS) accessible from all nodes |
| RAM | ≥ 512 GB per node |
| Storage | ~2 TB for checkpoints + data |

### Complete Installation (run once on shared FS)

```bash
# 1. Create conda environment on shared filesystem
conda create -p /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3 python=3.11 -y
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3

# 2. Install PyTorch (match your CUDA version — check with: nvidia-smi)
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126

# 3. Clone/copy the project to shared FS
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr

# 4. Install QASR with all dependencies
pip install -e ".[deepspeed,streaming,dev]"

# 5. Verify all critical imports
python -c "
import torch
import transformers
import deepspeed
import accelerate
import soundfile
import librosa
import scipy
import numpy
import yaml
import wandb
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA:          {torch.version.cuda}')
print(f'GPUs:          {torch.cuda.device_count()}')
print(f'Transformers:  {transformers.__version__}')
print(f'DeepSpeed:     {deepspeed.__version__}')
print(f'Accelerate:    {accelerate.__version__}')
print(f'BF16 support:  {torch.cuda.is_bf16_supported()}')
print(f'Flash/SDPA:    {torch.backends.cuda.flash_sdp_enabled()}')
print('All imports OK')
"

# 6. Pre-cache models on shared FS (avoids per-node downloads)
export HF_HOME=/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/.cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('CohereLabs/cohere-transcribe-03-2026')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('audarai/Audar-ASR-V1.2-Turbo')"

# 7. Login to WandB (for training monitoring)
wandb login
```

### Required Models

| Model | Source | Purpose |
|-------|--------|---------|
| Cohere Transcribe | `CohereLabs/cohere-transcribe-03-2026` | Conformer encoder weights |
| Qwen3-ASR-1.7B-hf | `audarai/Audar-ASR-V1.2-Turbo` | Decoder + processor + tokenizer |

### Verify on Every Node

```bash
# Run on each of the 4 nodes to confirm environment is visible:
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/qasr3
python -c "import torch; print(f'{torch.cuda.device_count()} GPUs, CUDA OK')"
ls /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/src/qasr/__init__.py
```

---

## Step 1: Environment Validation

**Catches:** NCCL connectivity, shared filesystem, GPU visibility, conda activation.

Run this **before any training** to avoid wasting hours on a broken cluster.

```bash
# Verify all 4 nodes can see each other and have GPUs
srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=8 \
  bash -c 'echo "$(hostname): $(nvidia-smi -L | wc -l) GPUs, FS=$(ls /lustrefs/shared/ | head -1)"'

# Verify NCCL all-reduce across nodes (2-minute test)
srun --nodes=4 --ntasks-per-node=1 --gpus-per-node=8 \
  python -c "
import torch, torch.distributed as dist
dist.init_process_group('nccl')
t = torch.ones(1024, device='cuda')
dist.all_reduce(t)
assert t.sum().item() == 1024 * dist.get_world_size()
if dist.get_rank() == 0:
    print(f'NCCL OK: {dist.get_world_size()} ranks, all-reduce verified')
dist.destroy_process_group()
"
```

**Success criteria:**
- All 4 nodes report 8 GPUs
- NCCL all-reduce completes without timeout
- Shared FS path is accessible from every node

**Common failures:**

| Symptom | Fix |
|---------|-----|
| `NCCL WARN Cuda failure` | Check `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| `Connection timed out` | Firewall blocking port 29500; try `MASTER_PORT=29501` |
| `No such file or directory` | Shared FS not mounted on compute nodes |
| `conda: command not found` | Add `module load conda` or use full path |

---

## Step 2: Weight Conversion

**Catches:** Architecture mismatches, missing tensors, config errors.

Builds the initial QASR checkpoint by merging Cohere encoder weights with
Qwen3 decoder weights and initializing the multimodal projector randomly.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr
export PYTHONPATH=src
export HF_HOME=/lustrefs/shared/shahin.konadath/workspace/train/stt/qasr/.cache

python -m qasr.convert_weights \
  --encoder CohereLabs/cohere-transcribe-03-2026 \
  --qwen audarai/Audar-ASR-V1.2-Turbo \
  --output-dir /lustrefs/shared/mohammed.naseem/workspace/expmt/qasr/initial \
  --verify-reload
```

**Success criteria:**
- `--verify-reload` passes (reloads the saved checkpoint and asserts the
  embeddings stayed tied; per-tensor equality is checked during the weight
  transfer itself, not on reload)
- Output contains: `model.safetensors`, `config.json`, `preprocessor_config.json`, tokenizer files

---

## Step 3: Dataset Validation & Filtering

**Catches:** Corrupt audio, missing files, bad transcripts, duration mismatches —
BEFORE they cause NaN losses or crashes during training.

This is a **critical step** that prevents mid-training failures. It decodes every
audio file, checks duration, validates transcripts, and produces a filtered
manifest with only clean samples.

```bash
export PYTHONPATH=src

# Validate ALL manifests in the training config (uses 64 parallel workers):
python -m qasr.validate_data \
  --config configs/train_full_4node.yaml \
  --output-dir /lustrefs/shared/shahin.konadath/workspace/data/filtered/ \
  --workers 64

# Or validate a single manifest:
python -m qasr.validate_data \
  --manifest /path/to/train.json \
  --output /path/to/train_filtered.json \
  --min-duration 0.1 \
  --max-duration 35.0 \
  --max-target-length 512 \
  --workers 32
```

**What this validates per sample:**
- [ ] Audio file exists on disk
- [ ] Audio file is non-zero size
- [ ] Audio decodes without errors (WAV/FLAC/OGG)
- [ ] Audio contains no NaN/Inf values
- [ ] Actual duration is within [0.1s, 35.0s] range
- [ ] Transcript is non-empty and contains alphanumeric characters
- [ ] Transcript is not excessively long for max_target_length

**Expected output:**
```
======================================================================
 DATASET VALIDATION REPORT
======================================================================

  Manifest: /lustrefs/.../train_filter_ar.json
  Total: 245,000 | Valid: 243,891 (99.5%) | Invalid: 1,109
  Valid hours: 1,245.3h | Lost hours: 4.2h
  Errors:
    file_not_found: 847
    decode_error: 156
    too_short: 89
    punctuation_only_text: 17

----------------------------------------------------------------------
 TOTALS:
   Records: 512,000 total | 509,847 valid | 2,153 invalid
   Hours:   3,412.8h valid | 8.1h lost
   Health:  99.6% samples usable
======================================================================
```

**After validation:** Update your training config YAML to point to the filtered
manifests in `/lustrefs/shared/shahin.konadath/workspace/data/filtered/`.

**When to worry:**
- Health < 95%: Check filesystem mounts and path prefixes
- Many `decode_error`: Corrupt audio batch; investigate source
- Many `too_short`/`too_long`: Adjust duration filters or re-segment data

---

## Data Cleaning & Manifest Pipeline

Upstream of validation sits the manifest supply chain. `src/data_processing`
is an LLM-based transcript-cleaning pipeline: it slices raw manifests into
record ranges, sends each transcript to a local vLLM server (OpenAI-compatible
API) for correction against the audio's language conventions, validates and
audits the results, and writes cleaned manifests. Its configs live in
`configs/data_processing/` and its SLURM launchers are
`scripts/qasr_clean_arabic.slurm` / `scripts/qasr_reprocess.slurm`.

The cleaned outputs are then assembled into versioned training sets by
`scripts/prepare_q3asr_filter.py`, `scripts/assemble_training_manifests.py`,
`scripts/audit_training_manifests.py`, and `scripts/backfill_durations.py`,
which together produce `training_manifests/vN/` (currently `v7.6/`). The
`configs/v7.6/` training configs consume these manifests directly and use
`sampling_strategy: stratified` — equal language share per batch, with every
record of every language seen at least once per epoch (smaller languages
repeat to fill their share).

---

## Step 4: Smoke Test

**Catches:** Data loading errors, OOM, config typos, DeepSpeed issues — in 2 training steps.

This is the **most important validation step**. It runs 2 training steps + 1 eval
on a tiny subset (256 train, 16 eval samples) to verify the entire pipeline works
end-to-end before committing to a multi-day run.

```bash
# Manual per-node (add --smoke-test on ALL 4 nodes):
# See qasr_train_4node_manual.txt for full per-node commands

# On each node, the training command becomes:
torchrun ... train.py --config configs/train_projector_4node.yaml --smoke-test
```

**What the smoke test validates:**
- [ ] YAML config parses correctly
- [ ] All manifest JSONL files are readable and well-formed
- [ ] Audio files decode at 16 kHz without errors
- [ ] Feature extraction produces correct shapes (128 mel bins)
- [ ] Collator pads batches correctly
- [ ] Model forward pass completes (no shape mismatches)
- [ ] Loss is finite and non-zero
- [ ] Backward pass + optimizer step complete
- [ ] DeepSpeed ZeRO-3 partitions correctly across GPUs
- [ ] Gradient checkpointing works with ZeRO-3
- [ ] **Augmentation pipeline runs** (SpecAugment, SpeedPerturb, Noise, Codec)
- [ ] Eval generation produces text output
- [ ] Checkpoint save/load works

**Success criteria:**
```
INFO | qasr | Trainable parameters: 4,263,168/3,620,177,408 (0.12%)
INFO | qasr | Global batch size: 128 (32 devices x 4 samples x 1 accumulation)
INFO | qasr | Audio augmentation enabled: ['spec_augment', 'speed_perturb', 'noise_injection', 'codec_augment']
{'loss': 12.4523, 'grad_norm': 3.21, ...}   ← Step 1
{'loss': 11.8901, 'grad_norm': 2.87, ...}   ← Step 2
INFO | qasr | Evaluation example 1/2
  Ground truth: مرحبا كيف حالك
  Prediction: [some text]                     ← Generation works
```

**WER/CER scoring:** Offline evaluation and the training WER callback preserve
diacritics (including Arabic harakat and Indic vowel signs), punctuation, case,
and number spellings. Only Unicode NFC and whitespace collapsing are applied.
WER compares whitespace-separated tokens; CER compares Unicode code points,
excluding whitespace. Use references that follow the same cleaning policy as
the training targets. Scores from the previous punctuation/diacritic-stripping,
casefolding scorer are not directly comparable; re-score those predictions or
rerun evaluation to establish the new baseline.

**Common failures:**

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: manifest.json` | Check paths in YAML; ensure shared FS mounted |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` to 2 |
| `RuntimeError: shape mismatch` | Encoder/decoder config mismatch; re-run Step 2 |
| `KeyError: 'audio_filepath'` | JSONL format wrong; needs `{"audio_filepath": ..., "text": ..., "duration": ...}` |
| `NCCL timeout during backward` | Increase `NCCL_TIMEOUT` or check IB links |
| Loss = `nan` | Corrupt audio; run Step 3 (data validation) first |

---

## Step 5: Phase 1 — Projector Training

**Purpose:** Train only the multimodal projector (4.3M params) to align audio
embeddings with the LLM's input space. Encoder and decoder are frozen.

**Duration:** ~3 hours on 32 GPUs (5,000 steps)

```bash
# Manual per-node launch (see qasr_train_4node_manual.txt for full commands):
# Run on ALL 4 nodes with --node_rank=0,1,2,3 respectively

torchrun \
  --nnodes=4 --nproc_per_node=8 --node_rank=<N> \
  --master_addr="inception-H100-hpc-001" --master_port=29500 \
  --rdzv_backend=static --rdzv_endpoint="inception-H100-hpc-001:29500" \
  --rdzv_conf=timeout=7200 \
  train.py --config configs/train_projector_4node.yaml
```

### Configuration Summary

| Parameter | Value |
|-----------|-------|
| Frozen | Encoder + Decoder (99.88% of params) |
| Trainable | Projector only (4.3M params) |
| Steps | 5,000 |
| Global batch | 128 (32 GPUs × 4 × 1 accum) |
| Learning rate | 1e-4, cosine, 500 warmup |
| Precision | BF16 + TF32 |
| DeepSpeed | ZeRO-3, no CPU offload |
| Augmentation | **All enabled** at p=0.37 (same recipe as Phase 2) |
| Eval | Every 2,500 steps |
| Checkpoints | Every 2,500 steps (keep 5) |

### Success Criteria

- Loss decreases smoothly from ~12 to ~4
- Eval examples show recognizable transcription (even if imperfect)
- No NaN/Inf in loss or gradients
- Checkpoint saved at `projector/checkpoint-5000`

---

## Step 6: Phase 2 — Full Fine-tuning

**Purpose:** Unfreeze all parameters and fine-tune the entire model end-to-end
with data augmentation for robust multi-lingual ASR.

**Duration:** ~5 days on 32 GPUs (500,000 steps)

```bash
# Same per-node launch, different config:
torchrun ... train.py --config configs/train_full_4node.yaml

# Resume from checkpoint after crash:
torchrun ... train.py --config configs/train_full_4node.yaml \
  --resume-from-checkpoint /lustrefs/.../full/checkpoint-250000
```

### Configuration Summary

| Parameter | Value |
|-----------|-------|
| Frozen | Nothing (all 3.62B params trainable) |
| Steps | 500,000 |
| Global batch | 128 (32 GPUs × 4 × 1 accum) |
| Learning rate | 2e-5, cosine, 2,500 warmup |
| Augmentation | **All enabled** (see below) |
| Eval | Every 5,000 steps |
| Checkpoints | Every 5,000 steps (keep 5) |
| Input model | `projector/checkpoint-5000` |

### Data Augmentation (Verified Active)

All augmentations are configured in `configs/train_full_4node.yaml` and applied
by the `QASRDataCollator` during training:

| Augmentation | Stage | Parameters | Probability |
|-------------|-------|-----------|-------------|
| **SpecAugment** | Post-feature (on 128-bin mel) | 2 time masks (5%), 2 freq masks (15%) | p=0.37 |
| **Speed Perturbation** | Pre-feature (on waveform) | 0.85×–1.15× continuous | p=0.37 |
| **Noise Injection** | Pre-feature (on waveform) | Real MUSAN audio from `noise_dir`, SNR 5–20 dB (Gaussian fallback only if no noise dir resolves) | p=0.37 |
| **Codec Augmentation** | Pre-feature (on waveform) | Random bandpass + random 6–16-bit quantization, blended with clean signal at random alpha | p=0.37 |

**Verification:** The smoke test log should show:
```
INFO | qasr | Audio augmentation enabled: ['spec_augment', 'speed_perturb', 'noise_injection', 'codec_augment']
```

If this line is missing, augmentations are NOT active. Check:
1. `augmentation:` section exists in your YAML config
2. Each augmentation has `enabled: true`
3. Each augmentation has a non-zero `p` (both projector and full configs
   ship with all four enabled at p=0.37)

### Success Criteria

- Training loss: 12 → 0.3–0.8 over 500K steps
- Eval loss tracks training loss (no divergence)
- Eval predictions show accurate, punctuated transcriptions
- Model generalizes across all 5 languages (ar, en, zh, hi, ml)

---

## Step 7: Phase 3 — EAGLE-2 Speculative Decoding Head

**Purpose:** Train a lightweight 8M-parameter draft head for ~2× faster inference.

**Duration:** ~1–2 hours on 64 GPUs (8 nodes, 10,000 steps, batch 32/device,
warmup 200)

```bash
# 8-node launch via SLURM (see scripts/qasr_train_eagle_8node.slurm):
sbatch scripts/qasr_train_eagle_8node.slurm

# Or manual per-node (run on ALL 8 nodes with --node_rank=0..7):
torchrun --nnodes=8 --nproc_per_node=8 --node_rank=<N> \
  --master_addr="inception-H100-hpc-001" --master_port=29501 \
  --rdzv_backend=static --rdzv_endpoint="inception-H100-hpc-001:29501" \
  train_eagle.py --config configs/train_eagle_8node_filtered.yaml
```

### Success Criteria

- KL loss converges to < 0.5 within 10,000 steps
- Acceptance rate ≥ 0.15 with end-to-end speedup ≥ 1.5× on held-out audio.
  Acceptance is measured as accepted/(rounds × K) with K=5 draft tokens;
  the shipped head measures ~0.18 acceptance, 1.76 tokens per target
  forward, and ~2× speedup (see `eval_results_eagle_v2.json`)
- Output: `eagle/eagle_head.pt` (~639 MB on disk — `save_pretrained`
  serializes the frozen copy of the 311M-param `lm_head` alongside the
  ~16.8 MB of actually-trained weights)

---

## Training Speed Optimizations

These are already configured in the default configs. Verify they're active:

### Already Enabled (check your config)

| Optimization | Config Key | Effect |
|-------------|-----------|--------|
| **BF16 mixed precision** | `bf16: true` | 2× memory savings, 1.5× speedup on H100 |
| **TF32 matmul** | `tf32: true` | 3× faster FP32 matmuls on Ampere+ |
| **SDPA attention** | `attn_implementation: sdpa` | Flash-attention-like speedup |
| **Gradient checkpointing** | `gradient_checkpointing: true` | 60% less memory → larger batches |
| **ZeRO-3 overlap comm** | `deepspeed_zero3.json: overlap_comm` | Overlap gradient sync with compute |
| **Persistent workers** | `dataloader_persistent_workers: true` | Avoid worker respawn overhead |
| **Pin memory** | `dataloader_pin_memory: true` | Faster CPU→GPU transfer |
| **8 dataloader workers** | `dataloader_num_workers: 8` | Parallel audio decode + augmentation |

### Additional Speed Tips

```bash
# 1. Ensure NCCL uses InfiniBand (not TCP):
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# 2. Disable tokenizer parallelism (avoids fork warnings):
export TOKENIZERS_PARALLELISM=false

# 3. Set OMP threads to 1 (avoid thread contention with dataloader):
export OMP_NUM_THREADS=1

# 4. If data loading is the bottleneck (GPU util < 80%):
#    Increase dataloader_num_workers to 12-16 in config
#    Ensure audio files are on fast storage (NVMe/Lustre, not NFS)

# 5. For maximum throughput, increase batch size if memory allows:
#    per_device_train_batch_size: 6  (if GPU memory > 60GB free)
#    This reduces total steps proportionally

# 6. Pre-validate data (Step 3) to avoid runtime skips that stall the pipeline
```

### Expected Throughput

| Setup | Samples/sec | Steps/hour | Time for 500K steps |
|-------|------------|-----------|---------------------|
| 32× H100, batch 4/GPU | ~128 | ~3,600 | ~5.8 days |
| 32× A100, batch 4/GPU | ~96 | ~2,700 | ~7.7 days |
| 32× H100, batch 6/GPU | ~192 | ~3,600 | ~3.9 days |

---

## Inference & Deployment

### CLI Transcription

```bash
# Standard decoding
PYTHONPATH=src python inference.py audio.wav \
  --model /lustrefs/shared/shahin.konadath/workspace/expmt/qasr/full

# With EAGLE speculative decoding (~2× faster)
PYTHONPATH=src python inference.py audio.wav \
  --model /lustrefs/shared/shahin.konadath/workspace/expmt/qasr/full \
  --eagle /lustrefs/shared/shahin.konadath/workspace/expmt/qasr/eagle \
  --num-draft-tokens 5
```

### Streaming WebSocket Server

```bash
# With EAGLE (partials arrive ~200ms sooner)
PYTHONPATH=src python -m qasr.server \
  --model /lustrefs/shared/shahin.konadath/workspace/expmt/qasr/full \
  --eagle /lustrefs/shared/shahin.konadath/workspace/expmt/qasr/eagle \
  --port 8765
```

---

## Publishing to HuggingFace

Once Phase 2 (and optionally Phase 3) are done, `scripts/package_and_push.py`
assembles a **self-contained release** and pushes it to the Hub. Run it on the
cluster where the weights live.

The staged release contains:

- Model weights + configs + tokenizer/processor artifacts from `--model-dir`
  (training artifacts — optimizer/scheduler states, `trainer_state.json`,
  `global_step*`, intermediate `checkpoint-*` dirs — are excluded automatically)
- The 6 remote-code modules (`configuration.py`, `modeling.py`,
  `feature_extraction.py`, `processing.py`, `utils.py`, `eagle.py`) with
  `auto_map` injected into the JSON configs, so consumers can load the model
  with plain `transformers` + `trust_remote_code=True` — no `qasr` install needed
- The EAGLE draft head at `eagle/eagle_head.pt`
- A polished model card (rendered from `scripts/HF_MODEL_CARD.md`) as the repo `README.md`

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/qasr
huggingface-cli login          # once; needs a write token for the audarai org

# 1. Dry-run: assemble + verify the staging dir locally, no upload
python scripts/package_and_push.py --no-upload --verify
ls output/hf_release/          # inspect what would be uploaded

# 2. Push (defaults: private repo audarai/Audar-ASR-V1-Pro,
#    model output/full, EAGLE head output/eagle_v2/checkpoint-10000)
python scripts/package_and_push.py --verify
```

Useful flags: `--repo-id`, `--model-dir`, `--eagle-dir` (`none` to skip EAGLE),
`--staging-dir`, `--public`, `--commit-message`. Re-running is idempotent — the
staging dir is rebuilt from scratch and `upload_folder` only commits changes.

Consumers then load it exactly as documented in the published model card:
offline via `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`,
EAGLE via `EagleSpeculativeDecoder.from_pretrained(local, f"{local}/eagle")`,
and streaming via `qasr-stream --model "$LOCAL" --eagle "$LOCAL/eagle"`.

---

## Project Structure

```
qasr/
├── configs/
│   ├── data_processing/              # LLM manifest-cleaning configs (Arabic + multilingual)
│   ├── v7.6/                         # Current recipe: 01_projector / 02_full / 03_hq /
│   │                                 #   04_eagle (+ README), all 8-node
│   ├── deepspeed_zero2.json          # ZeRO-2 config
│   ├── deepspeed_zero3.json          # ZeRO-3 config (no CPU offload)
│   ├── train_projector_*.yaml        # Legacy Phase 1 configs (4node, 4node_filtered, 8node)
│   ├── train_full_*.yaml             # Legacy Phase 2 configs (4node, 4node_filtered,
│   │                                 #   8node_filtered)
│   ├── train_hq_8node.yaml           # Legacy HQ fine-tune config
│   └── train_eagle_8node_filtered.yaml  # Phase 3: EAGLE head (10K steps, 8-node)
├── scripts/
│   ├── HF_MODEL_CARD.md              # Model card template for the Hub release
│   ├── assemble_training_manifests.py  # Build training_manifests/vN/ from cleaned outputs
│   ├── audit_training_manifests.py   # Audit assembled manifest sets
│   ├── backfill_durations.py         # Backfill missing durations in manifests
│   ├── package_and_push.py           # Package checkpoint + EAGLE head → push to HF Hub
│   ├── prepare_q3asr_filter.py       # Prepare/filter q3asr SFT manifests
│   ├── qasr_train_4node.slurm        # 4-node SLURM (if sbatch available)
│   ├── qasr_train_8node.slurm        # 8-node SLURM
│   ├── qasr_train_eagle.slurm        # Legacy single-node SLURM (Phase 3)
│   └── qasr_train_eagle_8node.slurm  # 8-node SLURM (Phase 3: EAGLE head)
├── src/data_processing/              # LLM-based transcript cleaning pipeline (vLLM client,
│                                     #   validators, audit/reporting)
├── src/qasr/
│   ├── __init__.py                   # Package exports
│   ├── audio.py                      # Audio I/O utilities
│   ├── augmentation.py               # SpecAugment, SpeedPerturb, Noise, Codec
│   ├── collator.py                   # Batch collation + augmentation application
│   ├── config.py                     # TrainConfig dataclass + YAML parser
│   ├── configuration.py              # QASRConfig (HF model config)
│   ├── convert_weights.py            # Cohere encoder + Qwen3 → QASR checkpoint
│   ├── data.py                       # JSONL dataset, resilient wrapper
│   ├── eagle.py                      # EAGLE-2 head + speculative decoder
│   ├── feature_extraction.py         # 128-bin log-Mel, preemphasis, dither
│   ├── modeling.py                   # QASRModel, QASRForConditionalGeneration
│   ├── processing.py                 # QASRProcessor (tokenizer + features)
│   ├── server.py                     # FastAPI WebSocket streaming server
│   ├── streaming.py                  # PCM16 buffer, rolling-window transcriber
│   ├── train.py                      # Main training loop (HF Trainer)
│   ├── train_eagle.py                # EAGLE head training loop
│   ├── validate_data.py              # Dataset validation & filtering
│   ├── utils.py                      # Shared utilities
│   └── static/                       # Browser demo UI (HTML/JS/CSS)
├── tests/                            # Unit + integration tests
├── training_manifests/               # Assembled training manifest sets (vN/)
├── inference.py                      # CLI inference entry point
├── train.py                          # Root training entry point
├── train_eagle.py                    # Root EAGLE training entry point
├── pyproject.toml                    # Package metadata + dependencies
├── qasr_train_4node.txt              # Quick-reference (SLURM commands)
├── qasr_train_4node_manual.txt       # Per-node manual launch (no sbatch)
└── README.md                         # This file
```

---

## Data Format

Training manifests are JSONL files where each line is:

```json
{"audio_filepath": "/path/to/audio.wav", "text": "مرحبا كيف حالك", "duration": 3.2}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_filepath` | str | Absolute path to WAV/FLAC/OGG file |
| `text` | str | Ground-truth transcription |
| `duration` | float | Duration in seconds (used for filtering) |

---

## Troubleshooting

### Training Won't Start

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Hangs at `Initializing process group` | NCCL can't find IB | `export NCCL_IB_DISABLE=0; export NCCL_NET_GDR_LEVEL=2` |
| `Address already in use` | Port conflict | Change `MASTER_PORT` to 29501 |
| `ModuleNotFoundError: qasr` | PYTHONPATH not set | `export PYTHONPATH=$(pwd)/src` |
| `torch.cuda.OutOfMemoryError` | Batch too large | Reduce `per_device_train_batch_size` |
| `KeyError: 'attention_mask'` in processor | Checkpoint saved with native `CohereAsrFeatureExtractor` | Update code (fixed in `_process_audio` override) and reinstall |
| `model of type 'qasr' to instantiate type ''` warning | Benign transformers auto-mapping notice | Safe to ignore; use `QASRForConditionalGeneration.from_pretrained()` directly |

### Training Instability

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| Loss = NaN | Corrupt audio or overflow | Run Step 3 (data validation); check `max_grad_norm` |
| Loss spikes periodically | Bad samples in data | `ResilientAudioDataset` auto-skips; check logs |
| Loss plateaus | LR too low or data exhausted | Increase LR or add more manifests |
| Eval loss diverges | Overfitting | Stop early; use last good checkpoint |

### Multi-Node Issues

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| One node slower | IB link degraded | `ibstat` on each node; report to admin |
| Rendezvous timeout | Node not joining | Check all 4 nodes started within 2hr window |
| Checkpoint corrupt | FS sync issue | Use `save_total_limit: 5`; verify with `--verify-reload` |

---

## Quick Reference

```bash
# Full pipeline from scratch:
python -m qasr.convert_weights --encoder CohereLabs/cohere-transcribe-03-2026 --qwen audarai/Audar-ASR-V1.2-Turbo --output-dir .../initial --verify-reload
python -m qasr.validate_data --config configs/train_full_4node.yaml --output-dir filtered/ --workers 64
# Then launch smoke test on all 4 nodes (see qasr_train_4node_manual.txt)
# Then launch Phase 1 on all 4 nodes
# Then launch Phase 2 on all 4 nodes
# Then launch Phase 3 on 1 node

# Per-node manual commands:
cat qasr_train_4node_manual.txt
```
