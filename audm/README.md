# AUD models and training

For the **new AudFlash run from checkpoint-3700**, auxiliary CTC, stride-one
architecture, 40 × 2 × 8 training, and direct cluster commands, use
[NEXT_RUN_CLUSTER.md](NEXT_RUN_CLUSTER.md) and `scripts/audflash_next.sh`.

For the exact AudTurbo-only, per-node cluster launch sequence, see
[AUDTURBO_CLUSTER.md](AUDTURBO_CLUSTER.md) and `scripts/audturbo.sh`.

For the ordered AudFlash token-upgrade → decoder warm-up → full-training steps,
including the OpenSLR RIR downloader (`scripts/rirs.sh`), see
[ROBUST_CLUSTER.md](ROBUST_CLUSTER.md).

For the pinned Hugging Face non-speech download, optional listening review and training
manifests, see [NON_SPEECH_DATA.md](NON_SPEECH_DATA.md) and `scripts/non_speech.sh`.

For the OOM audit, bounded-memory loss, AudFlash microbatch-10 defaults and
longest-shape cluster rehearsal, see [MEMORY_AUDIT.md](MEMORY_AUDIT.md).

`audm` exports and fine-tunes three complete speech models. It uses the **trained
local QASR/GQASR decoders**, not fresh upstream replacements.

| Model | Audio encoder | Text decoder | Exported bridge | First training step |
|---|---|---|---|---|
| **AudPro** | Current QASR Cohere Conformer | Current QASR | Preserve trained bridge, depth and pooling | Optional full fine-tuning |
| **AudTurbo** | `Qwen/Qwen3-Omni-30B-A3B-Captioner` audio tower, including output projections | Current QASR | New residual bottleneck; initially identity when widths match | Adapter alignment, then full fine-tuning |
| **AudFlash** | `google/gemma-4-E2B-it` audio tower and Gemma log-mel frontend | Current GQASR | Reuse compatible trained GQASR LayerNorm → 384 → GELU → decoder-width projection | Adapter alignment, then full fine-tuning |

Captioner's published audio tower has 32 layers, width 1280 and output width
2048 (approximately **648M encoder parameters**). Gemma 4 E2B's audio tower
has 12 layers, width 1024, output width 1536 and approximately **305M parameters**. The
exporter writes actual component counts from the selected checkpoints into
`provenance.json`. Total model size also includes the retained decoder: with the
current approximately 1.7B/0.6B decoders, Turbo is roughly 2.4B total and Flash
roughly 0.9B. Names express the intended product tiers; speed and ASR accuracy
must be measured after training.

Sources: [Captioner configuration](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/blob/main/config.json),
[Captioner frontend](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner/blob/main/preprocessor_config.json),
[Gemma 4 E2B configuration](https://huggingface.co/google/gemma-4-E2B-it/blob/main/config.json).

Turbo and Flash are **new hybrids that require alignment training**. Matching
tensor dimensions does not make an encoder compatible with a trained decoder.
The full source encoder and decoder weights are loaded; bridge tensor provenance
is recorded explicitly. Flash reuses a compatible trained GQASR bridge by default. The existing
QASR bridge cannot be reused with Captioner representations.

## Artifact format and namespace

Each export is one directory containing **one `model.safetensors`**, `config.json`,
tokenizer assets, audio frontend config, generation config, source notices,
`provenance.json`, and SHA-256 checksums. It has no runtime dependency on the
source checkpoint directories or Hub repositories. Install `audm` and its
Python dependencies to load it offline; this is a weight artifact, not a
standalone executable or a GGUF/ONNX conversion.

All tensor keys start with `aud_`: `aud_encoder`, `aud_decoder`, `aud_projector`,
optional `aud_pool` and `aud_ctc_head`. Public architecture-specific layer classes
start with `AUD`. Underlying kernels and source attribution are retained.

AudPro preserves optional trained pooling/deep bridge/CTC weights. The optional
CTC head remains an inactive frozen scaffold in this training platform; a
nonzero CTC loss setting fails explicitly. Turbo keeps Captioner's pretrained
`proj1`/`proj2` inside its encoder. Flash defaults to Gemma's approximately 25 Hz output
(stride 1) and does not import Gemma's text decoder. E4B audio weights are also
supported via --encoder-source google/gemma-4-E4B-it.

Audio must be mono, 16 kHz. The trainer's QASR loader handles decoding and
resampling. Direct `AUDProcessor` callers pass already-resampled arrays.
AudPro uses the original normalized Cohere log-mel frontend; Turbo uses the
Captioner Whisper frontend; Flash uses Gemma log-mel features with a 30-second limit. Limits are explicit and
long utterances are rejected rather than silently truncated.

Legacy Moonshine exports remain loadable. The AUD Moonshine wrapper corrects upstream convolution-mask propagation for
unequal padded batches. Only valid time positions reach attention and the text
decoder. Tests compare batched results with native single-utterance results.
This package does not implement Moonshine's native C++ streaming runtime.

## Install

Run from the QASR repository root in your CUDA-capable Python environment:

```bash
python -m pip install -e . -e './audm[train,dev]'
python -m audm --help
```

The root project supplies the existing QASR training infrastructure. Export and
inference alone need only `pip install -e ./audm`; training also needs the root
package. Transformers is constrained to `>=5.14.1,<5.15`, matching the inspected
APIs. The root project pins PyTorch 2.10.0. Local tests ran with PyTorch 2.13.0
and Transformers 5.14.1; rehearse the installed CUDA/DeepSpeed combination on the
cluster before a long run. No changes to the root package dependencies are
required by this package.

Use the same environment, shared checkout, model directories and manifest paths
on every training node. Select your scheduler account, partition and time limit
through `sbatch` options. The supplied Slurm files request the node/GPU/CPU shape;
they do not assume your site's account or partition names.

## Export: the separate 8×H100 / 96-CPU node

```bash
export AUD_REPO_ROOT=/shared/path/to/qasr
export QASR_CHECKPOINT=/shared/path/to/trained/qasr/checkpoint
export GQASR_CHECKPOINT=/shared/path/to/trained/gqasr/checkpoint
export AUD_EXPORT_ROOT=/shared/path/to/new/aud-models

cd "$AUD_REPO_ROOT"
bash audm/scripts/export_all.sh
# Or submit the same job:
# sbatch audm/scripts/export.slurm
```

Exports run sequentially on CPU, followed by offline forward/generation smoke
checks on `cuda:0`. Eight GPU workers must not independently export the same
files. Allocate at least 32 GB host RAM as a starting provision, plus disk space
for selected source shards, all three artifacts and temporary output. Actual
peak memory depends on source tensor layouts; no full Captioner model is
instantiated.

The exporter resolves Hub `main` to an immutable revision and reads the tensor
index before downloading weight shards. It downloads only shards containing
the requested encoder tensors; a selected shard can also contain unrelated
weights. A source distributed as a single safetensors file requires that file.
Neither the Captioner MoE decoder nor Gemma's text decoder is instantiated.

Optional environment variables:

- `AUD_CAPTIONER_SOURCE`, `AUD_GEMMA_SOURCE`: alternate Hub IDs or local snapshots.
- `AUD_CAPTIONER_REVISION`, `AUD_GEMMA_REVISION`: commit/tag to select explicitly.
- `AUD_PYTHON`: Python executable; `AUD_VERIFY_DEVICE=cpu` for CPU verification.

Single-model export is useful after an interrupted batch:

```bash
python -m audm export --variant audturbo \
  --qasr-checkpoint "$QASR_CHECKPOINT" \
  --encoder-source Qwen/Qwen3-Omni-30B-A3B-Captioner \
  --output "$AUD_EXPORT_ROOT/AudTurbo"

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m audm verify "$AUD_EXPORT_ROOT/AudTurbo" --device cuda:0
```

Default export dtype is BF16; `--dtype float32` avoids reducing source precision.
Every target tensor is shape-checked and loaded strictly. Missing weights,
unknown AudPro tensors, conflicting tied embeddings and incomplete reloads are
errors. Successful exports are published with a directory rename; existing
destinations are never overwritten. A failed process can leave a hidden staging
directory, but it cannot publish a partial final artifact. Hub downloads are
cached normally.

## Dataset and augmentation inheritance

The supplied eight-node and twelve-node recipes reference **`configs/v8.1/02_full_12node.yaml`**, the
latest QASR stage-02 full-training recipe in this checkout. They import its
training/evaluation manifests, duration limits, target length limit, language
sampling and augmentation settings, with code-switch augmentation explicitly
disabled by default (`code_switch_index: null`, `code_switch_probability: 0.0`).
No code-switch index is required for default preflight or training. No corpus
is rebuilt or copied. To use an earlier stage-02 run, change `data_recipe` in a
new AUD recipe. Relative recipe and DeepSpeed paths resolve against the AUD YAML.

Inherited production behavior includes:

- Arabic, English, Hindi, Malayalam and Chinese manifest groups and full-coverage
  weighted language sampling. Quotas per GPU/update remain **10/6/2/1/1**.
- Speed perturbation, noise injection, codec augmentation and SpecAugment with
  the original settings and probabilities. Audio stays inside duration bounds.
- Optional indexed online code-switching remains available. To enable it, set
  `training.code_switch_index` to a valid index and
  `training.code_switch_probability` to a positive value (original QASR: **0.4**).
  Frozen source checks, tokenizer checks and train/eval exclusions still apply.
- Clean evaluation collators, loss, per-language WER/CER and macro WER;
  generation budgets are configurable. W&B and ordinary Trainer metrics files.
- Non-reentrant gradient checkpointing, BF16, preserved Conformer BatchNorm
  statistics, clipping, label smoothing, and separate learning rates for the
  audio/adapter, decoder, and token embeddings.

SpecAugment uses time-major mel features for Pro and a transposed view of Turbo's
mel features. Gemma-based Flash uses per-frequency mean fill over the original
valid log-mel frames, matching GQASR and avoiding unstable exact-zero prefixes.
Legacy Moonshine-based Flash instead masks STFT time/frequency bins and
reconstructs waveforms before its learned frontend. Evaluation never uses these
masks. Default all-rank loss and gradient guards abort before a nonfinite update;
AUD_DEBUG_NUMERICS=1 adds optional anomaly tracing and failing-batch capture.

For optional code-switching after the Malayalam cleanup, use the stage-03
`code_switch_indexes/cs_v1_ml_v2/assembled/index.json` revision, not stale `cs_v1`.
If absent, build it once from the repository root with the environment active:

```bash
RUN_ID=cs_v1_ml_v2 JOBS=8 PYTHON="$(command -v python)" \
  bash scripts/corpus/run_code_switch.sh refresh-ml \
  --source-index "$PWD/code_switch_indexes/cs_v1/assembled/index.json" \
  --quarantine "$PWD/asr_cleaned_manifests/ml_training_v2/quarantine/ml_damaged_transcripts.jsonl"
```

Keep the original `cs_v1` pools, which this revision references. Use a fresh
training output when enabling code-switching; exact resume requires unchanged
recipe and index receipts.

An old code-switch index is reusable only if its tokenizer fingerprint matches.
Preflight fails if Flash's tokenizer differs; it does not silently disable
mixing. In that case, copy `configs/code_switch/full_v1.yaml`, set its
`tokenizer_path` to the **AudFlash export**, keep the same source corpus, and
build a separate index through the existing commands:

```bash
python -m data_processing.code_switch plan --config /shared/audflash-cs.yaml --run-root /shared/new-audflash-cs
python -m data_processing.code_switch run --run-root /shared/new-audflash-cs --procs 32
python -m data_processing.code_switch verify --run-root /shared/new-audflash-cs --metadata-only
python -m data_processing.code_switch assemble --run-root /shared/new-audflash-cs
```

Then set `training.code_switch_index` in a copied AUD recipe to the new
`assembled/index.json` and set `training.code_switch_probability: 0.4`.
Do not modify the existing QASR index.

## Training: 8 nodes × 8 GPUs

The production launch shape is **64 ranks** with DeepSpeed **ZeRO-2**, one
`torchrun` launcher per node. The default is **10 samples/GPU/microbatch × 4
accumulation × 64 GPUs = 2,560 samples/update**, with two 20-example language quota windows per GPU/update.
Each quota window contains 10 Arabic, 6 English, 2 Hindi, 1 Malayalam and
1 Chinese sample. Two windows give 20/12/4/2/2 samples per GPU/update.
Microbatch must divide the quota window or contain whole multiples of it, and
each optimizer update must contain whole windows. For less activation memory, use microbatch 1 with accumulation
40; global batch and language quotas remain unchanged.

Use the export node for preflight before occupying the training nodes:

```bash
python -m audm preflight --config audm/configs/audturbo_adapter_8node.yaml \
  --model "$AUD_EXPORT_ROOT/AudTurbo" --output /shared/aud-runs/turbo-adapter --scan
```

Preflight checks shared paths, noise assets, tokenizer/index compatibility and
frozen code-switch metadata when explicitly enabled. `--scan` indexes the actual manifests and reports
full-coverage updates and per-language repeat factors. Startup still scans
manifests on training ranks, as the QASR loader does; large corpora can take time.

First run an NCCL connectivity/all-reduce check across the allocation:

```bash
AUD_COMMAND=distributed-check sbatch audm/scripts/train.slurm --expected-world-size 64
```

Run a two-update distributed rehearsal, then alignment training:

```bash
sbatch audm/scripts/train.slurm \
  --config audm/configs/audturbo_adapter_8node.yaml \
  --model "$AUD_EXPORT_ROOT/AudTurbo" --output /shared/aud-runs/turbo-adapter --smoke

sbatch audm/scripts/train.slurm \
  --config audm/configs/audturbo_adapter_8node.yaml \
  --model "$AUD_EXPORT_ROOT/AudTurbo" --output /shared/aud-runs/turbo-adapter
```

Smoke output goes under `smoke-test/`. Production can reuse a parent containing
only that rehearsal directory; it creates new files alongside it. Existing
production runs require `--resume` or a new output directory. Repeating a smoke
run also requires a fresh output location.

Adapter recipes freeze both pretrained towers and train only the new adapter
for an initial 2,000-update budget. Treat this as a starting experiment, not a
guarantee of convergence. Assess WER/CER and alignment before the next stage.

Full fine-tuning starts from the **alignment run's final model**, not the export:

```bash
sbatch audm/scripts/train.slurm \
  --config audm/configs/audturbo_full_8node.yaml \
  --model /shared/aud-runs/turbo-adapter --output /shared/aud-runs/turbo-full
```

Use the corresponding `audflash_adapter_8node.yaml` and
`audflash_full_8node.yaml` for Flash. AudPro can start directly with
`audpro_full_8node.yaml`. Run models sequentially on the 8-node pool unless
you intentionally divide resources and supply recipes with the new world size.
Full recipes use three coverage epochs with a cosine schedule; they do not blindly
copy a step count calculated for a different sampler or cluster size.

Fine-tuning options go under `training:` in a copied YAML. Unknown fields fail.
Partial augmentation overrides merge into the inherited augmentation settings;
`augmentation: null` disables the complete augmentation pipeline.
`stage: adapter`, `stage: decoder` (decoder + adapter), and `stage: full` are
supported. The existing QASR `TrainConfig` supplies optimizer, schedule,
augmentation, logging and dataset options. Auxiliary CTC requires an explicit
stride-one Gemma CTC export (see the new-run guide). AUD rejects FP16, torch.compile and ZeRO-3 configurations explicitly in this release.

### Resume versus a new fine-tuning stage

```bash
sbatch audm/scripts/train.slurm \
  --config audm/configs/audturbo_full_8node.yaml \
  --model /shared/aud-runs/turbo-adapter --output /shared/aud-runs/turbo-full \
  --resume auto
```

`--resume auto` selects the latest numbered Trainer checkpoint; an explicit
checkpoint path also works. Resume restores optimizer, scheduler, main-process
RNG, sampler epoch/data position and model state through Trainer/DeepSpeed.
Data-loader workers restart their augmentation RNG streams, so stochastic
multi-worker resumes are not bitwise identical. Resume equivalence tests use
deterministic, unaugmented runs. The recipe,
world size, data snapshot and code-switch index must match. A weights-only
export or final model directory is an input for a **new** fine-tuning run,
not an optimizer-resume checkpoint. Keep the original run directory/receipts
with its numbered checkpoints.

For a non-Slurm cluster, run the following on each node, changing
`AUD_NODE_RANK` from 0 to 7 and using the same reachable rank-0 address:

```bash
AUD_NNODES=8 AUD_NODE_RANK=0 AUD_MASTER_ADDR=training-node-0 \
  bash audm/scripts/train_distributed.sh \
  --config audm/configs/audflash_adapter_8node.yaml \
  --model "$AUD_EXPORT_ROOT/AudFlash" --output /shared/aud-runs/flash-adapter
```

Set `AUD_MASTER_PORT` per concurrent allocation. Choose network interface/HCA
environment variables using your site's known working QASR settings; this
package does not guess device names or disable InfiniBand.

## Offline use

```python
import soundfile as sf
from audm import AUDForConditionalGeneration, AUDProcessor

directory = "/shared/path/to/aud-models/AudPro"
processor = AUDProcessor.from_pretrained(directory)
model = (
    AUDForConditionalGeneration.from_pretrained(directory, local_files_only=True, dtype="auto")
    .to("cuda")
    .eval()
)
wave, rate = sf.read("mono-16khz.wav", dtype="float32")
batch = processor.apply_transcription_request([wave], language="en", sampling_rate=rate)
batch = {k: v.to(model.device) for k, v in batch.items() if k in processor.model_input_names}
tokens = model.generate(**batch, max_new_tokens=256, do_sample=False, use_cache=True)
text = processor.decode(tokens[0, batch["input_ids"].shape[1] :])
```

For training, `prepare_training_batch(audio, text, language)` produces masked
causal labels. `model.configure_training(stage)` configures trainable components.
Importing `audm` also registers its config/model with HF AutoConfig and
AutoModelForCausalLM. Source QASR/GQASR packages are not needed for inference.

## Validation and remaining cluster work

```bash
python -m pytest -c audm/pyproject.toml audm/tests -q
ruff check audm
```

Tests cover QASR forward/generation/processor parity, tied and untied decoder
weights, optional bridge/pool/CTC preservation, native encoder weight parity,
Captioner chunk-boundary lengths, Moonshine batch invariance, adapter gradients,
offline reload, missing/corrupt checkpoint rejection, all three training paths,
optimizer resume equivalence, augmentation layout and clean evaluation, and
64- and 96-rank sampler coverage. A two-process CPU DDP run has also exercised training,
checkpointing, evaluation and per-language WER/CER callbacks.

Full pretrained checkpoints and cluster datasets were not available locally.
The supplied NCCL check and distributed smoke run are required rehearsals for
your CUDA/DeepSpeed environment. Full-size peak memory, throughput, multilingual
ASR quality, CPU quantization and deployment latency remain to be measured.

## AudPro on five nodes

Use [AUDPRO_CLUSTER.md](AUDPRO_CLUSTER.md) for direct per-node commands alongside
an existing AudTurbo run. The dedicated `audpro_full_5node.yaml` uses 40 GPUs,
microbatch 10, accumulation 4, and three epochs (1,600 samples/update).
`audpro.sh` has separate `AUDPRO_*` paths and rendezvous settings.

## AudFlash on one node

Use [AUDFLASH_CLUSTER.md](AUDFLASH_CLUSTER.md) for the single-node, eight-GPU
workflow: export, verify, preflight, NCCL, smoke, adapter alignment, full training.
Adapter training uses microbatch 10 and accumulation 8; full training uses
microbatch 40 and accumulation 2. Both use four workers per GPU and 640
samples/update. Full loss and WER/CER evaluation use batches of up to 40. Adapter alignment runs 2,000 updates; full training runs
three epochs. The launcher isolates `AUDFLASH_*` settings from other AUD runs.

## Opt-in robustness experiment

See [ROBUSTNESS.md](ROBUSTNESS.md) for bounded waveform augmentation, actual codec
round trips, controlled non-speech sampling, CPU rehearsal and separate non-speech
evaluation. Existing production recipes remain unchanged.

For native decoder tokens that distinguish speech, music, noise and silence,
see [AUDIO_TYPES.md](AUDIO_TYPES.md). It covers tokenizer/model upgrade, decoder
warm-up, categorized manifests, structured inference and held-out evaluation.
Use [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md) for the audited, direct eight-GPU
workflow with asset scans, smoke runs and checkpoint resume.
