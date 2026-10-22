# AudFlash: audited robustness and audio-category training

This runbook uses one node with eight H100s, **10 × 8 × 8 = 640** sampled slots per
update, four DataLoader workers per GPU, BF16, ZeRO-2 and gradient checkpointing.
The decoder warm-up is 2,000 updates. Full training is three nominal weighted
coverage epochs; full evaluation/checkpoint saving stays at 3,700 updates.
Code-switch augmentation stays disabled. `RUN_ID=cs_v1_ml_v2` is retained, although
these recipes do not load a code-switch index.

The four new category tokens need decoder training. Use an already aligned or
trained **AudFlash Gemma4 stride-1** checkpoint as the source; choose a completed
checkpoint, not a directory currently being written by another training process.
This upgrade starts a separate experiment. It does not change the running
AudPro/AudTurbo/AudFlash outputs or resume their optimizer state.

## 1. Create an isolated code checkout and activate the existing environment

While other models are training, leave their source checkout in place. Create
this experiment's checkout once; on later logins just `cd` to it and restore the
environment variables. This avoids changing code that an active job's new workers
could import.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
git fetch origin
git worktree add --detach ../qasr_v2-aud-robust origin/main
cd ../qasr_v2-aud-robust
eval "$(conda shell.bash hook)"
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/audm

export PYTHONPATH="$PWD/audm/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RUN_ID=cs_v1_ml_v2
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/$USER/audflash-types-triton-$(hostname)"
mkdir -p "$TRITON_CACHE_DIR"

python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__); assert transformers.__version__.startswith("5.14."); assert torch.cuda.device_count() == 8; assert torch.cuda.is_bf16_supported()'
command -v ffmpeg
ffmpeg -hide_banner -version
```

Reuse the installed dependencies; do not upgrade the environment underneath
other training jobs. FFmpeg must include `libopus`, `aac`, `libmp3lame` and
`pcm_mulaw`. The deeper preflight checks actual encoding and decoding.

## 2. Set source, assets and experiment paths

First run the download, listening review and corpus-finalization steps in
[NON_SPEECH_DATA.md](NON_SPEECH_DATA.md) on one cluster node. The prepared corpus
lives outside the source checkout. Replace the three `/REPLACE/WITH/...` paths
below with your remaining immutable assets; the non-speech downloader does not
provide RIRs or a combined speech/non-speech evaluation set.

```bash
export AUD_TYPED_SOURCE=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/audflash-gemma4-v1/adapter
export AUD_TYPED_ROOT="$PWD/output/audflash-audio-types-v1"
export AUD_NON_SPEECH_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/aud-data/non-speech-v1
export AUD_NON_SPEECH_MANIFEST="$AUD_NON_SPEECH_ROOT/corpus-v1/train.jsonl"
export AUD_RIR_MANIFEST=/REPLACE/WITH/rir.jsonl
export AUD_TYPED_EVAL_MANIFEST=/REPLACE/WITH/audio-types-heldout.jsonl
export AUD_REHEARSAL_WAV=/REPLACE/WITH/representative-speech.wav
export AUD_TYPED_DECODER_CONFIG="$PWD/audm/configs/audflash_audio_types_decoder.yaml"
export AUD_TYPED_FULL_CONFIG="$PWD/audm/configs/audflash_robust_full.yaml"

test -f "$AUD_TYPED_SOURCE/model.safetensors"
test -f "$AUD_NON_SPEECH_MANIFEST"
test -f "$AUD_RIR_MANIFEST"
test -f "$AUD_TYPED_EVAL_MANIFEST"
test -f "$AUD_REHEARSAL_WAV"
mkdir -p "$AUD_TYPED_ROOT/logs"
```

If the default adapter path is not the checkpoint you want, change
`AUD_TYPED_SOURCE` before proceeding. Reuse `AUD_TYPED_ROOT` for this experiment's
subsequent commands and resumes. Choose a new root for a different recipe or corpus.

Negative JSONL rows require explicit labels and empty string transcripts:

```json
{"audio_filepath":"/shared/negative/rain.wav","duration":4.2,"text":"","audio_type":"noise"}
{"audio_filepath":"/shared/negative/instrumental.wav","duration":8.0,"text":"","audio_type":"music"}
{"audio_filepath":"/shared/negative/quiet.wav","duration":3.0,"text":"","audio_type":"silence"}
```

RIR rows require `audio_filepath`; relative audio paths resolve beside their
manifest. A held-out speech row includes `audio_type: "speech"`, a nonempty
transcript, and its language. Include all four categories in held-out evaluation.
Do not put noise/music under language keys. Exclude vocals and uncertain examples
from the negative pool; speech over noise or music remains speech with a transcript.
Split by original recording before chunking to avoid train/eval leakage.

The pool replaces 3% of slots in expectation, uniformly over eligible negative
records. Balance the negative pool deliberately and verify all intended categories
appear in the scan report. The speech datasets, language-slot sampler and 512-token
transcript limit are inherited from stage 02. Replacements reduce speech-only
coverage by approximately 3%; they do not increase the 640-slot update size.

## 3. Upgrade once, then verify and scan

```bash
python -m audm add-audio-types \
  --model-path "$AUD_TYPED_SOURCE" --output "$AUD_TYPED_ROOT/export"
python -m audm verify "$AUD_TYPED_ROOT/export" --device cuda:0

set -o pipefail
python -m audm preflight --config "$AUD_TYPED_DECODER_CONFIG" \
  --model "$AUD_TYPED_ROOT/export" --output "$AUD_TYPED_ROOT/decoder" --scan \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/preflight-decoder.log"

python -m audm.robustness_smoke --config "$AUD_TYPED_DECODER_CONFIG" \
  --audio "$AUD_REHEARSAL_WAV" \
  --output "$AUD_TYPED_ROOT/augmentation-rehearsal" --iterations 64
```

Run the upgrade once; rerunning it against the same destination intentionally
fails. The scan decodes eligible negatives, RIRs and real-noise files and exercises
all selected codecs. Review category counts, hours and exclusions. Listen to the
rehearsal WAV files. The rehearsal measures waveform processing plus WAV writing,
not GPU throughput or SpecAugment; the distributed smoke below exercises the full
training collator. Inspect its effect counters because the probabilistic rehearsal
does not guarantee every effect is selected.

A missing/broken asset or decoder should be fixed before training. The real-noise
directory comes from the inherited stage-02 augmentation settings. Never resume
after editing referenced audio files in place: manifests are hashed, WAV bytes are
not. Scanning can take time for a large noise corpus.

## 4. Direct eight-GPU smoke and decoder warm-up

```bash
export AUD_USE_SLURM=0 AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK=0
export AUD_MASTER_ADDR=127.0.0.1 AUD_MASTER_PORT=29613

AUD_COMMAND=distributed-check bash audm/scripts/train_distributed.sh \
  --expected-world-size 8

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" --smoke \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/smoke-decoder.log"
```

Check that all eight ranks finish, loss/gradients are finite, the saved checkpoint
reloads and GPU memory is acceptable on representative long recordings. The
ordinary smoke is short and may not encounter the worst-length batch. It uses at
most two loader workers per GPU; production uses four. Keep an eye on data-loader
wait time during the first production updates.

Also run the maximum-duration/transcript rehearsal in
[MEMORY_AUDIT.md](MEMORY_AUDIT.md) for both the decoder and full-stage recipes.
The trainer writes per-rank `cuda-memory-rank-N.json` diagnostics. Full training
has additional encoder activations and optimizer state, so passing decoder
warm-up does not establish that full training fits.

```bash
python -m audm verify "$AUD_TYPED_ROOT/decoder/smoke-test" --device cuda:0

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/train-decoder.log"

python -m audm.eval_audio_types --model-path "$AUD_TYPED_ROOT/decoder" \
  --manifest "$AUD_TYPED_EVAL_MANIFEST" --device cuda:0 \
  --limit 1000 --max-new-tokens 768 --output "$AUD_TYPED_ROOT/eval-decoder.json"
```

Decoder warm-up trains the decoder, embeddings/LM head and bridge, with the audio
encoder frozen. Evaluation and saves occur every 500 updates. Review missed-speech
rate, false-speech rate, non-speech text rate, invalid output formats and per-class
precision/recall alongside multilingual WER/CER. Generated class labels are not
calibrated confidence. An accuracy number alone can hide speech being suppressed.

## 5. Full training for three epochs

```bash
python -m audm preflight --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_TYPED_ROOT/full" --scan \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/preflight-full.log"

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" --smoke \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/smoke-full.log"

python -m audm verify "$AUD_TYPED_ROOT/full/smoke-test" --device cuda:0

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/train-full.log"

python -m audm.eval_audio_types --model-path "$AUD_TYPED_ROOT/full" \
  --manifest "$AUD_TYPED_EVAL_MANIFEST" --device cuda:0 \
  --limit 1000 --max-new-tokens 768 --output "$AUD_TYPED_ROOT/eval-full.json"
```

Full training unfreezes the encoder. It needs its own memory smoke; success of
decoder warm-up does not establish full-training memory headroom. Compare normal
speech WER/CER against the source checkpoint on the same evaluation data. Final
evaluation/save also run at training completion, even off the 3,700-step interval.

## 6. Restart an interrupted stage

Use the same environment, config, assets, model argument and output directory:

```bash
# Resume decoder warm-up from its latest complete Trainer checkpoint:
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" --resume auto

# OR resume full training:
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" --resume auto
```

Do not run both simultaneously on the same GPUs. `--resume` requires optimizer
state in a Trainer checkpoint; final weights alone are not sufficient. If a crash
left the newest checkpoint incomplete, select an earlier completed checkpoint
explicitly instead of `auto`. Changed recipe/manifest/policy receipts are rejected;
use weights-only fine-tuning into a new output for such changes. Runs started
before robustness receipt version 2 also need a new weights-only run.
Persistent-worker RNG/prefetch state is not restored bit-for-bit.

## Validation scope

Local tests cover the native token extension, target masking and gradients,
all three AUD variants, invalid labels/settings/assets, bounded effects and
codec round trips, clean evaluation, code-switch exclusion and persistent-worker
distributed save/resume. Two CPU/Gloo ranks exercise decoder and full stages.
These tests use tiny models and synthetic fixtures, not the pretrained checkpoint
or the cluster corpus. H100 memory, NCCL/DeepSpeed behavior, corpus label quality,
throughput and actual WER/category accuracy require the cluster runs above.

The current waveform effects are speed perturbation, pitch-preserving time
stretch, boundary-safe shift, RIR convolution, real noise, gain, white noise and
actual codec round trips, plus existing SpecAugment. Approximate codec degradation
remains optional. Independent pitch shifting, packet loss, and dedicated microphone
clipping/compression are not implemented by this audit.
