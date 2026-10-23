# AudFlash: ordered cluster runbook

Run each step to completion before starting the next. All GPU commands run directly
on **one node with eight H100s**, without Slurm/srun. Use a persistent terminal.
Stop on a failed command; do not continue to production after a failed probe.

The sequence is **completed AudFlash checkpoint → add category tokens → decoder
warm-up → full training**. Do not repeat adapter-only training: its frozen decoder
cannot learn the new tokens. Choose your best completed full-training checkpoint,
or the completed adapter checkpoint if that is what you have.

Both training stages use **10 × 8 × 8 = 640**, four workers/GPU, BF16, ZeRO-2 and
checkpointed audio/text layers. Decoder warm-up is 2,000 updates; full training is
three weighted coverage epochs, with evaluation/checkpoints every 3,700 updates.
SpecAugment stays enabled with GQASR-style per-frequency mean fill for Gemma;
code-switching stays disabled; `RUN_ID=cs_v1_ml_v2`.
Both robust recipes include silence padding (0–100 ms per end, candidate p=0.20)
without cutting speech or exceeding 30 seconds. Independent pitch, clipping,
dropouts and compression are available but disabled by default; see
[ROBUSTNESS.md](ROBUSTNESS.md) for settings and behavior. Existing `rir` is the
impulse-response convolution augmentation.
All-rank finite-loss and gradient guards are enabled by default, including
inside the DeepSpeed ZeRO-2 update path. Anomaly tracing is optional.

## 1. Set up the code checkout once

Create this checkout once while leaving other jobs' source checkouts in place.
If you already created this exact checkout, reuse it rather than running
`worktree add` again; update it only when no job is using it.

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2
git fetch origin
git worktree add --detach ../qasr_v2-aud-robust origin/main
cd ../qasr_v2-aud-robust
```

## 2. Save the experiment environment

Run once in the new checkout. The source defaults to your completed adapter;
edit `AUD_TYPED_SOURCE` if using a better completed full checkpoint. Never select
a checkpoint still being written. Keep this experiment root for later resumes.

```bash
cat > audflash-types.env <<'ENV'
export AUD_REPO_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2-aud-robust
export AUD_TYPED_SOURCE=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2/output/audflash-gemma4-v1/adapter
export AUD_TYPED_ROOT="$AUD_REPO_ROOT/output/audflash-audio-types-v1"
export AUD_NON_SPEECH_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/v2/aud-data/non-speech-v1
export AUD_NON_SPEECH_MANIFEST="$AUD_NON_SPEECH_ROOT/corpus-all-v1/train.jsonl"
export AUD_RIR_ROOT=/lustrefs/shared/shahin.konadath/workspace/train/stt/rirs-noises
export AUD_RIR_MANIFEST="$AUD_RIR_ROOT/prepared-v1/rir.jsonl"
export AUD_TYPED_DECODER_CONFIG="$AUD_REPO_ROOT/audm/configs/audflash_audio_types_decoder.yaml"
export AUD_TYPED_FULL_CONFIG="$AUD_REPO_ROOT/audm/configs/audflash_robust_full.yaml"
export RUN_ID=cs_v1_ml_v2
export PYTHONPATH="$AUD_REPO_ROOT/audm/src:$AUD_REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/$USER/audflash-types-triton-$(hostname)"
export AUD_USE_SLURM=0 AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK=0
export AUD_MASTER_ADDR=127.0.0.1 AUD_MASTER_PORT=29613
ENV

# Edit AUD_TYPED_SOURCE here if needed, before adding the tokens.
source audflash-types.env
eval "$(conda shell.bash hook)"
conda activate /lustrefs/shared/shahin.konadath/workspace/conda/envs/audm
set -o pipefail
mkdir -p "$AUD_TYPED_ROOT/logs" "$TRITON_CACHE_DIR"

python -c 'import torch, transformers; print(torch.__version__, transformers.__version__); assert transformers.__version__.startswith("5.14."); assert torch.cuda.device_count() == 8; assert torch.cuda.is_bf16_supported()'
command -v curl
command -v ffmpeg
ffmpeg -hide_banner -version
```

On later logins, `cd` to this checkout, activate the same conda environment and
`source audflash-types.env` again. Keep the same W&B login or set `WANDB_MODE=offline`.
Do not upgrade the environment while other jobs are using it.

## 3. Complete the uncapped non-speech corpus

If `corpus-all-v1` is already finalized, skip these first two commands. Existing
outputs are protected; do not overwrite them. No repeat download is needed when
the pinned FSD50K shards are already present.

```bash
bash audm/scripts/non_speech.sh prepare \
  --output "$AUD_NON_SPEECH_ROOT/candidates-all-v1"

bash audm/scripts/non_speech.sh finalize --metadata-only \
  --candidates "$AUD_NON_SPEECH_ROOT/candidates-all-v1/candidates.jsonl" \
  --output "$AUD_NON_SPEECH_ROOT/corpus-all-v1" \
  --silence-per-split 100 --seed 1729

cat "$AUD_NON_SPEECH_ROOT/corpus-all-v1/report.json"
test -s "$AUD_NON_SPEECH_MANIFEST"
```

If candidates-all-v1 already exists but the corpus does not, run only finalize.
No music/noise candidate-count caps apply. The 100 silences are synthetic examples,
not a limit on downloaded clips. The report records metadata-only selection. Inspect
the category balance: the pool samples uniformly over records and replaces 3% of
training slots. Growing the pool reduces the relative share of 100 silence rows.
Do not put negative manifests under the speech-language keys.

## 4. Download and prepare RIRs next to MUSAN

```bash
bash audm/scripts/rirs.sh all 2>&1 | tee "$AUD_TYPED_ROOT/logs/prepare-rirs.log"
bash audm/scripts/rirs.sh verify
cat "$AUD_RIR_ROOT/prepared-v1/report.json"
test -s "$AUD_RIR_MANIFEST"
test -d /lustrefs/shared/shahin.konadath/workspace/train/stt/musan
```

The source is [OpenSLR SLR28](https://www.openslr.org/28/), approximately 1.3 GB
compressed, with real and simulated RIRs. Reserve space for the archive and expanded
mono float WAVs (several additional GB). The downloader resumes its partial file and
checks the publisher's pinned MD5; the receipt also records SHA-256. Preparation
keeps every nonzero microphone channel separately, excludes the bundled point-source
and isotropic noise audio, preserves source README/license files, and writes
`$AUD_RIR_MANIFEST`. No Kaldi installation or new Python dependency is required.
Only one process should prepare this root at a time. `all` verifies an existing
preparation instead of overwriting it. `download`, `prepare`, and `verify` are also
available separately. No waveform or manifest edits after starting a training run.

MUSAN continues to supply background mixing on speech, at 5–20 dB SNR. It is not
added as standalone empty-target data. RIRs model reverberation, not background noise.

## 5. Add the four special tokens once

```bash
python - <<'PY'
import json, os
from pathlib import Path
source = Path(os.environ['AUD_TYPED_SOURCE'])
cfg = json.loads((source / 'config.json').read_text())
assert cfg['variant'] == 'audflash' and cfg['flash_encoder'] == 'gemma4'
assert cfg['flash_audio_stride'] == 1
assert not cfg.get('audio_type_version', 0), 'Source already has category tokens'
assert (source / 'model.safetensors').is_file()
print('Source:', source)
PY

python -m audm add-audio-types \
  --model-path "$AUD_TYPED_SOURCE" --output "$AUD_TYPED_ROOT/export"
python -m audm verify "$AUD_TYPED_ROOT/export" --device cuda:0
```

This adds `<|audio_speech|>`, `<|audio_music|>`, `<|audio_noise|>`, and
`<|audio_silence|>`, seeding their rows from existing category-word embeddings.
Existing assigned token IDs, encoder, bridge and pretrained decoder weights are
preserved. A completed export needs no second upgrade. If it already exists,
verify it and continue; never extend it again.

## 6. Check resolved configuration and scan real assets

```bash
python - <<'PY'
import os
from audm.training_config import load_recipe
for name, stage in [('AUD_TYPED_DECODER_CONFIG', 'decoder'), ('AUD_TYPED_FULL_CONFIG', 'full')]:
    cfg, actual, receipt = load_recipe(os.environ[name],
        model=os.environ['AUD_TYPED_ROOT'] + '/export',
        output=os.environ['AUD_TYPED_ROOT'] + '/' + stage)
    assert actual == stage
    assert (cfg.per_device_train_batch_size, cfg.gradient_accumulation_steps,
            cfg.expected_world_size, cfg.dataloader_num_workers) == (10, 8, 8, 4)
    assert cfg.code_switch_probability == 0
    assert cfg.augmentation['spec_augment']['enabled']
    assert cfg.aud_robustness['non_speech']['probability'] == 0.03
    if stage == 'full':
        assert cfg.num_train_epochs == 3 and cfg.max_steps == -1
        assert cfg.eval_steps == cfg.save_steps == 3700
    else:
        assert cfg.max_steps == 2000
    print(stage, 'SpecAugment:', cfg.augmentation['spec_augment'])
    print(stage, 'waveform policy:', cfg.aud_robustness['augmentation'])
PY

python -m audm preflight --config "$AUD_TYPED_DECODER_CONFIG" \
  --model "$AUD_TYPED_ROOT/export" --output "$AUD_TYPED_ROOT/decoder" --scan \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/preflight-decoder.log"
```

The scan decodes negatives, RIRs and MUSAN assets, and exercises the configured
Opus/AAC/MP3/G.711 codecs and enabled extra effects. Check the report's three negative categories and global
batch 640. It may take time to scan the uncapped corpus. SpecAugment uses p=0.37
on the non-clean branch, with up to two time/frequency masks (maximum ratios
0.05/0.15). The policy retains a 20% clean branch and at most two waveform effects.
Examples labeled `silence` and evaluation remain unaugmented.

## 7. Check eight-GPU communication

```bash
AUD_COMMAND=distributed-check bash audm/scripts/train_distributed.sh \
  --expected-world-size 8
```

## 8. Define the longest-shape memory rehearsal

Run this function definition once per shell. It uses the normal trainer with the
production batch/accumulation/workers, maximum audio duration and near-maximum
transcript length. Two updates exercise initialized optimizer state. Synthetic
probe weights must never be used for training continuation or evaluation.

```bash
aud_memory_probe() {
  local stage="$1" config="$2" model="$3"
  local probe="$AUD_TYPED_ROOT/memory-$stage"
  python -m audm.memory_probe --config "$config" --model "$model" \
    --output "$probe" --steps 2 || return
  AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
    --config "$probe/recipe.yaml" --model "$model" --output "$probe/run" \
    2>&1 | tee "$AUD_TYPED_ROOT/logs/memory-$stage.log" || return
  python - "$probe/run" <<'PY'
import json, sys
from pathlib import Path
paths = sorted(Path(sys.argv[1]).glob('cuda-memory-rank-*.json'))
assert len(paths) == 8, f'Expected eight reports, got {len(paths)}'
for path in paths:
    report = json.loads(path.read_text())
    assert report['status'] == 'completed', report
    print('rank', report['rank'], 'peak allocated GiB', report['peak_allocated_gib'],
          'peak reserved GiB', report['peak_reserved_gib'])
PY
}

aud_memory_probe decoder "$AUD_TYPED_DECODER_CONFIG" "$AUD_TYPED_ROOT/export"
```

Do not pass `--smoke` to this probe. It disables augmentation and evaluation to
stress model shapes; the next step tests the real data/augmentation pipeline.
A probe output must be new. If rerunning, choose a new name such as `decoder-retry`.

## 9. Smoke-test real data, then train the decoder

```bash
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" --smoke \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/smoke-decoder.log"

python -m audm verify "$AUD_TYPED_ROOT/decoder/smoke-test" --device cuda:0

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/train-decoder.log"

python -m audm verify "$AUD_TYPED_ROOT/decoder" --device cuda:0
```

Decoder warm-up trains the decoder, embeddings/LM head and bridge while the encoder
stays frozen. Review finite loss/gradients, multilingual WER/CER, data-loader wait
time, and sample predictions. A smoke run uses fewer workers/shorter generation;
it does not replace the maximum-shape probe. Completion should report step 2,000.

## 10. Scan and rehearse full training separately

```bash
python -m audm preflight --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_TYPED_ROOT/full" --scan \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/preflight-full.log"

aud_memory_probe full "$AUD_TYPED_FULL_CONFIG" "$AUD_TYPED_ROOT/decoder"

AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" --smoke \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/smoke-full.log"

python -m audm verify "$AUD_TYPED_ROOT/full/smoke-test" --device cuda:0
```

Full training adds encoder gradients and optimizer state. Check `AUD architecture`
in the startup log: both encoder and decoder checkpoint-enabled counts must match
their layer counts, and decoder cache must be false. In inference-only `verify`,
checkpoint-enabled counts of zero are normal. Stop if any rank OOMs or fails.

## 11. Start full training for three epochs

```bash
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/train-full.log"

python -m audm verify "$AUD_TYPED_ROOT/full" --device cuda:0
```

Use the completed decoder checkpoint as the input, never the smoke/probe weights.
Normal speech WER/CER callbacks continue. Final evaluation/save also run on completion,
even off the 3,700-update interval. Before deploying, evaluate held-out categories
and speech suppression as well as ASR accuracy; unreviewed negative labels do not
establish ground-truth quality. A four-class evaluation manifest needs held-out
speech rows plus the held-out negatives, not only noise/music/silence.

## 12. Resume only an interrupted stage

Restore the same environment, code, recipes, assets and output paths first. Run
only the command for the stage that was interrupted, not both:

```bash
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_DECODER_CONFIG" --model "$AUD_TYPED_ROOT/export" \
  --output "$AUD_TYPED_ROOT/decoder" --resume auto

# OR, for full training:
AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_TYPED_ROOT/full" --resume auto
```

Resume requires a complete Trainer checkpoint with optimizer state. If the newest
checkpoint is incomplete, specify an earlier completed checkpoint explicitly.
Changed recipes/manifests/robustness policies require weights-only continuation
into a new output. The current robustness receipt is version 5; persistent-worker
RNG/prefetch state is not restored bit-for-bit.

## Updating an existing experiment for silence padding

For an experiment with finalized datasets, added tokens and a completed healthy
decoder checkpoint, do not repeat those preparations or decoder warm-up. This
policy change requires a new full-training output and weights-only continuation;
do not resume optimizer state from the earlier robustness version or use a failed
smoke checkpoint. Keep the existing experiment root and decoder source.

When no job is using this source checkout, update it and restore your existing
environment. The examples assume the same conda environment is active. Run one
command at a time and stop if any command fails:

```bash
cd /lustrefs/shared/shahin.konadath/workspace/train/stt/v2/qasr_v2-aud-robust
git pull --ff-only origin main
source audflash-types.env
set -o pipefail
export AUD_SILENCE_FULL_OUTPUT="$AUD_TYPED_ROOT/full-silence-v1"

python -m audm preflight --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_SILENCE_FULL_OUTPUT"

AUD_DEBUG_NUMERICS=0 AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_SILENCE_FULL_OUTPUT" --smoke \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/smoke-full-silence-v1.log"

python -m audm verify "$AUD_SILENCE_FULL_OUTPUT/smoke-test" --device cuda:0

AUD_DEBUG_NUMERICS=0 AUD_COMMAND=train bash audm/scripts/train_distributed.sh \
  --config "$AUD_TYPED_FULL_CONFIG" --model "$AUD_TYPED_ROOT/decoder" \
  --output "$AUD_SILENCE_FULL_OUTPUT" \
  2>&1 | tee "$AUD_TYPED_ROOT/logs/train-full-silence-v1.log"
```

Add `--scan` to preflight if assets have not already been scanned or have changed.
All-rank numerical guards remain on when anomaly tracing is off. The smoke run
tests the stochastic policy on a small sample; it does not guarantee every effect
is selected or establish production accuracy/throughput. Do not use its weights
as the full-training source. Batch, workers, epochs and save/evaluation intervals
remain unchanged.

## Scope and validation

RIR preparation tests cover publisher-checksum enforcement, mono-channel preservation,
noise exclusion, invalid audio, unsafe ZIP paths, immutable output, verification and
actual convolution through the training augmenter. Full source assets are downloaded
on the cluster, not during local tests. See [MEMORY_AUDIT.md](MEMORY_AUDIT.md) for
memory changes and [NON_SPEECH_DATA.md](NON_SPEECH_DATA.md) for corpus details.

The default waveform policy includes speed perturbation, pitch-preserving time
stretch, silence padding, boundary-safe shift, RIR convolution, MUSAN mixing, gain,
white noise and real codec round trips, followed by feature-level SpecAugment.
Independent pitch shifting, zero-filled dropouts, clipping and compression are
implemented as optional experiments, disabled by default. CPU regression tests
do not establish full-size H100 memory, throughput or accuracy.
