# Opt-in AUD robustness training

Existing production recipes are unchanged. `configs/audflash_robust_full.yaml`
is a separately selected experiment based on single-node AudFlash full training:
20 × 4 × 8 = 640 examples/update, four workers/GPU, three nominal weighted
coverage epochs, evaluation and checkpoint saving every 3,700 updates.

## Implemented policy

- 20% of sampled training slots bypass waveform effects **and SpecAugment**.
- Remaining slots draw configured effect candidates; at most two waveform
  effects are selected. Candidate probabilities are conditional on the non-clean
  branch; the selection budget reduces actual application rates.
- Speed and pitch-preserving time stretch are mutually exclusive. Actual codecs
  and approximate codec degradation are also mutually exclusive.
- Order: speed/tempo, boundary-safe shift, RIR, real noise, gain, white noise,
  codec. Existing SpecAugment follows feature extraction, with a per-example gate.
- Gain uses floating-point scaling without automatic clipping. White noise has
  an absolute noise level independent of speech SNR. RIR uses mono 16-kHz input,
  aligns to its dominant path, normalizes response energy, convolves with the
  complete tail, and preserves waveform RMS. This is an AUD implementation,
  inspired by the NeMo categories, not an exact NeMo reproduction.
- Shifts preserve length using zeros and only discard boundary samples below
  the configured peak threshold. They never wrap audio around.
- Length-changing effects that exceed the recipe bounds are rejected; speech
  is never cropped to make a perturbation fit. Near-maximum-length recordings
  may therefore skip slow-down, RIR or codec-tail augmentation.
- FFmpeg round trips: Opus, AAC, MP3 and G.711 are selected in the example.
  AMR-NB is supported when `libopencore_amrnb` is installed, but not enabled by
  default. Codec delay/padding is retained rather than blindly trimming the
  decoded waveform. Resulting length must fit the recipe.
- Missing encoders/assets, invalid output, and failed real-noise loading fail
  explicitly. No fallback to synthetic noise or approximate codecs.
- Worker-local counters report attempted/changed/duration-skipped effects,
  shift boundary skips, codec selections and non-speech replacements. These
  are diagnostics, not distributed aggregate statistics.

## Non-speech data

Supply curated, speech-free recordings: room tone, environmental sounds,
instrumental music and silence. Audible speech, singing and speech-containing
noise clips must not be labeled empty. The code validates labels, not acoustic
speech absence; that must be established when curating the corpus.

The manifest must explicitly contain empty `text` or `transcript`:

```json
{"audio_filepath":"/shared/negatives/room-tone.wav","duration":4.2,"text":""}
```

For checkpoints upgraded with [audio-category tokens](AUDIO_TYPES.md), every
negative row additionally requires `audio_type: "music"`, `"noise"` or
`"silence"`. Its supervised output is the category token plus EOS, without a
language marker. Typed silence bypasses all augmentation; typed music/noise
skip real-noise mixing to preserve their labels. The legacy manifest example
above remains valid for checkpoints without the token extension.

Training replaces each sampled slot independently with probability 0.03. This
is **3% in expectation**, not an exact per-batch quota. Total batch size remains
unchanged. Each replacement retains the original slot's language tag and is
never code-switched. Legacy checkpoints emit empty transcript content with
native language/task markers and EOS; typed checkpoints emit category plus EOS.
Language quotas describe sampled slots; speech-only counts and effective speech
coverage decrease by approximately 3%. Existing speech rows are not deleted.
A separate clean evaluation collator never injects non-speech replacements.

The framework's old `training.non_speech_manifest` and this controlled pool cannot
be combined. Pool/RIR manifest hashes and resolved robustness settings are saved
in the training receipt and checked on resume. Keep referenced WAV assets immutable;
the receipt hashes manifest contents, not every audio file.

## Cluster preparation

Use the existing AUD environment and a NEW output root. FFmpeg must be available
on every node/worker host. No NeMo installation is required. Dependencies for DSP
are already in AUD (SciPy/librosa). AMR encoder availability varies by FFmpeg build.

```bash
export PYTHONPATH="$PWD/audm/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AUD_NON_SPEECH_MANIFEST=/REPLACE/WITH/curated-non-speech-train.jsonl
export AUD_RIR_MANIFEST=/REPLACE/WITH/rir.jsonl
export AUDFLASH_FULL_CONFIG="$PWD/audm/configs/audflash_robust_full.yaml"
```

RIR JSONL rows require `audio_filepath`; relative paths resolve beside that
manifest. Non-speech audio also supports manifest-relative paths. Use absolute
manifest paths in the environment variables. They are resolved when the recipe
loads; unset variables fail instead of silently disabling effects.

Rehearse CPU augmentation and inspect the resulting float WAV files:

```bash
python -m audm.robustness_smoke \
  --config "$AUDFLASH_FULL_CONFIG" \
  --audio /REPLACE/WITH/representative-short.wav /REPLACE/WITH/representative-long.wav \
  --output output/robustness-rehearsal --iterations 32
```

This reports single-process waveform time including WAV output writing. It does
not establish eight-GPU throughput or test SpecAugment. Listen for intelligibility
and measure data-loader wait time on the cluster before a long training run.
Preflight checks selected encoder availability; actual round trips are exercised
when their effects are selected. An invalid/unsupported codec fails explicitly.

To retain the original adapter location while using a new full-training output,
launch directly with the existing distributed wrapper:

```bash
python -m audm preflight --config "$AUDFLASH_FULL_CONFIG" \
  --model "$AUDFLASH_RUN_ROOT/adapter" --output output/audflash-robust-full --scan

AUD_USE_SLURM=0 AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK=0 \
AUD_MASTER_ADDR=127.0.0.1 AUD_MASTER_PORT=29611 AUD_COMMAND=train \
bash audm/scripts/train_distributed.sh \
  --config "$AUDFLASH_FULL_CONFIG" --model "$AUDFLASH_RUN_ROOT/adapter" \
  --output output/audflash-robust-full --smoke
```

After reviewing the rehearsal, run the same launch without `--smoke`. Do not
resume an old production run with these changed data/augmentation settings.
Other AUD variants can use the same optional top-level `robustness` block.

## Separate non-speech evaluation

Use a held-out, explicitly empty-target manifest, not the training negative pool:

```bash
python -m audm.eval_non_speech \
  --model-path output/audflash-robust-full \
  --manifests /REPLACE/WITH/non-speech-eval.jsonl \
  --device cuda:0 --language en --limit 200 --max-new-tokens 64 \
  --output output/non-speech-eval.json
```

Reports nonempty-transcript rate, mean generated tokens (including task markers),
token-cap hits and per-record predictions. It takes the first eligible records
in manifest order; curate or shuffle a representative held-out manifest first.
Nonempty includes punctuation-only outputs. Inspect raw predictions and cap hits;
no ordinary WER is calculated against empty references. Compare the original and
new checkpoints on the same negatives and normal multilingual WER/CER sets.

Source reference: [NeMo audio augmentors](https://docs.nvidia.com/nemo/speech/nightly/asr/api.html#audio-augmentors).
