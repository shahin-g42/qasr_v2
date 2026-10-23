# Opt-in AUD robustness training

Existing production recipes are unchanged. `configs/audflash_robust_full.yaml`
is a separately selected experiment based on single-node AudFlash full training:
10 × 8 × 8 = 640 examples/update, four workers/GPU, three nominal weighted
coverage epochs, evaluation and checkpoint saving every 3,700 updates.

## Implemented policy

- 20% of sampled training slots bypass waveform effects **and SpecAugment**.
- Remaining slots draw configured effect candidates; at most two waveform
  effects are selected. Candidate probabilities are conditional on the non-clean
  branch; the selection budget reduces actual application rates.
- Speed, pitch-preserving time stretch and independent pitch shift are mutually
  exclusive. So are shift/silence padding, and actual/approximate codec degradation.
- Order: speed/tempo/pitch, silence padding or boundary-safe shift, RIR, real noise,
  gain, white noise, compression, clipping, dropouts, codec. Existing SpecAugment
  follows feature extraction, with a per-example gate and per-frequency mean fill
  for Gemma audio features.
- Gain uses floating-point scaling without automatic clipping. White noise has
  an absolute noise level independent of speech SNR. RIR uses mono 16-kHz input,
  aligns to its dominant path, normalizes response energy, convolves with the
  complete tail, and preserves waveform RMS. This is an AUD implementation,
  inspired by the NeMo categories, not an exact NeMo reproduction.
- Shifts preserve length using zeros and only discard boundary samples below
  the configured peak threshold. They never wrap audio around.
- Silence padding preserves **every original sample** and adds independently
  sampled 0–100 ms zero padding at each end in the shipped robust recipes.
  Only the added padding is shortened when the 30-second budget is tight.
  This shifts speech within a longer recording without deleting phonemes.
  The category/transcript stay unchanged: it does not create a `silence` label.
- Length-changing effects that exceed the recipe bounds are rejected; speech
  is never cropped to make a perturbation fit. Near-maximum-length recordings
  may therefore skip slow-down, RIR or codec-tail augmentation.
- FFmpeg round trips: Opus, AAC, MP3 and G.711 are selected in the example.
  AMR-NB is supported when `libopencore_amrnb` is installed, but not enabled by
  default. Codec delay/padding is retained rather than blindly trimming the
  decoded waveform. Resulting length must fit the recipe.
- Missing encoders/assets, invalid output, and failed real-noise loading fail
  explicitly. No fallback to synthetic noise or approximate codecs.
- Settings are checked for unknown keys, invalid types, NaN/Inf, reversed ranges,
  empty codec lists and unsupported codecs. FFmpeg encode/decode uses one codec
  thread per subprocess and a 30-second timeout.
- Worker-local counters report attempted/changed/duration-skipped effects,
  shift boundary skips, codec selections and non-speech replacements. These
  are diagnostics, not distributed aggregate statistics.
- Startup logs show the complete effective AUD policy. Earlier QASR component
  initialization logs do not describe the final waveform selection probabilities.

## Extra waveform effects

Both `audflash_audio_types_decoder.yaml` and `audflash_robust_full.yaml` enable
silence padding with candidate probability `p: 0.20`. The 20% clean branch and
two-effect cap still apply; `p` is not the final observed application rate.
Existing shift remains ±5 ms with its boundary check. SpecAugment remains enabled.

The following implementations are available with `p: 0.0` in both recipes. Enable
one at a time in a separate experiment and compare multilingual WER/CER, category
errors and throughput before making it a production default.

| Effect | Configured range if enabled | Behavior |
|---|---|---|
| `pitch` | −1 to +1 semitone | Librosa pitch shift with unchanged duration; excludes speed/time stretch on the same sample. |
| `clipping` | 70–95% of input peak | Symmetric hard clipping; no automatic gain recovery. Models one type of microphone overload. |
| `dropout` | 5–20 ms, at most two, total ≤1% | Zero-filled gaps, bounded by total original samples. Does not model a codec's packet-loss concealment. May hide a brief phoneme, so disabled by default. |
| `compression` | −24 to −12 dBFS threshold, 2:1–4:1 ratio, 5 ms attack, 50 ms release | Real FFmpeg RMS compressor, no makeup gain, unchanged length. Requires the `acompressor` filter. |

Independent pitch shift and compression cost CPU time; disabled effects perform
no DSP work. Bounds are validated even when disabled. Silence padding accepts
up to 1,000 ms per end for explicit experiments; that is a supported bound, not
a recommended training setting. Cropping/random segments are deliberately absent:
unaligned cropping can remove words while keeping their transcript as the target.

**Impulse augmentation is already implemented as `rir`.** It convolves the input
with a sampled room impulse response from `AUD_RIR_MANIFEST`; no second impulse
effect or new download is needed. Its full reverberation tail is kept when it fits
the duration budget. This differs from injecting isolated clicks or impulse noise.

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
skip real-noise mixing to preserve their labels. Music also skips additive white
noise, which could dominate quiet music. Untyped empty-target negatives skip
real-noise mixing as well. The legacy manifest example
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

Both `text` and `transcript`, when present, must be empty strings. Nulls, numbers,
punctuation, missing text fields and contradictory aliases are rejected. Negative
replacement happens before code-switching; empty-target rows are never mixed.
Relative robustness manifest paths resolve beside the AUD recipe. Unset environment
variables produce an explicit error.

The audited policy uses `robustness_version: 5` in its run receipt. Real-noise
augmentation reads random bounded windows and resamples with a polyphase filter,
instead of decoding whole files and repeatedly using their first seconds. Resume within
the same run keeps optimizer/scheduler progress and checks the recipe/assets.
An experiment started with the earlier robustness code needs a new weights-only
run (`--model` and a new `--output`), because its augmentation semantics changed.
With persistent workers, stochastic augmentation and prefetch state are not
checkpointed; resume does not promise bit-identical augmented audio.

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
`preflight --scan` decodes every eligible negative, every RIR and every selected
real-noise asset, runs an actual round trip for every selected codec, and exercises
enabled extra effects (reported as `extra_effects_checked`). It checks
decoded negative durations against the training limits and reports eligible
category counts, hours and metadata duration exclusions. It does not decode all
98M sampled speech slots or determine whether a recording acoustically contains
speech. This can take time for a large noise corpus; run it once on rank 0 before
launch. An invalid/unsupported codec fails explicitly.

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

For the complete audited AudFlash category-token workflow, including setup,
checkpoint resume and acceptance checks, use [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md).
