# QASR Lab

An independently installable package for evaluating the Cohere Conformer → bridge → Qwen ASR architecture and running controlled adaptation pilots.

All implementation, examples, and tests live in this directory. Nothing imports this package from the production pipeline. It reads existing QASR code and checkpoints, creates new output directories, and never edits source checkpoints. Existing output directories are refused.

## Start in a separate environment

Run these commands from the repository root:

~~~sh
python3.11 -m venv qasr_lab/.venv
qasr_lab/.venv/bin/python -m pip install -e './qasr_lab[hf,dev]'
qasr_lab/.venv/bin/qasr-lab --help
~~~

The core package has no runtime dependencies: installing just ./qasr_lab is sufficient for panels, saved-prediction scoring, comparisons, and plan generation. Model execution uses optional dependencies. The HF integration targets Transformers 5.14.x. It uses the QASR checkout explicitly selected by **qasr_source**; it does not install or upgrade the root project.

Use a **separate SDK environment** for original Audar/Qwen checkpoints requiring the official qwen-asr loader:

~~~sh
python3.11 -m venv qasr_lab/.venv-sdk
qasr_lab/.venv-sdk/bin/python -m pip install -e './qasr_lab[audio]' qwen-asr
~~~

Keep the original checkpoint's supported SDK version. The qwen-sdk backend follows the [official Qwen3-ASR SDK](https://github.com/QwenLM/Qwen3-ASR). The hf-qwen backend is for checkpoints in the native HF Qwen3-ASR format; the formats are not interchangeable.

## Use the existing v8.1 configuration

When the production YAML already defines the checkpoint and language-grouped datasets, use it directly. With qasr3 active and both qasr_lab/src and src on PYTHONPATH, run from the repository root:

~~~sh
python -m qasr_lab from-training-config \
  --config configs/v8.1/02_full_12node.yaml --inspect

export QASR_LAB_RUN="$PWD/qasr_lab/outputs/v81-pilot-$(date +%Y%m%d-%H%M%S)"

python -m qasr_lab from-training-config \
  --config configs/v8.1/02_full_12node.yaml \
  --output "$QASR_LAB_RUN" \
  --train-per-language 500 --eval-per-language 50 \
  --seeds 42 43 --hash-audio

CUDA_VISIBLE_DEVICES=0 python -u -m qasr_lab train \
  --config "$QASR_LAB_RUN/smoke.json"
~~~

The 10-node YAML can be supplied instead; its checkpoint and dataset lists currently match the 12-node YAML. Inspection does not scan manifests, load weights, or start a job. Preparation reads the local checkpoint's processor/tokenizer without loading model weights. The default `--sampling records` makes one raw read through each uncached manifest, retains a uniform random sample of record lines, and parses only that small sample. It writes the candidate cache and prepared panels under lab outputs. It does not parse or resolve audio paths for every row in a large corpus.

The adapter reads model_name_or_path, train_manifest, eval_manifest, optional non_speech_manifest, audio_root, duration limits, transcript-token limit, learning rates, precision, gradient checkpointing, and generation-token limit. Language assignments come from the YAML groups, using the production parse_record behavior, including text/transcript fallback and /vast remapping. Missing candidate durations are read from audio metadata. Overlong candidate transcripts are rejected without truncation.

`--candidates-per-manifest 2000` controls the reservoir size. Reservoir sampling gives every nonblank record line the same chance of inclusion, regardless of byte length; it does not require a line count or seek to random bytes. With 22 training and 6 validation manifests, at most 56,000 candidate records are parsed. The final per-language budget is divided approximately equally across that language's manifests, then eligible candidates are selected deterministically. For example, 500 Arabic training samples across 13 manifests means up to 38 or 39 from each, rather than 500 from each. Quotas are caps: filtering, duplicates or small manifests can produce fewer examples. Per-manifest selected counts are printed and recorded; raise the candidate budget if too few eligible examples remain. This source-balanced pilot does not reproduce full-corpus source proportions.

Candidates are cached in `qasr_lab/outputs/manifest-sample-cache/` when using the example output paths. `--sample-cache` selects another location. The cache stores sampled raw records, line numbers, byte offsets, the full manifest hash from the original pass, and the selection seed. Unchanged files reuse the sample without rereading the original manifest. Cache lookup checks the resolved source path, size, modification/change timestamps, seed, and candidate limit; it does not rehash the entire source on a cache hit. Each completed manifest is cached separately, so an interruption only loses the unfinished manifest's pass. All seeds and variants then share the same frozen panels. A first uncached pass still reads all manifest text; the command reports record throughput, and its wall time depends on filesystem speed.

In record-sampling mode, overlap checks cover the sampled training candidates and the selected validation panel. They do not establish that validation is absent from the unsampled 90-million-record corpus. This limitation is recorded as `sampled_candidates_only` when any training source is subsampled. Supplied identity/hash metadata and selected audio hashes are still checked. Use `--sampling full` explicitly to restore the expensive whole-manifest parse/audit and global per-language ranking; it does not use the candidate cache. Neither mode establishes that an existing checkpoint has never seen the validation data.

The configured model_name_or_path is used exactly. The v8.1 files currently point to models/bck/full/v1; the adapter does not guess a newer checkpoint from output_dir or resume an optimizer state. Use --checkpoint with an explicit local directory when intentionally starting from a newer QASR model.

Preparation creates:

- train-panel and validation-panel, plus a non-speech-panel when configured or requested with MUSAN.
- base-42.json / base-43.json and seed-42 / seed-43 plans with separate run directories.
- smoke.json with two optimizer steps and small subsets of both panels.
- eval-initial.json and eval-variant-seed.json configurations for generated transcripts.
- source-config.json and prepared.json with provenance, scan/filter counts, and the pilot settings.

These are small adaptation pilots: batch size 1, accumulation 8, 200 steps, cosine scheduling, and the trainable-component choices described below. They do not reproduce production world size, weighted language quotas, augmentation, online code switching, label smoothing, or the full training schedule. The production YAML and pipeline remain unchanged.

The current v8.1 files contain 22 training manifests, 6 validation manifests, and no non-speech manifest. They do not define a separate test set. The adapter therefore labels the evaluation panel as existing configured validation data and never creates a fictional held-out test panel. A later independent test set is still needed for final claims.

### Reuse the configured MUSAN directory

Add `--musan` to prepare empty-target noise examples using `augmentation.noise_injection.noise_dir` from the same YAML. The v8.1 path is `/lustrefs/shared/shahin.konadath/workspace/train/stt/musan`. This is an opt-in lab setting; it does not change the production YAML or augmentation.

~~~sh
python -m qasr_lab from-training-config \
  --config configs/v8.1/02_full_12node.yaml --musan --inspect

export QASR_LAB_RUN="$PWD/qasr_lab/outputs/v81-musan-$(date +%Y%m%d-%H%M%S)"

python -m qasr_lab from-training-config \
  --config configs/v8.1/02_full_12node.yaml \
  --output "$QASR_LAB_RUN" --musan \
  --sampling records --candidates-per-manifest 2000 \
  --train-per-language 500 --eval-per-language 50 \
  --non-speech-per-language 50 --seeds 42 43 --hash-audio
~~~

Only the standard `noise/` partition is read. The [MUSAN paper](https://arxiv.org/abs/1510.08484) describes this partition as excluding intelligible speech, although indistinct crowd voices can occur. The speech and music partitions are excluded; music can contain vocals. This trusts the corpus labels and does not run a speech detector. The loader requires the standard 16 kHz mono WAV format and fails explicitly for a missing partition or incompatible file.

The lab hashes eligible source files, groups byte-identical copies, reserves 20% of unique recordings for diagnostics, and creates one deterministic crop of at most 10 seconds per selected recording (subject to the configured duration limits). Sampling caps apply after the split. Crops and their source hashes/offsets are saved under `musan-audio/`; the original corpus is unchanged. Decoder prompt languages are distributed across configured languages; this does not assign a spoken language to noise. The limit of 50 per language is a cap, not a requirement when the partition has fewer recordings.

`non-speech-panel/` feeds only the additional `non_speech` training variant. `non-speech-validation-panel/` contains different source recordings and is used for generation-based diagnostics. This makes five variants per seed, or ten jobs for two seeds; run at most eight simultaneously on the 8-GPU machine. Compare `non_speech` with `control`, using both the ordinary speech validation panel and the noise diagnostic panel. Other variants and the CTC smoke test retain their original speech-only data. A configured `non_speech_manifest` and `--musan` cannot be combined in one preparation run.

To measure the initial checkpoint's false-transcription rate:

~~~sh
CUDA_VISIBLE_DEVICES=0 python -m qasr_lab run \
  --model "$QASR_LAB_RUN/eval-initial.json" \
  --panel "$QASR_LAB_RUN/non-speech-validation-panel" \
  --output "$QASR_LAB_RUN/evaluation-initial-noise"
~~~

After training, use the same panel with `eval-control-42.json` and `eval-non_speech-42.json`, writing each result to its own new output directory. Repeat for seed 43. Report false-transcription rate and generated words/characters per minute alongside speech WER/CER. These MUSAN recordings may already have been used in production augmentation, so the noise panel is a diagnostic split for this pilot, not an independent held-out test set.

### Launch and evaluate the prepared pilots

After the smoke test passes, each plan can run on one GPU. For example:

~~~sh
CUDA_VISIBLE_DEVICES=0 python -u -m qasr_lab train \
  --config "$QASR_LAB_RUN/seed-42/control.json"
~~~

The four default variants across seeds 42 and 43 fit eight independent GPU jobs; adding non-speech gives ten. Evaluate the starting checkpoint before the pilots, and compare trained checkpoints against both it and their appropriate adaptation control:

~~~sh
CUDA_VISIBLE_DEVICES=0 python -m qasr_lab run \
  --model "$QASR_LAB_RUN/eval-initial.json" \
  --panel "$QASR_LAB_RUN/validation-panel" \
  --output "$QASR_LAB_RUN/evaluation-initial"
~~~

## 1. Freeze panels from manually specified manifests

Input is JSONL, one clip per row:

~~~json
{"id":"en-001","audio_filepath":"/data/heldout/en-001.wav","text":"The reference transcript.","language":"en","duration":3.2,"source":"corpus-a","speaker_id":"spk7","recording_id":"rec18","tags":["noisy","proper_names"]}
~~~

Use an empty text string for silence, music, or other non-speech. Duration is seconds. Language codes should be consistent (ar, en, zh, hi, ml); a manifest lacking a language can use --language en. Relative audio paths resolve against the manifest directory, or an explicit --audio-root. Keep code-switching and domain labels in tags; stratum optionally defines a sampling category within a language.

~~~sh
qasr_lab/.venv/bin/qasr-lab panel \
  --manifest /data/heldout/multilingual.jsonl \
  --manifest /data/heldout/non_speech.jsonl \
  --exclude-manifest /data/training/train.jsonl \
  --per-stratum 500 --seed 42 --require-audio --hash-audio \
  --output qasr_lab/outputs/panels/heldout-v1
~~~

Selection is deterministic and bounded in memory per (language, stratum), independent of manifest ordering. Default strata separate speech and non-speech. The saved panel records sample IDs, manifest hashes, selection settings, and a content checksum. The --hash-audio option additionally fingerprints selected audio bytes; inference verifies supplied audio hashes. Without it, manifest integrity and duration checks do not prove unchanged waveform contents.

The leakage check compares paths, supplied audio hashes, and corpus-qualified speaker/recording IDs against each excluded manifest. It fails on overlap. It cannot identify copied recordings under different paths without hashes or identity metadata. It does not invent speaker IDs or establish held-out status when training manifests are unavailable.

Prepare a small training panel and a separate validation panel the same way. The pilot runner rejects train/validation identity overlap. Reserve a third panel for final model comparisons; do not tune on it.

## 2. Evaluate the three models on the same panel

Copy and fill in [qasr.json](examples/qasr.json), [audar.json](examples/audar.json), and [cohere.json](examples/cohere.json). Checkpoint and source paths in configurations resolve relative to the configuration file. Hub identifiers require an immutable 40-character revision; local checkpoints require safetensors and are hashed by default.

~~~sh
qasr_lab/.venv/bin/qasr-lab run \
  --model qasr_lab/examples/qasr.json \
  --panel qasr_lab/outputs/panels/heldout-v1 \
  --output qasr_lab/outputs/evaluation/qasr

qasr_lab/.venv-sdk/bin/qasr-lab run \
  --model qasr_lab/examples/audar.json \
  --panel qasr_lab/outputs/panels/heldout-v1 \
  --output qasr_lab/outputs/evaluation/audar

qasr_lab/.venv/bin/qasr-lab run \
  --model qasr_lab/examples/cohere.json \
  --panel qasr_lab/outputs/panels/heldout-v1 \
  --output qasr_lab/outputs/evaluation/cohere
~~~

Inference is one clip at a time, greedy, at 16 kHz, with language hints by default. Use pre-segmented audio within each processor's limit. The runner rejects chunked inputs rather than accidentally scoring only the first chunk. It records per-sample failures and returns a nonzero exit code for incomplete runs.

Outputs include predictions.jsonl, report.json, and a standalone report.html:

- Strict WER/CER, with substitution, deletion, and insertion counts.
- Recognition-normalized WER/CER, reported separately from strict scores.
- Per-language and overlapping tagged slices. Use CER for Chinese/Japanese.
- False transcription rate and generated words/characters per minute on non-speech.
- Known token-limit hits; unavailable SDK token-limit information is marked unobserved.
- Warmed, synchronized preprocessing + model latency, p50/p95, corpus RTF, and peak allocated GPU memory. File I/O is excluded.
- Checkpoint fingerprints, code hashes, software versions, and decoding settings.

Recognition normalization preserves Hindi/Malayalam combining marks and Arabic letter distinctions. It removes Arabic harakat/tatweel only for ar, normalizes decimal digits, casefolds, and treats punctuation consistently. It does not equate spelled-out numbers with numerals. CER counts Unicode code points after NFC, excluding whitespace; it is not a grapheme-cluster metric. Speech WER excludes empty-reference clips, which have their own metrics. Global corpus totals are descriptive; make decisions per language.

~~~sh
qasr_lab/.venv/bin/qasr-lab compare \
  --report qasr_lab/outputs/evaluation/audar/report.json \
  --report qasr_lab/outputs/evaluation/qasr/report.json \
  --report qasr_lab/outputs/evaluation/cohere/report.json \
  --bootstrap 2000 --seed 42 \
  --output qasr_lab/outputs/comparisons/three-models
~~~

The first report is the baseline. Comparisons require identical panels and scoring policies, with no failed samples. They report candidate-minus-baseline error changes with paired bootstrap intervals, resampling recordings when annotated and utterances otherwise. A single recording cannot support an interval. Intervals are exploratory and not corrected for multiple comparisons.

Timing speedups are reported only when recorded hardware, software, timing, and decoding protocols match. Identical protocols are necessary but do not eliminate shared-machine load or cross-runtime confounds. Repeat latency measurements on an idle device; different tokenizers also make an equal token cap an imperfect constraint.

To score predictions generated by your existing pipeline, export rows containing id and hypothesis:

~~~sh
qasr_lab/.venv/bin/qasr-lab score \
  --panel qasr_lab/outputs/panels/heldout-v1 \
  --predictions /path/predictions.jsonl --model-id existing-flow \
  --output qasr_lab/outputs/evaluation/existing-flow
~~~

Every panel ID must appear exactly once. Empty hypotheses are valid; inference errors should be explicit rows with status set to "error".

## 3. Audit conversion before retraining

~~~sh
qasr_lab/.venv/bin/qasr-lab audit-transfer \
  --target /checkpoints/converted-qasr \
  --encoder /checkpoints/cohere-source \
  --decoder /checkpoints/audar-source \
  --qasr-source . \
  --output qasr_lab/outputs/audits/conversion
~~~

This reads final saved tensors and compares them with the original sources after destination-dtype conversion. It checks encoder mapping completeness, decoder transfer, and the final LM head/embedding alias. This catches the case where tying an originally untied source overwrites one of the two tensors. A mismatch on an already fine-tuned checkpoint is expected; use a freshly converted checkpoint to audit conversion. Tensor equality does not establish frontend or end-to-end numerical parity.

## 4. Run small, controlled pilots

Fill in [pilot.json](examples/pilot.json), then:

~~~sh
qasr_lab/.venv/bin/qasr-lab plan \
  --base qasr_lab/examples/pilot.json \
  --non-speech-panel qasr_lab/outputs/panels/pilot-non-speech \
  --output qasr_lab/outputs/plans/pilot-v1

qasr_lab/.venv/bin/qasr-lab train \
  --config qasr_lab/outputs/plans/pilot-v1/control.json
~~~

Plan creation only writes JSON configurations. Launch the variants individually:

| Variant | Change | Comparator |
| --- | --- | --- |
| control | Train the existing bridge; freeze encoder and decoder | Initial checkpoint |
| non_speech | Add an empty-transcript panel to the training mixture | control |
| upper_encoder | Also train the last four Conformer blocks, configurable | control |
| residual_bridge | Add LayerNorm → Linear → GELU → zero-initialized Linear after the existing bridge | control |
| ctc | Add compact transcript-only CTC, weight 0.1 | upper_encoder |

All configurations start from the same vanilla checkpoint and retain the same seed, validation panel, optimizer-step budget, batch size, accumulation, and learning rates. The CTC comparator deliberately has the same trainable encoder blocks. Adding non-speech changes the sample mixture and speech exposure; record that distinction when interpreting results.

The residual bridge starts with exactly the original function and retains the existing projector weights and names. This is a separate hypothesis from the production projector_depth option, which changes input normalization. The upper-encoder variant is the initial alternative to implementing encoder adapters.

CTC uses only raw transcript tokens, without language markers, prompt tokens, or EOS. A vocabulary built exclusively from the training panel maps observed decoder tokens to a compact head, with blank at zero. Empty transcripts receive blank-only supervision. A target is feasible only if frames cover its length plus adjacent repeated labels. Infeasible rows are counted, never silently zeroed by zero_infinity; runs fail acceptance if total feasibility is below 95%, configurable. A batch with no feasible rows fails immediately. Vocabulary overflow/OOV fails instead of dropping labels. Validation reports autoregressive CE, not joint CE+CTC.

For a checkpoint with pooling stride greater than one, set mask_pool_padding to true in a separate pilot to zero padded encoder states before convolution. Stride one is rejected as an ineffective experiment. Processor/model stride settings must already agree. The lab does not change token rates or add vLLM support.

The pilot runner is an independent **single-device** PyTorch loop, without the production launcher, Trainer, augmentation, distributed sampler, or optimizer state. It uses float32 parameters/AdamW state with optional bfloat16 autocast, gradient clipping, cosine decay, and optional non-reentrant activation checkpointing. Frozen BatchNorm buffers stay fixed. Incomplete batches are dropped so accumulation has a constant batch size; small pilot panels repeat as needed.

It saves configuration, provenance, parameter counts, a loss/CTC journal, initial/final validation CE, and a new final checkpoint. It does not resume production runs or provide fault-tolerant intermediate optimizer checkpoints. Large models still need adequate accelerator memory; this is not a replacement for the production distributed trainer.

Evaluate resulting checkpoints with backend set to "qasr-lab", which explicitly loads the experimental class. Saved lab checkpoints have a distinct qasr_lab model type, so ordinary automatic production loaders cannot silently ignore experimental weights. Lab extensions are not registered globally and are not wired into production serving. The lab's ordinary qasr backend rejects lab checkpoints. Keep them in the lab output tree.

## Existing saved evaluations

Rescore an existing result without audio, GPUs, or model downloads:

~~~sh
PYTHONPATH=qasr_lab/src python3 -m qasr_lab import-legacy \
  --input eval_results_en_commentary.json --mode offline \
  --output qasr_lab/outputs/historical/commentary
~~~

This generates a frozen panel and a fresh report using the saved references/hypotheses. Historical timing and held-out status remain unverified. It is a scoring demonstration, not evidence that the architecture improves generalization.

## Validation and interpretation

~~~sh
PYTHONPATH=qasr_lab/src:src qasr_lab/.venv/bin/python -m pytest -q qasr_lab/tests
qasr_lab/.venv/bin/python -m ruff check qasr_lab
~~~

Tests cover scoring, Unicode handling, leakage detection, panel checksums, failure accounting, paired intervals, baseline forward parity, masked pooling boundaries/gradients, residual checkpoint reloads, CTC feasibility/silence/backpropagation, tied-head transfer audits, and tiny CPU pilots followed by inference.

Tiny-model tests verify implementation mechanics. They do not verify accuracy, throughput, or stability of the real 3.6B-parameter model. Real Audar, Cohere, and QASR checkpoints must be evaluated on the same held-out panels before selecting a winner. Choose the first experiment from measured error slices, repeat promising results across seeds, and promote a change only after multilingual, non-speech, latency, and serving validation.
