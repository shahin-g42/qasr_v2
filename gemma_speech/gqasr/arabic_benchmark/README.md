# GQASR Arabic leaderboard benchmark

This package evaluates a saved GQASR checkpoint on every test set listed by the [Open Universal Arabic ASR Leaderboard](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard): SADA, Common Voice 18 Arabic, MASC clean, MASC noisy, MGB-2, and Casablanca. The live leaderboard uses an equal-weight mean of the six per-dataset WERs for ranking; this benchmark reports that same mean under the leaderboard's Arabic text normalization, plus per-dataset WER/CER, strict WER/CER, optional metadata groups, and generation diagnostics.

The source manifests and scoring recipe are pinned to the upstream evaluation repository revision and SHA-256 values in `src/gqasr/arabic_benchmark/catalog.json`. The upstream [evaluation code](https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard) removes punctuation and Arabic diacritics, normalizes hamza/madda forms and Eastern Arabic digits, and attaches a standalone `و` to the following word. ELM CER counts spaces. We implement this normalization locally so inference does not require NeMo. Strict scores preserve punctuation, diacritics, and case after whitespace normalization.

## Dataset version note

The live Space currently lists MASC noisy-test as 14.9 hours. The checksum-pinned test manifest in the official code repository sums to 8.925 hours, matching the paper's 8.9-hour description. The benchmark deliberately uses the available pinned manifest, verifies its hash, and prints this discrepancy in every report. Treat the MASC noisy score as version-specific; resolve the upstream manifest change before claiming an exact current leaderboard reproduction. The full pinned six-manifest run is still useful for consistent internal model comparisons.

The pinned suite has 46,757 utterances and about 60.0 hours based on manifest durations. Choose remote test streaming below, or use existing local audio with the original local-data stages. Casablanca test audio comes from the pinned `UBC-NLP/Casablanca` dataset revision; its dataset card specifies CC-BY-NC-ND 4.0.

## Remote test streaming (no manually downloaded dataset folders)

The `stream-*` stages read test audio remotely and keep only prepared 16 kHz mono benchmark clips, approximately 14 GB for all six sets. They do not download training splits, save full archives, or extract source datasets. Small compressed clips are decoded through temporary files and immediately removed. Streaming still transfers audio over the network, and Parquet readers may fetch complete pages/row groups. This is streamed **preparation**, followed by local-cache inference: 88 model replicas share the cache instead of making 88 independent network streams. It is not zero-disk live inference.

Pinned source revisions are in `src/gqasr/arabic_benchmark/stream_sources.json`:

| Dataset | Remote access and exact sample matching |
|---|---|
| SADA | `khaledalganem/sada2022`: range-read original WAV recordings, seeking to the official start/end samples |
| Common Voice 18 Arabic | `fsicoli/common_voice_18_0`: stream the Arabic test tar; match official clip basenames; Hugging Face access approval required |
| MASC clean/noisy | `MohamedRashad/MASC-Arabic`: test Parquet audio, mapped through pinned `pain/MASC` CSV video/start/type metadata; verify text and duration |
| MGB-2 | `MohamedRashad/mgb2-arabic`: test Parquet audio matched by exact `segment_id` |
| Casablanca | `UBC-NLP/Casablanca`: publisher test Parquet audio matched by exact `seg_id` |

These include public mirrors, not just original publishers. Preparation preserves the official ELM references and checks identity, duration, and complete coverage. It does not independently prove byte equivalence to a separately obtained original-provider copy; reports retain source revisions and disclose this limit. No fuzzy transcript matching or shuffled-row alignment is used. MASC's duplicate official clip is retained as two evaluation rows; identical video/start/end/text aliases in the source mapping resolve to one deterministic audio alias.

For Common Voice, review/accept the access conditions on [the dataset page](https://huggingface.co/datasets/fsicoli/common_voice_18_0), then authenticate on the cluster with `gemma_speech/.venv/bin/hf auth login` (or use an already configured `HF_TOKEN`). The scripts do not accept terms on your behalf. A denied source fails explicitly; it is never silently dropped from the six-set average.

From the repository root:

```bash
export GQASR_BENCH_INPUT=stream
export GQASR_BENCH_WORK="$PWD/gemma_speech/gqasr/verification/arabic-stream-cache"
export GQASR_BENCH_CHECKPOINT="$PWD/gemma_speech/gqasr/outputs/full_b40_frozen/final"
export GQASR_BENCH_OUTPUT="$PWD/gemma_speech/gqasr/verification/arabic-final-streamed"
# GQASR_BENCH_DATA is not required. WORK may instead be a node-local NVMe path.
# Unset an old custom config override when switching to the provided stream config.
unset GQASR_BENCH_CONFIG

bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh stream-check &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh stream-smoke &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh doctor &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh stream-prepare &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh preflight &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh smoke &&
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh run
```

`stream-check` validates remote access and ID metadata without fetching audio columns. Common Voice tar member coverage and actual audio contents are checked during preparation. `stream-smoke` decodes two official clips per dataset (override `--per-dataset`); its partial cache and report live under `WORK/stream-smoke` and cannot masquerade as a full prepared dataset. These three remote stages require only `GQASR_BENCH_WORK`, no GPU or checkpoint.

`STREAM_WORKERS=16` bounds concurrent network readers and decoders independently of GPU workers; increase it if network capacity permits. Streaming uses threads for network I/O, libsndfile/SciPy decoding, and Parquet reads, with one BLAS thread per process. Preparation prints progress, verifies durations and hashes, and publishes full manifests only after exact coverage succeeds. Rerunning it reuses valid clips and repairs damaged cache entries. A partially read tar must be streamed again to reach uncached members; a completely validated cached archive is not reread. Authentication, socket timeouts, or read failures stop the stage and retain valid per-clip caches for a rerun. No remote dataset Python scripts are executed.

To test the five public sets while waiting for Common Voice access, explicitly pass `--datasets sada,masc_clean,masc_noisy,mgb2,casablanca` to `stream-check`, `stream-smoke`, and `stream-prepare` (and to inference stages if evaluating just those sets). This produces a declared partial benchmark, not the six-set leaderboard average.

## Existing local audio

| Test set | Rows | Hours in pinned manifest | Local data root |
|---|---:|---:|---|
| SADA | 6,189 | 10.75 | `$GQASR_BENCH_DATA/sada` |
| Common Voice 18 Arabic | 10,471 | 12.56 | `$GQASR_BENCH_DATA/commonvoice18` |
| MASC clean | 8,612 | 10.49 | `$GQASR_BENCH_DATA/masc/test-clean` |
| MASC noisy | 9,394 | 8.92 | `$GQASR_BENCH_DATA/masc/test-noisy` |
| MGB-2 | 5,365 | 9.58 | `$GQASR_BENCH_DATA/mgb2` |
| Casablanca | 6,726 | 7.70 | `$GQASR_BENCH_WORK/casablanca-test-audio` |

If your files use another directory layout, edit the roots in `configs/arabic_leaderboard_6set.yaml`. The resolver searches each root recursively by the official audio basename. If a basename is missing or ambiguous, supply an `audio_map` JSONL file for that dataset. Each line maps an official `relative_audio` key to a path under that dataset root; optional `offset_seconds` and `duration_seconds` fields select a segment from an original recording. For example:

```jsonl
{"relative_audio":"clips/sample.wav","path":"recordings/show.wav","offset_seconds":12.4,"duration_seconds":3.2}
```

The preflight and preparation steps fail on unresolved files or duration mismatches; they do not skip test utterances.

## Local data and protocol checks

Run the executable regression tests and a small public-data smoke test without model weights:

```bash
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh test -q
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh source-smoke
```

`source-smoke` streams two official test utterances from each of Casablanca's eight dialects using ranged parquet reads, then runs the same decoding, 16 kHz resampling, ID matching, and duration validation used by full preparation. Its default output is `gemma_speech/gqasr/verification/arabic-source-smoke/source-smoke.json`. Override the destination with `GQASR_BENCH_SOURCE_SMOKE_OUTPUT`, or pass `--per-dialect 4 --workers 4`. Streaming may read whole parquet pages or row groups even for a small sample. It never downloads model weights or marks a partial sample as a completed benchmark.

The offline regression suite exercises real parquet extraction, missing-source detection, audio resampling and cache repair, CPU multiprocessing preparation, spawned inference scheduling with a dummy recognizer, partial-run resume, reference integrity, OOM batch splitting, and official scoring rules. Real H100 model execution is checked by the separate `smoke` stage below.

## One-node H100 run

Use one 8×H100 node. Put the work/cache and result directories on node-local NVMe when possible; the prepared 16 kHz mono NumPy audio is approximately 14 GB, plus metadata and prediction outputs. The original dataset roots can stay on shared storage. Point the checkpoint at the selected saved checkpoint directory; it must contain `config.json`, `preprocessor_config.json`, `model.safetensors`, and the `tokenizer/` directory.

```bash
export GQASR_BENCH_DATA=/lustrefs/shared/your-user/asr-benchmark-data
export GQASR_BENCH_WORK=/local_nvme/your-user/gqasr-arabic-benchmark
export GQASR_BENCH_CHECKPOINT=/lustrefs/shared/your-user/qasr_v2/gemma_speech/gqasr/outputs/full_b40_frozen/final
export GQASR_BENCH_OUTPUT="$GQASR_BENCH_WORK/results/full_b40_frozen"

bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh doctor
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh references
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh download-casablanca
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh preflight
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh prepare
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh smoke
bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh run
```

Install or update the unified environment from the repository root before using the new entry point:

```bash
gemma_speech/.venv/bin/python -m pip install -e './gemma_speech[dev]'
```

The `doctor` stage requires the shared Torch 2.11 / Transformers 5.19 environment, the benchmark's Python dependencies, `ffmpeg`, and checks that all eight allocated GPUs support BF16. `references` fetches and checks the upstream test manifests. `download-casablanca` fetches only test shards. `preflight` checks all test audio paths before decoding. `prepare` uses 96 CPU processes by default to decode, resample to 16 kHz mono, verify durations, and cache audio with provenance hashes. Each stage prints progress and writes a log under `gemma_speech/logs/`.

The dedicated-node launcher sets `CUDA_VISIBLE_DEVICES` from `GPU_IDS`, defaulting to `0,1,2,3,4,5,6,7` after environment activation, so doctor and inference use the same devices.

The default `smoke` evaluates 128 utterances per dataset with the requested 11 workers per GPU, checking that all 88 replicas start and the workload completes before the full run. It writes to a separate `-smoke` output directory. Run `tune` optionally to compare batch sizes 1, 2, and 4 using all eight GPUs and 11 workers on every GPU; it selects a batch size with no OOM splitting when one is available. Tuning measures completed-subset throughput including startup and any resumed attempts. The full `run` defaults to batch size 1 and starts **88 independent workers total (11 per H100)**. Each process has a bounded CUDA memory share, uses one CPU thread, loads one persistent model replica, and receives work from a shared queue. The two decoding modes share each prepared waveform and frontend feature pass.

To override the worker or batch settings without editing code:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 WORKERS_PER_GPU=11 BATCH_SIZE=1 \
  bash gemma_speech/cluster/gqasr_arabic_leaderboard.sh run
```

The run is resumable: completed atomic task files are reused only when the checkpoint contents, prepared-audio checksums, pinned protocol, sample selection, decoding and batch settings, runtime versions, and implementation match. The implementation hashes include GQASR's model and processor code. GPU count and worker count may change on resume; changing batch size creates a different run identity because batching can affect numerical outputs. Progress is emitted as JSON every five seconds, with a final 100% state after complete coverage has been validated. Startup or inference with no task completion for 30 minutes fails with a diagnostic; adjust `startup_timeout_seconds` and `progress_timeout_seconds` in the YAML if necessary. Completed task files remain available for resume. Worker-specific tracebacks and memory peaks are retained in the output directory. The coordinator starts no more model replicas than there are inference tasks; the full run, smoke run, and default tuning run each create at least 88 tasks and therefore use 11 replicas on each of the eight GPUs.

## Decoding modes and result interpretation

The `arabic` mode adds only the `language Arabic<asr_text>` prompt; the `auto` mode lets GQASR emit its language header. Both use deterministic greedy decoding. Arabic-conditioned results are the primary task-conditioned GQASR score; this does not imply identical decoding controls to every other model's leaderboard submission. Automatic language generation is a useful diagnostic. The report saves predictions for both and shows the official ELM-normalized metrics beside strict metrics. It also includes the current leaderboard snapshot for context, but does not claim a rank when the MASC manifest version or training overlap has not been resolved.

The final report and `scores.json` are written to `$GQASR_BENCH_OUTPUT`. Prediction manifests retain the test reference, raw model output, decoded text, language header, generated token count, EOS/limit status, audio checksum, and any available dialect/speaker metadata. `attempts.json`, `progress.json`, and `worker-*.log` document performance and resumable execution. The reported throughput is concurrent H100-node throughput and should not be compared directly to the paper's single-A100, batch-one, warmed-up RTF.

`scores.json` records `complete_pinned_suite_scored` separately from `comparable_to_leaderboard`. The latter remains false while the MASC version, training exposure, and submission-specific decoding differences are unresolved; the published table is context only.

The leaderboard asks model authors to avoid training on the same test datasets. Audit QASR and upstream-model training exposure before describing GQASR as zero-shot or submitting a leaderboard result. The GQASR report explicitly leaves this overlap unaudited.
