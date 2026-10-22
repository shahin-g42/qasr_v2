# Native decoder tokens for audio categories

This opt-in extension works with AudPro, AudTurbo and AudFlash. It adds four
dedicated special tokens with the tokenizer's `add_special_tokens` API and
resizes the decoder embeddings/output vocabulary only when necessary. There is
no separate classification head or change to the audio encoder or bridge.

| Category | Supervised completion |
| --- | --- |
| speech | `<\|audio_speech\|>language English<asr_text>Hello.<EOS>` |
| music | `<\|audio_music\|><EOS>` |
| noise | `<\|audio_noise\|><EOS>` |
| silence | `<\|audio_silence\|><EOS>` |

The backslashes above are Markdown table escapes; the actual token strings are
`<|audio_speech|>`, `<|audio_music|>`, `<|audio_noise|>` and `<|audio_silence|>`.
`<EOS>` represents the checkpoint's existing EOS token, not an added token.
For speech, the native language marker, transcript and EOS are preserved exactly
after the category token. Non-speech has no generated language marker or text.
The category is a supervised output, never supplied as an inference hint.

## Labeling contract

These are mutually exclusive **clip-level** labels, not event timestamps:

- **speech:** intelligible speech, including speech over music or background
  noise. Keep the transcript and native language supervision.
- **music:** instrumental music with no speech or vocals.
- **noise:** environmental/non-musical sound with no speech or vocals.
- **silence:** curated silence or near-silence. Audible room tone belongs to
  noise. Use a consistent annotation policy across the corpus.

Exclude uncertain clips, singing, and indistinct vocalizations from the initial
negative pool. Do not label all files in a noise dataset empty automatically.
VAD screening can help curate candidates but does not prove absence of speech.
Keep train/eval recordings disjoint before segmenting them.

Normal speech manifests remain unchanged. A negative row must explicitly declare
both its category and empty text:

```json
{"audio_filepath":"/shared/negatives/rain.wav","duration":4.2,"text":"","audio_type":"noise"}
{"audio_filepath":"/shared/negatives/instrumental.wav","duration":8.0,"text":"","audio_type":"music"}
{"audio_filepath":"/shared/negatives/silence.wav","duration":3.0,"text":"","audio_type":"silence"}
```

Use `robustness.non_speech.manifests`, not language keys named `noise` or `music`.
The pool samples uniformly across eligible records, so curate its category mix
deliberately. The example recipe replaces 3% of training slots in expectation;
this is a starting recipe requiring evaluation, not a demonstrated optimal ratio.
Speech quotas and batch size retain the behavior documented in [ROBUSTNESS.md](ROBUSTNESS.md).

Typed silence bypasses waveform effects and SpecAugment. Music/noise examples
do not receive real-noise mixing, since a mixed noise corpus can contain speech
or music that changes the label. Music also skips additive white noise. Other
bounded robustness effects remain available.
When using the older, non-category-aware augmenter, all typed negatives bypass
augmentation. Speech retains the configured augmentation and code-switch policy.

## Upgrade an existing AUD artifact

Use an aligned adapter or trained checkpoint as the source. The command saves
a new one-piece artifact and leaves the source untouched. Existing token IDs,
assigned embedding rows, special tokens, and encoder/bridge weights are preserved.
New rows use the mean of the pretrained token representations of their category
word. This is an initialization, not a trained audio detector.

Some pretrained decoders have more embedding rows than assigned tokenizer IDs.
The extension can use those spare rows without increasing tensor dimensions;
it never shrinks the decoder or overwrites an assigned token's row. Tied input
and output weights remain tied, and untied heads are initialized separately.
The upgraded tokenizer, model config, generation config and weights are saved
together; reload validates the versioned category-to-ID map. Provenance and
checksums accompany the artifact.

Run from the repository root in the existing AUD environment:

```bash
export PYTHONPATH="$PWD/audm/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export AUD_TYPED_ROOT="$PWD/output/audflash-audio-types-v1"
export AUD_TYPED_SOURCE=/REPLACE/WITH/aligned-or-trained-audflash
export AUD_NON_SPEECH_MANIFEST=/REPLACE/WITH/categorized-non-speech-train.jsonl
export AUD_RIR_MANIFEST=/REPLACE/WITH/rir.jsonl

python -m audm add-audio-types \
  --model-path "$AUD_TYPED_SOURCE" --output "$AUD_TYPED_ROOT/export"
python -m audm verify "$AUD_TYPED_ROOT/export"
```

Keep the tokenizer and model from the same artifact at deployment. Do not add
tokens a second time or manually edit token IDs. The upgrade refuses an existing
destination, collisions, or an already-extended checkpoint. It imports weights,
not old optimizer state; this is a new fine-tuning run, not an exact resume.

## AudFlash: decoder warm-up, then full training

New token embeddings and LM-head rows need training. Adapter-only training
freezes them, so it is rejected for typed checkpoints. The supplied warm-up
trains the decoder and bridge while keeping the encoder frozen. It uses 2,000
updates, 100 warm-up updates, no label smoothing, and evaluation/checkpoints
every 500 updates. Base LR is 2e-5, decoder factor 0.5 and embedding factor 0.25.
Use category metrics and ordinary multilingual WER/CER to decide whether the
warm-up is sufficient before full training.

Both stages retain **10 × 8 × 8 = 640** examples/update and four loader workers
per GPU. Full training uses three nominal weighted coverage epochs, with
evaluation and checkpoints every 3,700 updates. Production configs and run roots
are unchanged. FFmpeg, RIR assets and the real-noise corpus inherited from the
stage-02 data recipe must be available; see [ROBUSTNESS.md](ROBUSTNESS.md).

```bash
export AUD_USE_SLURM=0 AUD_NNODES=1 AUD_GPUS_PER_NODE=8 AUD_NODE_RANK=0
export AUD_MASTER_ADDR=127.0.0.1 AUD_MASTER_PORT=29613 AUD_COMMAND=train
export AUD_TYPED_DECODER_CONFIG="$PWD/audm/configs/audflash_audio_types_decoder.yaml"
export AUD_TYPED_FULL_CONFIG="$PWD/audm/configs/audflash_robust_full.yaml"

python -m audm preflight --config "$AUD_TYPED_DECODER_CONFIG" \
  --model "$AUD_TYPED_ROOT/export" --output "$AUD_TYPED_ROOT/decoder" --scan
bash audm/scripts/train_distributed.sh --config "$AUD_TYPED_DECODER_CONFIG" \
  --model "$AUD_TYPED_ROOT/export" --output "$AUD_TYPED_ROOT/decoder" --smoke
bash audm/scripts/train_distributed.sh --config "$AUD_TYPED_DECODER_CONFIG" \
  --model "$AUD_TYPED_ROOT/export" --output "$AUD_TYPED_ROOT/decoder"
```

Evaluate the decoder warm-up checkpoint with the next section's command, inspect
predictions, then proceed:

```bash
python -m audm preflight --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_TYPED_ROOT/full" --scan
bash audm/scripts/train_distributed.sh --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_TYPED_ROOT/full" --smoke
bash audm/scripts/train_distributed.sh --config "$AUD_TYPED_FULL_CONFIG" \
  --model "$AUD_TYPED_ROOT/decoder" --output "$AUD_TYPED_ROOT/full"
```

These are direct launches on the node, with no scheduler required. Other AUD
variants use the same upgrade and supervision code, with their own node/batch
recipe. No corpus is downloaded by the upgrade or training commands.

## Structured inference and held-out evaluation

Generate normally using `processor.apply_transcription_request([wave])`.
Omit the language hint when evaluating automatic detection. Decode **completion
IDs only**, after removing the input prompt:

```python
completion = tokens[0, batch["input_ids"].shape[1]:]
event = processor.decode(completion, return_format="audio_event")
# {"audio_type": "speech", "language": "English", "text": "Hello.", "format_valid": True}
text = processor.decode(completion)  # Existing transcription-only API.
```

The parser does not suppress hallucinations after a non-speech category. It
returns their text and marks the format invalid. Missing category/EOS is also
invalid. Unknown categories are returned as `audio_type: null`. Do not interpret
a generated category as calibrated confidence or use it as a hard VAD gate
without measuring missed speech. Old, non-extended artifacts retain old behavior.

Build a representative held-out manifest with all four labels; speech rows also
need their transcript and language. The evaluator uses the first `--limit`
records in file order, with no reference category or language hint:

```bash
python -m audm.eval_audio_types --model-path "$AUD_TYPED_ROOT/decoder" \
  --manifest /REPLACE/WITH/audio-types-eval.jsonl \
  --device cuda:0 --limit 1000 --max-new-tokens 768 \
  --output "$AUD_TYPED_ROOT/decoder-audio-types-eval.json"
```

Reports confusion counts, per-class precision/recall, category accuracy,
invalid-format rate, missed-speech rate, non-speech text rate and token-cap hits,
with per-record predictions. Category accuracy scores the category independently
of format validity, so inspect both. Continue using the normal multilingual
WER/CER callback to check transcription regressions; empty references are not
included in ordinary WER. Full-size quality, GPU memory and throughput must be
measured on the cluster; local tests use tiny models.

See [ROBUST_CLUSTER.md](ROBUST_CLUSTER.md) for the audited cluster runbook, deeper
asset scan, distributed validation commands and restart procedure.
