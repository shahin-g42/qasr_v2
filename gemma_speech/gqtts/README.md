# GQTTS

The TTS package inside `gemma_speech`, sharing its single environment with GQASR. Install the parent project; the `gqtts` command provides `cache`, `speaker-train`, `train` and `synthesize`.

The implementation replaces reference-voice speaker conditioning with an EmbeddingGemma audio encoder plus a trained speaker adapter, while retaining the Qwen TTS talker, code predictor and codec. Core TTS implementation lives in `src/gqtts/`; shared data utilities and the compatible private TTS port remain in the parent runtime. Legacy `gemma-speech` TTS commands remain available.

Use [the cluster runbook](../cluster/README.md) and `configs/speaker.json` / `configs/arabic.json`. These are multilingual Arabic-first TTS templates, not an automatic conversion of QASR's ASR corpus. Speaker IDs and suitable paired audio/text data are required. The cluster can export a standalone base before any training:

```bash
bash gemma_speech/cluster/gqtts_stage.sh download
bash gemma_speech/cluster/gqtts_stage.sh export
bash gemma_speech/cluster/gqtts_stage.sh verify --device cuda
bash gemma_speech/cluster/gqtts_stage.sh test -q
```

The artifact lives at `gemma_speech/gqtts/models/gqtts-base/`. It contains one `model.safetensors`, its architecture configuration, text processor, audio and codec frontends, checksums, provenance and notices. All state keys use `gqtts_` prefixes. Public Gemma/Qwen module class names become local GQTTS subclasses while retaining their computations; internal codec configuration identifiers retain the upstream schema needed by the retained codec.

The pretrained acoustic encoder, talker/code predictor and codec are preserved. The original reference speaker encoder is **excluded**; the replacement speaker adapter is **random** and must be trained. This is a training initialization, not a validated voice-cloning model. Keep the original upstream TTS checkpoint for teacher embeddings when preparing speaker-adapter training later.

`GQTTSBase.from_pretrained(path, device="cuda")` reloads the exported components locally without the original checkpoints. Its `gqtts_speaker_adapter`, `gqtts_generator.talker` and `api` expose the components needed for subsequent adaptation. The current cache/train commands retain their existing teacher-based workflow; they do not automatically initialize from this new artifact yet. ASR fine-tuning proceeds first; wiring the TTS training stages to this saved base comes with the later TTS fine-tuning work.

Verification checks hashes, offline loading, speaker-adapter gradients, codec round-trip and a short generation probe. It does not assess speaker similarity, Arabic pronunciation or multilingual quality. No local model export is performed by preparing these scripts.
