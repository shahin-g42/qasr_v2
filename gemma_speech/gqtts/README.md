# GQTTS

The TTS package inside `gemma_speech`, sharing its single environment with GQASR. Install the parent project; the `gqtts` command provides `cache`, `speaker-train`, `train` and `synthesize`.

The implementation replaces reference-voice speaker conditioning with an EmbeddingGemma audio encoder plus a trained speaker adapter, while retaining the Qwen TTS talker, code predictor and codec. Core TTS implementation lives in `src/gqtts/`; shared data utilities and the compatible private TTS port remain in the parent runtime. Legacy `gemma-speech` TTS commands remain available.

Use [the cluster runbook](../cluster/README.md) and `configs/speaker.json` / `configs/arabic.json`. These are multilingual Arabic-first TTS templates, not an automatic conversion of QASR's ASR corpus. Speaker IDs and suitable paired audio/text data are required. This reorganization does not claim a newly trained or independently exported GQTTS base checkpoint.
