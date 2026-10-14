# Qwen 12 Hz compatibility port

Source: Alibaba Qwen team's `qwen-tts==0.1.1` distribution, [upstream repository](https://github.com/QwenLM/Qwen3-TTS). The original files are licensed under Apache-2.0; copyright notices and the full license are retained in this directory. `UPSTREAM.json` records SHA-256 hashes of every original copied source file.

This is a modified, limited-scope copy maintained inside `gemma_speech`, not the unmodified Qwen SDK. It supports the Qwen3-TTS 12Hz Base checkpoint and its codec on Transformers 5.19.0. Changes:

- Move imports into a private package namespace; omit 25 Hz codec imports and their unrelated dependencies.
- Replace the removed decorator factory with `merge_with_config_defaults`; document the legacy codec cache-position argument.
- Preserve the upstream default rotary-embedding calculation locally after its removal from the Transformers RoPE registry. Other RoPE variants are not supported by this port.
- Adapt causal-mask argument names and remove the obsolete mask `cache_position` argument, using local wrappers.
- Express untied codec-head metadata in the Transformers 5 dictionary format and correct the talker's output-embedding accessor to return its codec head.
- Supply explicit fallbacks for config attributes no longer supplied by the Transformers base configuration class; use current layer-type validation.

No Transformers globals are modified. Checkpoint tensor names and dimensions are preserved. The original SDK's CLI, 25 Hz tokenizer, gradio application and packaging metadata are not included. See the package verification records for tested paths and limitations.
