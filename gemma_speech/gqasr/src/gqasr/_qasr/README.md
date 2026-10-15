# QASR augmentation runtime

Snapshot of this repository's `src/qasr/{augmentation,audio,code_switch}.py`.
It is packaged inside gemma_speech so installing the original QASR model and
its Transformers dependencies is unnecessary. Keep attribution to QASR.

Acoustic augmentation/audio loading retain the original implementation. The
code-switch loader validates the same frozen version-2 plan without importing
`data_processing`. It accepts an explicit `retokenize` option: GQASR reuses the
immutable clip pools while computing candidate token counts with its own
runtime tokenizer. GQASR also validates full target lengths and duration bounds.
The original QASR runtime is unchanged.
