# GQASR attribution

GQASR is a derived ASR system, not a claim of independently pretrained weights.

- Acoustic weights: Google DeepMind, google/embeddinggemma-2, revision 914f7f89142e33e77833254d9c9b90c3cef7303b. The model card identifies Apache-2.0: https://huggingface.co/google/embeddinggemma-2 and https://ai.google.dev/gemma/docs/gemma_4_license.
- Text decoder/tokenizer weights: Alibaba Qwen team, Qwen/Qwen3-ASR-0.6B, revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0, Apache-2.0: https://huggingface.co/Qwen/Qwen3-ASR-0.6B.
- Architecture implementations are supplied by Hugging Face Transformers (Apache-2.0), with Google and Alibaba/Hugging Face copyright notices in that dependency.

Modifications: removal of the original ASR audio tower, addition of a newly initialized audio-to-decoder bridge, GQASR module/config/tensor names, and independent ASR training/export code. Retained tokenizer tokens and pretrained tensor values are not rewritten to erase provenance. Original upstream products remain identifiable in this notice and provenance.json.

The accompanying Apache-2.0 license applies to the upstream materials. Retain their copyright and attribution notices when redistributing them.
