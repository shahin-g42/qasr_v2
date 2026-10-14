# EmbeddingGemma 2 + Qwen speech: architecture decision

Investigation date: 2026-10-14. Objective: Arabic first, mandatory multilingual retention, compact offline deployment; training resources: 12 machines × 8 H100, 96 CPU cores per machine, CUDA 12.9. This document separates upstream evidence, implementation choices, and unmeasured hypotheses.

## Decision

**ASR audio-encoder replacement is feasible with retraining. TTS reference-speaker-encoder replacement is feasible with retraining. A TTS codec replacement would be a different research project and is excluded by the selected design.** The implemented candidate retains each Qwen generation model and swaps only its audio-conditioning encoder.

There is no evidence yet that this candidate is better than the original Qwen systems for Arabic, speed or size. Keep the native systems as controls. For minimum standalone size, the native Qwen speaker encoder is particularly hard to justify replacing; for ASR, the separate Gemma+CTC baseline is substantially smaller than keeping an autoregressive Qwen decoder.

## What was actually inspected

| Component | Observed contract | Consequence |
|---|---|---|
| EmbeddingGemma 2 | Multimodal embedding model; audio frontend expects 16 kHz, audio frames have 1536 output channels at approximately 25 Hz; retrieval embedding is 768-dimensional | Use audio **frames** and their valid-frame mask. Pooled retrieval vectors discard alignment. |
| Qwen3-ASR-0.6B | Audio encoder produces 1024-channel decoder inputs; the text decoder is Qwen3-shaped, 28 layers, 1024 hidden channels, vocabulary 151936 | A learned bridge and temporal-rate experiment are needed; tensor shape compatibility is insufficient. |
| Qwen3-TTS-12Hz-0.6B-Base | Speaker vector is 1024-dimensional; talker hidden size 1024; text embeddings project from 2048; 16 codec streams at roughly 12.5 Hz; output 24 kHz | Replace speaker conditioning while preserving text projection, talker, residual-code predictor and codec decoder. |

Sources: [EmbeddingGemma card](https://huggingface.co/google/embeddinggemma-2), [its exact config](https://huggingface.co/google/embeddinggemma-2/blob/914f7f89142e33e77833254d9c9b90c3cef7303b/config.json), [Qwen ASR exact config](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/5eb144179a02acc5e5ba31e748d22b0cf3e303b0/config.json), [Qwen TTS exact config](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/blob/5d83992436eae1d760afd27aff78a71d676296fc/config.json).

Counts below were calculated from checkpoint tensors, not inferred from model names:

| Tensor group | Parameters |
|---|---:|
| Gemma audio encoder | 304,824,608 |
| Original Qwen ASR audio encoder | 186,376,192 |
| Retained Qwen ASR decoder, with verified tied input/output embeddings | 596,049,920 |
| Implemented ASR bridge | 987,520 |
| **Gemma–Qwen ASR total** | **901,862,048** |
| Original Qwen TTS speaker encoder | 8,854,336 |
| Implemented Gemma speaker adapter | 2,301,697 |

The original ASR checkpoint serializes both embedding and LM-head tensors; actual values were compared before tying them. A naive sum of serialized tensors double-counts those tied parameters. The new ASR system adds about 119M parameters versus that original tower/decoder combination. Even sharing one Gemma encoder across both tasks does not by itself beat the combined 195M parameters of the two native audio encoders. Quantization, model sharing and CTC distillation are separate deployment experiments.

## Evidence from the linked experiment

The [linked LinkedIn post](https://www.linkedin.com/posts/shahirabdul_i-turned-google-deepminds-embeddinggemma-share-7515368854995447808-4-Nh/) led to the author's [EmbeddingGemma ASR demonstration](https://eg2.haku.sh/). It describes a frozen audio encoder and a small learned reader aligned to text embeddings, with an English word lexicon. Its speed and WER figures lack a sufficiently specified shared test set, normalization and hardware setup for a fair comparison here.

It is useful evidence that the audio representation contains decodable speech content. It does not establish Arabic recognition, open-vocabulary morphology, voice identity or TTS reconstruction. An English nearest-word system is a weak fit for Arabic clitics, inflection, diacritics, names and code switching. The implemented Qwen decoder retains a multilingual token vocabulary instead of a fixed word lookup table.

## Relevant research and what it changes

| Primary source | Relevance to this design |
|---|---|
| [Qwen3-ASR report, Jan 2026](https://arxiv.org/abs/2601.21337) | Establishes the retained decoder/audio model family and multilingual ASR baseline. It does not validate an encoder swap. |
| [Qwen3-TTS report, Jan 2026](https://arxiv.org/abs/2601.15621) | Supports keeping the talker, residual codec prediction and waveform codec together. The speaker interface is an adaptation boundary, not a waveform representation. |
| [Granite 5 TurboCTC, Sep 2026](https://arxiv.org/html/2609.20104v1) | Supports testing compact subword CTC, temporal reduction and intermediate supervision. Its reported batched-H100 speed is not comparable to the linked single-device demonstration. |
| [HLoRA multilingual CTC, Jan 2026](https://arxiv.org/html/2601.00557v1) | Motivates parameter-efficient multilingual adaptation as an additional experiment if partial unfreezing forgets other languages. LoRA is not implemented in this package. |
| [F5-TTS, 2024](https://arxiv.org/abs/2410.06885), [ZipVoice, 2025](https://arxiv.org/html/2506.13053v1) | Support a flow-matching comparison path with explicit lexical conditioning and temporal alignment. They do not show that semantic embeddings alone preserve pronunciation. |
| [More Data, Fewer Diacritics, Mar 2026](https://arxiv.org/html/2603.01622v1) | Motivates comparing Arabic scale and diacritization regimes. Preserve labels and measure vocalized versus unvocalized prompts; do not assume semantic conditioning resolves ambiguous vowels. |
| [SHAMS, Oct 2026](https://arxiv.org/abs/2610.01427) | Provides a recent Levantine pronunciation-oriented evaluation direction, alongside MSA and your held-out dialect data. |
| [Arabic constrained CTC diacritization, Jul 2026](https://arxiv.org/abs/2607.18946) | Motivates an explicit diacritization experiment, separate from recognition normalization. No constrained diacritizer is claimed here. |

The package includes simpler CTC and flow-matching comparison models; it is not a reproduction of every paper's full architecture or reported result.

## ASR architecture and training

Start with frozen Gemma audio → layer normalization → 1536→384→1024 MLP → optional adjacent-frame pooling → retained Qwen decoder. Preserve the original Qwen chat template, special tokens, end-of-turn token and language/transcript syntax. Train the bridge with supervised transcript cross-entropy. Then unfreeze the final four audio and decoder layers at a lower learning rate. Use Arabic-heavy multilingual replay throughout.

The decoder extraction preserves original tensor names and verifies strict loading. In the inspected ASR source, all three rotary-position axes use the same one-dimensional positions; native Qwen3 default RoPE therefore implements the same position geometry for this text/audio sequence. This is source-derived compatibility, supported by pretrained forward/backward/generation probes; it is not a transcript-parity or quality benchmark against the original audio encoder. [Official ASR implementation](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py).

Compare stride 1 versus 2, bridge width 384 versus 768, frozen versus partially unfrozen audio, and the original Qwen encoder. Alignment loss from the original tower and decoder-logit distillation are promising extensions, but are not implemented. Do not distill aggressively before checking whether the teacher preserves your Arabic dialect vocabulary.

Qwen ASR lists Arabic, Hindi, English and Chinese among its languages; Malayalam is an additional adaptation target in this project's scope. A multilingual tokenizer makes text representable but does not prove recognition competence. [Official supported-language list](https://github.com/QwenLM/Qwen3-ASR).

## TTS architecture and training

Use time-resolved Gemma audio features → masked attentive mean/standard-deviation pooling → 3072→512→1024 adapter. Train against Qwen's native speaker vector using cosine distance plus relative MSE, preserving vector magnitude. Hold out speakers and reference recordings. Then freeze that adapter and fine-tune the Qwen talker and all residual-code predictions against paired text/audio codec targets.

The implementation injects the adapted vector through `VoiceClonePromptItem` in speaker-vector-only mode, retaining the Qwen codec. A semantic audio encoder may suppress speaker identity; a small projection cannot recover information that is absent. Test verification EER, held-out speaker similarity, same-speaker cross-language consistency and synthesized-voice similarity. Mid-layer Gemma representations or limited audio-encoder unfreezing are follow-up experiments if the frozen-frame adapter fails. [Official speaker-prompt API](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py).

The original TTS model lists ten languages and excludes Arabic, Hindi and Malayalam. The implemented talker training uses existing Auto-language conditioning with Arabic/multilingual examples. It retains the existing codec vocabulary; it does not assign unused token IDs to new languages. Evaluate Arabic codec reconstruction before investing in talker training, because pronunciation cannot exceed the retained codec's acoustic coverage. [TTS model card](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base).

Teacher-forced codec losses are not synthesis quality. Generate held-out text with independent reference audio, score ASR consistency using an independent recognizer, and conduct blinded listening for pronunciation, naturalness and identity. Evaluate digits, dates, acronyms, names, code switching, long sentences, missing diacritics and dialect-specific words separately. Ground-truth target duration/audio must never be supplied at synthesis evaluation.

## Fair experiment plan for 96 H100s

The starting language mixture is Arabic 65%, English 15%, Hindi 7.5%, Chinese 7.5%, Malayalam 5%. This is a tunable sampling policy, not a statement about corpus sizes. Stratify Arabic by MSA and your actual dialect distribution; inspect per-language hours and speaker diversity before setting final weights.

1. Establish native Qwen ASR and speaker/TTS baselines on identical held-out data. For new TTS languages, compare against a native-speaker-encoder Qwen talker adapted on the **same** Arabic data and compute budget; stock unsupported-language output is not a fair trained baseline.
2. Run one-node, small-subset overfit checks and speaker probes. Reject catastrophic alignment/identity failures before scaling.
3. Screen stride, bridge width and freezing policies with matched examples and update counts. Keep seeds, durations, training hours and evaluation normalization fixed.
4. Move selected candidates to two nodes, then 12. Record effective batch, all-reduce overhead, data-loader wait, GPU memory and acoustic seconds per second. A larger global batch changes optimization; scale learning rate only after measuring.
5. Repeat the best candidates with multiple seeds. For ASR report WER/CER by language, dialect, noise, code switching and silence. For TTS report pronunciation and blinded listener scores with uncertainty, alongside speaker metrics.
6. Benchmark CPU and target offline device latency separately from H100 batch throughput: batch 1, warm/cold start, p50/p95, peak RAM/VRAM, encoder/decoder time, model bytes, and real-time factor `wall seconds / audio seconds`. Specify precision and backend. No quality/speed ranking is justified by synthetic tests.

A deployment candidate should pass a pre-agreed Arabic improvement target **and** multilingual regression limits against matched controls. With no target data paths, representative recordings or benchmark results supplied, the current outcome is a reproducible implementation and evidence-backed experiment design, not a demonstrated best speech model.
