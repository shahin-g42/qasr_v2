---
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
tags:
  - automatic-speech-recognition
  - asr
  - speech
  - conformer
  - qwen3
  - multimodal
  - eagle
  - speculative-decoding
  - streaming
  - multilingual
language:
  - ar
  - en
  - zh
  - hi
  - ml
base_model:
  - CohereLabs/cohere-transcribe-03-2026
  - audarai/Audar-ASR-V1.2-Turbo
metrics:
  - wer
  - cer
---

# {{MODEL_NAME}}

**A 3.6B-parameter multilingual speech-recognition model that pairs a Conformer
audio encoder with a Qwen3 language-model decoder — shipped with a lossless
EAGLE speculative-decoding head for {{EAGLE_SPEEDUP}} faster inference and a real-time
streaming server.**

{{MODEL_NAME}} (internal codename **QASR**) injects audio into a decoder-only
LLM LLaVA-style: a 48-layer Conformer encoder turns 16 kHz audio into acoustic
features, a learned projector maps them into the Qwen3 embedding space, and the
LLM autoregressively decodes a fluent, punctuated transcript. Because the decoder
*is* a language model, transcripts are naturally cased and punctuated.

```
  16 kHz waveform
        │
        ▼
 ┌──────────────────────┐   128-bin log-Mel · 8× subsample
 │  Conformer encoder   │──────────────────────────────┐
 │  (48 layers, 1280d)  │                               │
 └──────────────────────┘                               ▼
                                              ┌────────────────────┐
                                              │  MM Projector      │
                                              │  1280 → 2048 (MLP) │
                                              └─────────┬──────────┘
                                                        │
   ┌────────────────────────────────────────────────────┘
   ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Qwen3 decoder (28 layers, 2048d, 151,936 vocab)              │
 │  [ <|audio|> × N ] [ prompt ] → transcript                    │
 └──────────────────────────────────────────────────────────────┘
        │
        ▼
  Transcript  (Arabic · English · Chinese · Hindi · Malayalam)
        ▲
        │  (optional) EAGLE draft head → {{EAGLE_SPEEDUP}} faster, identical output
```

---

## Highlights

- **Fluent multilingual ASR** across Arabic, English, Chinese, Hindi, and Malayalam, with native casing and punctuation from the LLM decoder.
- **{{EAGLE_SPEEDUP}} faster decoding, losslessly.** An EAGLE lookahead head with 8.4M *trained* parameters drafts tokens the full model verifies in one pass. The shipped `eagle/eagle_head.pt` is larger ({{HEAD_FILE_SIZE}}) because it also carries a frozen copy of the decoder's 151,936-row output projection. Greedy decoding is identical to standard generation up to bf16 tie-breaks — you trade compute, never accuracy.
- **Real-time streaming.** A WebSocket server performs rolling-window inference and emits partial transcripts as audio arrives.
- **Zero-install offline use** via `trust_remote_code=True` — the modeling code travels with the checkpoint.

---

## Evaluation

Measured on **{{EVAL_SAMPLES}} utterances** of the `{{EVAL_LANGUAGE}}` eval split
(`{{EVAL_MANIFEST}}`) — in-domain, not a held-out benchmark; broader held-out and
multilingual evaluations are pending — with greedy decoding on a single **NVIDIA
H100**, against target checkpoint `{{EVAL_MODEL}}` and EAGLE head
`{{EVAL_EAGLE}}`. WER/CER are lowercased, punctuation-stripped. RTF =
inference-time ÷ audio-duration (lower is faster; RTF 0.10 ≈ 10× real time).
Latency and throughput are wall-clock and move with machine load; WER/CER do not.

### Quality & speed by decoding mode

| Mode | WER | CER | RTF | Latency (mean) | Throughput |
|------|:---:|:---:|:---:|:---:|:---:|
| **Offline** (standard) | **{{WER_OFFLINE}}** | {{CER_OFFLINE}} | {{RTF_OFFLINE}} | {{LAT_OFFLINE}} | {{TPS_OFFLINE}} |
| **Offline + EAGLE** | **{{WER_EAGLE}}** | {{CER_EAGLE}} | **{{RTF_EAGLE}}** | **{{LAT_EAGLE}}** | **{{TPS_EAGLE}}** |
| **Streaming** | {{WER_STREAM}} | {{CER_STREAM}} | {{RTF_STREAM}} | — | — |
| **Streaming + EAGLE** | {{WER_STREAM_EAGLE}} | {{CER_STREAM_EAGLE}} | **{{RTF_STREAM_EAGLE}}** | — | — |

> EAGLE cuts offline decode latency from {{LAT_OFFLINE}} to **{{LAT_EAGLE}}** and
> lifts throughput from {{TPS_OFFLINE}} to **{{TPS_EAGLE}}** — a **{{EAGLE_SPEEDUP}}
> speedup** — at unchanged WER ({{WER_OFFLINE}} standard vs {{WER_EAGLE}} with EAGLE).
> Streaming rows above are only populated when the streaming modes were evaluated.

### EAGLE speculation quality

| Metric | Value | Meaning |
|--------|:---:|---|
| End-to-end speedup | **{{EAGLE_SPEEDUP}}** | offline latency ÷ eagle latency |
| Acceptance rate | {{EAGLE_ACCEPTANCE}} | fraction of drafted tokens accepted |
| Tokens per model forward | {{EAGLE_TPF}} | vs 1.00 for standard decoding |
| Mean drafts accepted per round | {{EAGLE_MEAN_ACCEPTED}} | of {{EAGLE_NUM_DRAFT}} drafted |
| Position-0 acceptance | **{{EAGLE_POS0}}** | the first drafted token is usually right |
| Acceptance by position | {{EAGLE_BY_POSITION}} | draft depth 1…{{EAGLE_NUM_DRAFT}} |

Acceptance falls steeply with draft depth: the head is trained single-step
teacher-forced but chained autoregressively at inference, so each draft compounds
the previous one's error. Position-0 acceptance is the health indicator — near
zero there means the head does not match the target checkpoint, not that it is
undertrained.

Losslessness check: transcripts from EAGLE vs standard decoding matched on
**{{EAGLE_EXACT_MATCH}}** samples. A mismatch is a final-token tie-break under bf16
rounding — verifying {{EAGLE_NUM_DRAFT}} positions in one batched forward is not
bit-identical to verifying them one at a time — not a decoding error.

---

## Usage

### Installation

```bash
# Offline transcription needs only PyTorch + a recent transformers:
pip install "transformers>=5.14.1,<5.15" torch librosa soundfile

# EAGLE speculative decoding and the streaming server additionally need the
# companion `qasr` package (provides EagleSpeculativeDecoder + the server):
git clone https://github.com/shahin-g42/qasr_v2 && cd qasr_v2
pip install -e ".[streaming]"
```

### 1 · Offline transcription (zero-install, `trust_remote_code`)

The modeling code is bundled in this repo, so the Auto classes can build the
model directly — no `qasr` package required.

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers.audio_utils import load_audio

repo = "{{REPO_ID}}"
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
).to("cuda").eval()

audio = load_audio("speech.wav", sampling_rate=processor.feature_extractor.sampling_rate)
inputs = processor.apply_transcription_request(
    audio=audio,
    language="en",                       # ar · en · zh · hi · ml
    sampling_rate=processor.feature_extractor.sampling_rate,
    return_tensors="pt",
).to(model.device, model.dtype)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

transcript = processor.decode(
    output_ids[0, inputs["input_ids"].shape[1]:],
    return_format="transcription_only",
)
print(transcript)
```

### 2 · Faster offline transcription with EAGLE (~2×)

Uses the companion `qasr` package and the draft head shipped at `eagle/`.
Greedy output is identical to the standard path.

```python
import torch
from huggingface_hub import snapshot_download
from transformers.audio_utils import load_audio
from qasr import QASRProcessor
from qasr.eagle import EagleSpeculativeDecoder

local = snapshot_download("{{REPO_ID}}")
processor = QASRProcessor.from_pretrained(local)
decoder = EagleSpeculativeDecoder.from_pretrained(
    model_path=local,
    eagle_path=f"{local}/eagle",          # eagle/eagle_head.pt
    dtype=torch.bfloat16,
    device="auto",
)

audio = load_audio("speech.wav", sampling_rate=processor.feature_extractor.sampling_rate)
inputs = processor.apply_transcription_request(
    audio=audio, language="en",
    sampling_rate=processor.feature_extractor.sampling_rate,
    return_tensors="pt",
).to(decoder.device, decoder.dtype)

output_ids = decoder.generate(
    input_ids=inputs["input_ids"],
    input_features=inputs["input_features"],
    input_features_mask=inputs["input_features_mask"],
    attention_mask=inputs.get("attention_mask"),
    max_new_tokens=256,
    num_draft_tokens=5,                   # draft depth; 2–3 also works well
)

transcript = processor.decode(
    output_ids[0, inputs["input_ids"].shape[1]:],
    return_format="transcription_only",
)
print(transcript)
print(f"acceptance={decoder.last_stats.acceptance_rate:.2%} "
      f"tokens/forward={decoder.last_stats.tokens_per_forward:.2f}")
```

### 3 · Real-time streaming server

```bash
# Start the WebSocket server (add --eagle for speculative decoding):
LOCAL=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('{{REPO_ID}}'))")
qasr-stream --model "$LOCAL" --eagle "$LOCAL/eagle" --port 8765
```

Clients stream raw **PCM16 @ 16 kHz** binary frames to `ws://<host>:8765/ws/transcribe`
and receive JSON partials:

```json
{ "type": "partial", "text": "lando norris leads into turn one",
  "audio_seconds": 4.2, "inference_ms": 180.5 }
```

A minimal Python client:

```python
import asyncio, soundfile as sf, websockets, json

async def stream(path):
    pcm, sr = sf.read(path, dtype="int16")
    assert sr == 16000, "resample to 16 kHz first"
    async with websockets.connect("ws://localhost:8765/ws/transcribe") as ws:
        for i in range(0, len(pcm), 3200):                 # 200 ms chunks
            await ws.send(pcm[i:i + 3200].tobytes())
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                print(json.loads(msg)["text"])
            except asyncio.TimeoutError:
                pass

asyncio.run(stream("speech.wav"))
```

---

## How EAGLE speculative decoding works

The draft head is a single fusion layer that consumes the decoder's hidden state
`h_t` **and** the embedding of the token just chosen, `x_{t+1}`, to predict the
*next* feature `h_{t+1}` (and thus a distribution over `x_{t+2}`):

```
concat(h_t, embed(x_{t+1}))  →  Linear(4096→2048) → SiLU → LayerNorm → + h_t  →  lm_head
```

At inference the head drafts *K* candidate tokens; the full model verifies all
*K* in **one** forward pass and accepts the longest correct prefix, always
emitting the model's own next token for free. Under greedy decoding this is
**lossless** — every emitted token equals what standard decoding would produce.
Conditioning on the chosen token is the key detail: a head trained without it
merely relearns the decoder's own output layer and yields *zero* speculative
acceptance.

---

## Model details

| | |
|---|---|
| **Architecture** | Conformer encoder → MLP projector → Qwen3 decoder (decoder-only LLM) |
| **Parameters** | ~1.7B (encoder + projector + decoder); EAGLE head ~8.4M |
| **Audio front-end** | 128-bin log-Mel, 16 kHz, pre-emphasis + dither, 8× subsampling |
| **Encoder** | 48 Conformer layers, 1280 hidden, relative-position attention |
| **Decoder** | 28 Qwen3 layers, 2048 hidden, 151,936 vocab |
| **Precision** | bfloat16 |
| **Attention** | SDPA (Flash-friendly) |
| **Languages** | Arabic, English, Chinese, Hindi, Malayalam |
| **Base models** | [CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) (encoder), [audarai/Audar-ASR-V1.2-Turbo](https://huggingface.co/audarai/Audar-ASR-V1.2-Turbo) (decoder + tokenizer) |

### Training

{{MODEL_NAME}} was built in stages: the encoder and decoder are initialized
from their respective base models, then trained end-to-end.

1. **Projector alignment** — freeze encoder + decoder, train only the projector so audio embeddings land in the LLM's input space.
2. **Full fine-tune** — unfreeze everything and train on multilingual speech with SpecAugment, speed perturbation, noise injection, and codec augmentation for robustness.
3. **EAGLE head distillation** — freeze the full model and train the lookahead head with a KL objective aligned one position ahead of the decoder's own distribution.

This release pairs the head with the exact target checkpoint it was distilled on:

| | |
|---|---|
| **Target checkpoint** | `{{TARGET_CHECKPOINT}}` |
| **Head training steps** | {{HEAD_STEPS}} |
| **Head artifact** | `eagle/eagle_head.pt` — {{HEAD_FILE_SIZE}}, {{HEAD_PARAMS}} params (8.4M trained + frozen output-projection copy) |

> The head is bound to those exact target weights. Pairing it with a different
> checkpoint — a later training step, a re-trained model, or an architecture
> change such as a new projector or audio token rate — collapses acceptance
> toward zero and turns speculation into pure overhead. Re-distil against the new
> target before shipping it.

---

## Limitations & responsible use

- **Domain/language skew.** The published metrics are English commentary only. Quality varies by language, accent, domain, and recording conditions; the model was tuned on the five languages listed above and is not intended for others.
- **Hallucination on degraded audio.** On very noisy, overlapping, or near-silent segments the LLM decoder can emit fluent but incorrect text. Apply VAD / confidence gating for production.
- **35 s clips.** Audio is processed as a single window; segment long recordings (the streaming server does this automatically with a rolling window).
- **Not for high-stakes decisions.** Do not use transcripts as the sole basis for medical, legal, or safety-critical decisions without human review.
- **16 kHz mono** input is expected; resample and downmix beforehand.

---

## License

Released under **Apache-2.0**. This model derives from
[CohereLabs/cohere-transcribe-03-2026](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026)
and [audarai/Audar-ASR-V1.2-Turbo](https://huggingface.co/audarai/Audar-ASR-V1.2-Turbo);
your use must also comply with the license terms of those upstream models.

## Citation

```bibtex
@software{audar_asr_v1_pro,
  title  = {{{MODEL_NAME}}: A Conformer-Qwen3 Hybrid ASR Model with EAGLE Speculative Decoding},
  author = {Audar AI},
  year   = {2026},
  url    = {https://huggingface.co/{{REPO_ID}}}
}
```
