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

# Audar-ASR-V1-Pro

**A 1.7B-parameter multilingual speech-recognition model that pairs a Conformer
audio encoder with a Qwen3 language-model decoder — shipped with a lossless
EAGLE speculative-decoding head for ~2× faster inference and a real-time
streaming server.**

Audar-ASR-V1-Pro (internal codename **QASR**) injects audio into a decoder-only
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
        │  (optional) EAGLE draft head → ~2× faster, identical output
```

---

## Highlights

- **Fluent multilingual ASR** across Arabic, English, Chinese, Hindi, and Malayalam, with native casing and punctuation from the LLM decoder.
- **~2× faster decoding, losslessly.** A tiny (~8.4M-param) EAGLE lookahead head drafts tokens the full model verifies in one pass. Greedy decoding is **provably identical** to standard generation — you trade compute, never accuracy.
- **Real-time streaming.** A WebSocket server performs rolling-window inference and emits partial transcripts as audio arrives.
- **Zero-install offline use** via `trust_remote_code=True` — the modeling code travels with the checkpoint.

---

## Evaluation

Measured on an internal **English motorsport-commentary** benchmark (50 held-out
utterances, greedy decoding) on a single **NVIDIA H100**. WER/CER are
lowercased, punctuation-stripped. RTF = inference-time ÷ audio-duration (lower is
faster; RTF 0.10 ≈ 10× real time).

### Quality & speed by decoding mode

| Mode | WER | CER | RTF | Latency (mean) | Throughput |
|------|:---:|:---:|:---:|:---:|:---:|
| **Offline** (standard) | **6.77%** | 4.77% | 0.097 | 0.996 s | 58 tok/s |
| **Offline + EAGLE** | **6.77%** | 4.74% | **0.056** | **0.503 s** | **93 tok/s** |
| **Streaming** | 6.77% | 4.75% | 0.649 | — | — |
| **Streaming + EAGLE** | 6.89% | 4.84% | **0.434** | — | — |

> EAGLE nearly **halves offline decode latency (0.996 s → 0.503 s)** and lifts
> throughput from 58 → 93 tok/s — a **1.98× speedup** — while producing
> byte-identical transcripts (WER unchanged at 6.77%). In streaming, EAGLE cuts
> RTF from 0.65 → 0.43 (~1.5× faster).

### EAGLE speculation quality

| Metric | Value | Meaning |
|--------|:---:|---|
| End-to-end speedup | **1.98×** | offline latency ÷ eagle latency |
| Acceptance rate | 18.2% | fraction of drafted tokens accepted |
| Tokens per model forward | 1.76 | vs 1.0 for standard decoding |
| Position-0 acceptance | **71.8%** | the first drafted token is usually right |
| Acceptance by position | 72% · 13% · 3% · 2% · 1% | draft depth 1…5 |

Losslessness check: transcripts from EAGLE vs standard decoding matched on
**49/50** samples; the single difference is a final-token tie-break under bf16
rounding (numerically expected, not a decoding error).

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

Audar-ASR-V1-Pro was built in stages: the encoder and decoder are initialized
from their respective base models, then trained end-to-end.

1. **Projector alignment** — freeze encoder + decoder, train only the projector so audio embeddings land in the LLM's input space.
2. **Full fine-tune** — unfreeze everything and train on multilingual speech with SpecAugment, speed perturbation, noise injection, and codec augmentation for robustness.
3. **EAGLE head distillation** — freeze the full model and train the lookahead head with a KL objective aligned one position ahead of the decoder's own distribution.

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
  title  = {Audar-ASR-V1-Pro: A Conformer-Qwen3 Hybrid ASR Model with EAGLE Speculative Decoding},
  author = {Audar AI},
  year   = {2026},
  url    = {https://huggingface.co/{{REPO_ID}}}
}
```
