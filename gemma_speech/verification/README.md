# Verification record

Executed locally on 2026-10-14. CPU, macOS ARM64, Python 3.13.7. The clean temporary environment was installed normally from `gemma_speech[dev]`, with **PyTorch 2.11.0 and Transformers 5.19.0**. It used no `PYTHONPATH` overrides, no `--no-deps` installation, and no separately installed Qwen SDK. `pip check` reported no broken requirements. [Resolved environment](requirements-cpu-lock.txt).

| Check | Result | Evidence / scope |
|---|---|---|
| Package installation and dependency resolution | Passed | Clean venv; pinned torch/Transformers; no dependency conflict |
| Unit and integration tests | 30 passed, 1 skipped | [pytest output](pytest.txt); the skip is optional torchaudio/Vocos mel compatibility |
| Pretrained Gemma → Qwen ASR | Passed | Actual pinned weights; finite loss and bridge gradients; token generation |
| Pretrained Gemma → speaker adapter → Qwen TTS | Passed | Actual audio tower, speaker target, codec encoder/decoder and TTS talker in one process; one synthetic adapter update; finite waveform |
| TTS training/inference conditioning | Passed | Training prefix compared numerically to the retained Qwen generation prefix |
| Offline pretrained execution | Passed | `HF_HUB_OFFLINE=1` for the unified probe |
| Offline ASR checkpoint reload | Passed | Tests removed upstream model directories and reloaded the saved hybrid |
| Training/resume | Passed | Exact deterministic continuation tests for ASR, flow TTS and Qwen talker |
| CPU distributed ASR | Passed | Two torchrun workers, Gloo, two optimizer steps, validation and rank-zero checkpoint |
| Lint / shell syntax | Passed | Ruff; `bash -n` on both launch scripts |
| Wheel build | Passed | Includes the private Qwen implementation, Apache license, change notice and original source hashes |

[Pretrained integration values](pretrained-integration.json) and [distributed smoke result](distributed-smoke.json) are saved here. The earlier [baseline tensor probes](baseline-pretrained-probe.json) used actual Gemma weights with newly initialized CTC/flow heads under PyTorch 2.13; they are supplementary and are **not** evidence for a different deployment environment. The final unified probe uses 2.11 throughout.

The pretrained probe uses a sine wave and a newly initialized adapter. Its transcript tokens, loss, generated audio duration and waveform finiteness demonstrate execution, not quality. It does not establish intelligibility, speaker identity or Arabic support. Tiny training tests likewise use synthetic data and random small instances of the real model classes.

Not executed: real-corpus ASR/TTS training, WER/MOS/voice-similarity benchmarks, Linux CUDA/BF16 validation, H100 throughput, multi-node NCCL, 96-GPU scaling, streaming inference, quantization/export, or the optional Vocos waveform path. These are the next experiments on the supplied cluster/data, not completed results.

The initial CPU distributed attempt failed because of local socket restrictions and an unresolved macOS hostname. The successful run used an explicitly permitted localhost socket, `127.0.0.1`, and the `lo0` Gloo interface. That setting is a local test detail; the H100 cluster must use its real inter-node interface.
