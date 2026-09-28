"""WebSocket streaming server for QASR real-time transcription.

Clients connect to /ws/transcribe and send binary PCM16 audio chunks at 16 kHz,
bracketed by JSON control messages ({"type": "start"} / {"type": "stop"}).
The server accumulates audio, runs rolling-window inference, and streams partial
transcriptions back as JSON messages, finishing with a "final" message.

Serves the bundled browser demo UI from src/qasr/static at "/".

Supports optional EAGLE-2 speculative decoding for 2-3x faster partial transcripts.

Usage:
    qasr-stream --model /path/to/qasr/checkpoint --port 8765
    qasr-stream --model /path/to/qasr/checkpoint --eagle /path/to/eagle --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .processing import resolve_qasr_language
from .streaming import PCM16Buffer, QASRTranscriber, merge_windowed_transcript

# FastAPI is an optional dependency, but WebSocket/WebSocketDisconnect must be
# module-globals: FastAPI resolves the endpoint's parameter annotations against
# the module's globals, and a function-local import makes it misread the
# ``websocket`` parameter as a query field (rejecting every handshake).
try:
    from fastapi import WebSocket, WebSocketDisconnect
except ImportError:
    WebSocket = None  # type: ignore[assignment,misc]
    WebSocketDisconnect = None  # type: ignore[assignment,misc]

LOGGER = logging.getLogger("qasr.server")


@dataclass
class SessionState:
    """Per-connection transcription state."""

    session_id: str
    pcm_buffer: PCM16Buffer
    language: str = "ar"
    last_partial: str = ""
    last_partial_time: float = 0.0
    merged_transcript: str = ""
    total_audio_seconds: float = 0.0
    total_inference_time: float = 0.0
    inference_count: int = 0
    # Cumulative sample count at the last inference — used to throttle
    # inference to at most once per ``step_seconds`` of new audio.
    last_infer_samples: int = 0
    # A decode failure is reported to the client once per session, not once
    # per window, to avoid flooding the UI with identical error frames.
    decode_error_sent: bool = False
    created_at: float = field(default_factory=time.time)


class TranscriptionServer:
    """Manages the QASR model and WebSocket transcription sessions."""

    def __init__(
        self,
        model_path: str,
        eagle_path: str | None = None,
        num_draft_tokens: int = 5,
        device: str = "auto",
        dtype: torch.dtype = torch.bfloat16,
        window_seconds: float = 20.0,
        step_seconds: float = 10.0,
        min_audio_seconds: float = 0.5,
        max_audio_seconds: float = 30.0,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_path = model_path
        self.eagle_path = eagle_path
        self.num_draft_tokens = num_draft_tokens
        self.window_seconds = window_seconds
        self.step_seconds = step_seconds
        self.min_audio_seconds = min_audio_seconds
        self.max_audio_seconds = max_audio_seconds
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.dtype = dtype
        self._transcriber: QASRTranscriber | None = None
        self._eagle_decoder = None

    def load(self) -> None:
        """Load model and processor (call once at startup)."""
        from .modeling import QASRForConditionalGeneration
        from .processing import QASRProcessor

        LOGGER.info("Loading QASR model from %s", self.model_path)
        self.processor = QASRProcessor.from_pretrained(self.model_path)
        model_kwargs: dict[str, Any] = {
            "dtype": self.dtype,
            "attn_implementation": "sdpa",
        }
        if self.device == "auto":
            model_kwargs["device_map"] = "auto"
        self.model = QASRForConditionalGeneration.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if self.device != "auto":
            self.model.to(self.device)
        self.model.eval()
        self.sample_rate = int(self.processor.feature_extractor.sampling_rate)
        self._transcriber = QASRTranscriber(processor=self.processor, model=self.model)

        # Load EAGLE head if provided, sharing the already loaded target model
        # instead of paying for a second 8 GB copy.
        if self.eagle_path:
            from .eagle import EagleHead, EagleSpeculativeDecoder
            LOGGER.info("Loading EAGLE head from %s (speculative decoding enabled)", self.eagle_path)
            eagle_head = EagleHead.from_pretrained(self.eagle_path)
            self._eagle_decoder = EagleSpeculativeDecoder(model=self.model, eagle_head=eagle_head)
            LOGGER.info("EAGLE speculative decoder ready (K=%d)", self.num_draft_tokens)

        LOGGER.info("Model loaded successfully")

    def transcribe_window(self, audio: np.ndarray, language: str) -> str:
        """Run inference on an audio window.

        Blocking; callers on an event loop must dispatch via asyncio.to_thread.
        """
        if self._eagle_decoder is not None:
            # The EAGLE decoder drives the same target model the transcriber
            # wraps, so both branches serialize on the transcriber's lock.
            with self._transcriber.lock:
                return self._transcribe_eagle(audio, language)
        return self._transcriber.transcribe(audio, language=language, max_new_tokens=self.max_new_tokens)

    def _transcribe_eagle(self, audio: np.ndarray, language: str) -> str:
        """Transcribe using EAGLE speculative decoding."""
        inputs = self.processor.apply_transcription_request(
            audio=audio,
            language=language,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
        )
        inputs = inputs.to(self._eagle_decoder.device, self._eagle_decoder.dtype)
        output_ids = self._eagle_decoder.generate(
            input_ids=inputs["input_ids"],
            input_features=inputs["input_features"],
            input_features_mask=inputs["input_features_mask"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=self.max_new_tokens,
            num_draft_tokens=self.num_draft_tokens,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated_ids[0], return_format="transcription_only")


def create_app(server: TranscriptionServer):
    """Create the FastAPI application with WebSocket endpoint and demo UI.

    Wire protocol (matches src/qasr/static/app.js):
      server -> client on connect : {"type": "ready", "sample_rate": int}
      client -> server            : binary PCM16 audio chunks
      client -> server            : {"type": "start", "language": str | null}
      client -> server            : {"type": "stop"}
      server -> client            : {"type": "partial", "text", "audio_seconds",
                                     "latency_ms", "session_id"}
      server -> client            : {"type": "final", ...} after a stop request
      server -> client            : {"type": "error", "message": str} on a bad
                                     frame, bad language code, or decode failure
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse
        from fastapi.staticfiles import StaticFiles
    except ImportError as exc:
        raise ImportError(
            "FastAPI is required for the streaming server. "
            "Install with: pip install 'qasr[streaming]'"
        ) from exc

    sessions: dict[str, SessionState] = {}

    @asynccontextmanager
    async def lifespan(_: Any):
        server.load()
        yield

    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="QASR Streaming ASR", version="0.2.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": server.model_path, "eagle": server.eagle_path is not None}

    @app.websocket("/ws/transcribe")
    async def transcribe_endpoint(websocket: WebSocket):
        await websocket.accept()
        session_id = f"session-{int(time.time() * 1000)}"
        session = SessionState(
            session_id=session_id,
            pcm_buffer=PCM16Buffer(
                max_samples=int(server.window_seconds * server.sample_rate)
            ),
        )
        sessions[session_id] = session
        LOGGER.info("Client connected: %s", session_id)
        await websocket.send_json({"type": "ready", "sample_rate": server.sample_rate})

        # Inference cadence: at most one pass per step_seconds of new audio
        step_samples = max(1, int(server.step_seconds * server.sample_rate))

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(message.get("code", 1000))

                raw = message.get("bytes")
                if raw is not None:
                    try:
                        session.pcm_buffer.append(raw)
                    except ValueError as e:
                        # A malformed frame is dropped; the session stays alive.
                        LOGGER.warning("Dropping bad audio frame from %s: %s", session_id, e)
                        await websocket.send_json({"type": "error", "message": str(e)})
                        continue
                    available = session.pcm_buffer.total_samples / server.sample_rate
                    session.total_audio_seconds = available

                    # Check if we have enough audio for a window
                    if available < server.min_audio_seconds:
                        continue

                    # Throttle: wait for a full step of new audio before the
                    # next inference pass (the first pass runs immediately).
                    if (
                        session.inference_count > 0
                        and session.pcm_buffer.total_samples - session.last_infer_samples
                        < step_samples
                    ):
                        continue

                    # The bounded buffer already holds only the newest window
                    audio = session.pcm_buffer.waveform()
                    if audio.size < 800:  # < 50ms
                        continue

                    # Run inference off the event loop so a 0.5-1.0s decode
                    # does not stall every other session and HTTP route.
                    session.last_infer_samples = session.pcm_buffer.total_samples
                    t0 = time.time()
                    try:
                        partial = await asyncio.to_thread(
                            server.transcribe_window, audio, language=session.language
                        )
                    except Exception as e:
                        LOGGER.error("Inference error: %s", e)
                        if not session.decode_error_sent:
                            session.decode_error_sent = True
                            await websocket.send_json(
                                {"type": "error", "message": f"Transcription failed: {e}"}
                            )
                        continue
                    inference_time = time.time() - t0
                    session.total_inference_time += inference_time
                    session.inference_count += 1

                    # Send partial result, merging across the rolling window so
                    # text that fell out of the audio buffer is not lost.
                    if partial and partial != session.last_partial:
                        session.last_partial = partial
                        session.last_partial_time = time.time()
                        session.merged_transcript = merge_windowed_transcript(
                            session.merged_transcript, partial
                        )
                        await websocket.send_json({
                            "type": "partial",
                            "text": session.merged_transcript,
                            "audio_seconds": round(available, 2),
                            "latency_ms": round(inference_time * 1000, 1),
                            "session_id": session_id,
                        })
                    continue

                text = message.get("text")
                if not text:
                    continue
                try:
                    control = json.loads(text)
                except json.JSONDecodeError:
                    LOGGER.warning("Ignoring non-JSON control message from %s", session_id)
                    continue

                kind = control.get("type")
                if kind == "start":
                    language = control.get("language")
                    if language:
                        # Reject unknown codes here instead of letting every
                        # decode fail silently inside the inference try/except.
                        try:
                            resolve_qasr_language(language)
                        except ValueError:
                            await websocket.send_json({
                                "type": "error",
                                "message": (
                                    f"Unsupported language {language!r}; "
                                    f"keeping {session.language!r}"
                                ),
                            })
                        else:
                            session.language = language
                elif kind == "stop":
                    # One last pass over the buffered window, then finalize.
                    latency_ms = 0.0
                    audio = session.pcm_buffer.waveform()
                    if audio.size >= 800:
                        t0 = time.time()
                        try:
                            partial = await asyncio.to_thread(
                                server.transcribe_window, audio, language=session.language
                            )
                        except Exception as e:
                            LOGGER.error("Final inference error: %s", e)
                            if not session.decode_error_sent:
                                session.decode_error_sent = True
                                await websocket.send_json(
                                    {"type": "error", "message": f"Transcription failed: {e}"}
                                )
                        else:
                            latency_ms = (time.time() - t0) * 1000
                            session.total_inference_time += latency_ms / 1000
                            session.inference_count += 1
                            if partial:
                                session.last_partial = partial
                                session.merged_transcript = merge_windowed_transcript(
                                    session.merged_transcript, partial
                                )
                    await websocket.send_json({
                        "type": "final",
                        "text": session.merged_transcript,
                        "audio_seconds": round(
                            session.pcm_buffer.total_samples / server.sample_rate, 2
                        ),
                        "latency_ms": round(latency_ms, 1),
                        "session_id": session_id,
                    })
                    break

        except WebSocketDisconnect:
            LOGGER.info(
                "Client disconnected: %s (%.1fs audio, %d inferences, avg %.0fms)",
                session_id,
                session.total_audio_seconds,
                session.inference_count,
                (session.total_inference_time / max(session.inference_count, 1)) * 1000,
            )
        finally:
            sessions.pop(session_id, None)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="QASR streaming transcription server")
    parser.add_argument("--model", required=True, help="Path to QASR checkpoint")
    parser.add_argument("--eagle", default=None, help="Path to EAGLE head for speculative decoding")
    parser.add_argument("--num-draft-tokens", type=int, default=5)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--window-seconds", type=float, default=20.0)
    parser.add_argument("--step-seconds", type=float, default=10.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    server = TranscriptionServer(
        model_path=args.model,
        eagle_path=args.eagle,
        num_draft_tokens=args.num_draft_tokens,
        device=args.device,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
    )

    app = create_app(server)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
