import unittest

import numpy as np

from qasr.server import SessionState, TranscriptionServer
from qasr.streaming import PCM16Buffer, StreamingConfig, merge_windowed_transcript

try:
    import fastapi
except ImportError:
    fastapi = None


class PCM16BufferTest(unittest.TestCase):
    def test_decodes_little_endian_pcm_and_normalizes(self) -> None:
        pcm = np.array([-32768, -16384, 0, 16384, 32767], dtype="<i2")
        buffer = PCM16Buffer(max_samples=10)

        buffer.append(pcm.tobytes())

        np.testing.assert_allclose(
            buffer.waveform(),
            [-1.0, -0.5, 0.0, 0.5, 32767 / 32768],
        )
        self.assertEqual(buffer.total_samples, 5)
        self.assertFalse(buffer.is_windowed)

    def test_keeps_only_the_latest_bounded_window(self) -> None:
        buffer = PCM16Buffer(max_samples=4)
        buffer.append(np.arange(3, dtype="<i2").tobytes())
        buffer.append(np.arange(3, 7, dtype="<i2").tobytes())

        expected = np.arange(3, 7, dtype=np.float32) / 32768
        np.testing.assert_allclose(buffer.waveform(), expected)
        self.assertEqual(buffer.total_samples, 7)
        self.assertEqual(buffer.buffered_samples, 4)
        self.assertTrue(buffer.is_windowed)

    def test_rejects_partial_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole number of samples"):
            PCM16Buffer(max_samples=4).append(b"\x00")


class TranscriptMergeTest(unittest.TestCase):
    def test_merges_word_overlap_and_uses_new_hypothesis(self) -> None:
        merged = merge_windowed_transcript(
            "one two three four",
            "two three four five",
        )
        self.assertEqual(merged, "one two three four five")

    def test_merges_unsegmented_text(self) -> None:
        merged = merge_windowed_transcript("你好世界今天", "世界今天很好")
        self.assertEqual(merged, "你好世界今天很好")

    def test_replaces_when_no_reliable_overlap_exists(self) -> None:
        self.assertEqual(merge_windowed_transcript("old words", "new result"), "new result")


class StreamingConfigTest(unittest.TestCase):
    def test_converts_durations_to_samples(self) -> None:
        config = StreamingConfig(
            sample_rate=100,
            partial_interval_seconds=0.25,
            min_audio_seconds=0.5,
            window_seconds=2,
        )
        self.assertEqual(config.partial_interval_samples, 25)
        self.assertEqual(config.min_audio_samples, 50)
        self.assertEqual(config.window_samples, 200)

    def test_rejects_a_window_shorter_than_minimum_audio(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least min_audio_seconds"):
            StreamingConfig(min_audio_seconds=2, window_seconds=1)


class _FakeTranscriber:
    def transcribe(self, waveform, *, language, max_new_tokens) -> str:
        del language, max_new_tokens
        return f"heard {waveform.size}"


class TranscriptionServerTest(unittest.TestCase):
    def test_transcribe_window_delegates_to_the_plain_transcriber(self) -> None:
        server = TranscriptionServer(model_path="unused")
        server._transcriber = _FakeTranscriber()
        server.max_new_tokens = 4

        result = server.transcribe_window(np.zeros(800, dtype=np.float32), language="en")

        self.assertEqual(result, "heard 800")

    def test_session_state_tracks_a_bounded_pcm_window(self) -> None:
        session = SessionState(
            session_id="s1",
            pcm_buffer=PCM16Buffer(max_samples=4),
            language="ar",
        )
        session.pcm_buffer.append(np.arange(6, dtype="<i2").tobytes())

        self.assertEqual(session.pcm_buffer.total_samples, 6)
        self.assertEqual(session.pcm_buffer.buffered_samples, 4)
        self.assertEqual(session.language, "ar")
        self.assertEqual(session.merged_transcript, "")


@unittest.skipIf(fastapi is None, "streaming server dependencies are not installed")
class StreamingAppTest(unittest.TestCase):
    def test_app_exposes_health_and_websocket_routes(self) -> None:
        from qasr.server import create_app

        app = create_app(TranscriptionServer(model_path="unused"))
        paths = {route.path for route in app.routes}
        self.assertTrue({"/health", "/ws/transcribe"}.issubset(paths))

    def test_websocket_protocol_matches_the_browser_client(self) -> None:
        from fastapi.testclient import TestClient

        from qasr.server import create_app

        server = TranscriptionServer(model_path="unused")
        server.sample_rate = 16000
        server._transcriber = _FakeTranscriber()
        server.load = lambda: None  # bypass model loading in the lifespan

        app = create_app(server)
        pcm = np.zeros(8000, dtype="<i2").tobytes()  # 0.5s at 16 kHz

        with TestClient(app) as client, client.websocket_connect("/ws/transcribe") as ws:
            ready = ws.receive_json()
            self.assertEqual(ready["type"], "ready")
            self.assertEqual(ready["sample_rate"], 16000)

            ws.send_json({"type": "start", "language": "ar"})
            ws.send_bytes(pcm)

            partial = ws.receive_json()
            self.assertEqual(partial["type"], "partial")
            self.assertEqual(partial["text"], "heard 8000")
            self.assertIn("latency_ms", partial)

            ws.send_json({"type": "stop"})
            final = ws.receive_json()
            self.assertEqual(final["type"], "final")
            self.assertEqual(final["text"], "heard 8000")


if __name__ == "__main__":
    unittest.main()
