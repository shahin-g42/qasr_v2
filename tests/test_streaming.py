import asyncio
import unittest
from unittest.mock import patch

import numpy as np

from qasr.server import _ConnectionState, _transcription_worker
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


class _FakeWebSocket:
    def __init__(self) -> None:
        self.messages = []

    async def send_json(self, message) -> None:
        self.messages.append(message)


class _FakeTranscriber:
    def transcribe(self, waveform, *, language, max_new_tokens) -> str:
        del language, max_new_tokens
        return f"heard {waveform.size}"


class StreamingWorkerTest(unittest.TestCase):
    def test_stop_emits_a_final_result(self) -> None:
        async def call_inline(function, *args, **kwargs):
            return function(*args, **kwargs)

        async def scenario():
            config = StreamingConfig(
                sample_rate=10,
                partial_interval_seconds=0.2,
                min_audio_seconds=0.2,
                window_seconds=1,
                max_new_tokens=4,
            )
            state = _ConnectionState(config=config, language="en")
            state.buffer.append(np.arange(4, dtype="<i2").tobytes())
            state.finished = True
            state.changed.set()
            websocket = _FakeWebSocket()

            with patch("qasr.server._run_in_thread", side_effect=call_inline):
                await _transcription_worker(
                    websocket,
                    state,
                    _FakeTranscriber(),
                    asyncio.Lock(),
                )
            return websocket.messages

        messages = asyncio.run(scenario())
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["type"], "final")
        self.assertEqual(messages[0]["text"], "heard 4")
        self.assertEqual(messages[0]["audio_seconds"], 0.4)

    def test_audio_received_during_inference_is_included_in_final(self) -> None:
        async def scenario():
            config = StreamingConfig(
                sample_rate=10,
                partial_interval_seconds=0.2,
                min_audio_seconds=0.2,
                window_seconds=1,
                max_new_tokens=4,
            )
            state = _ConnectionState(config=config, language=None)
            state.buffer.append(np.arange(2, dtype="<i2").tobytes())
            state.changed.set()
            websocket = _FakeWebSocket()
            invocation = 0

            async def receive_more_audio(function, *args, **kwargs):
                nonlocal invocation
                invocation += 1
                if invocation == 1:
                    state.buffer.append(np.arange(2, 4, dtype="<i2").tobytes())
                    state.finished = True
                    state.changed.set()
                return function(*args, **kwargs)

            with patch(
                "qasr.server._run_in_thread",
                side_effect=receive_more_audio,
            ):
                await _transcription_worker(
                    websocket,
                    state,
                    _FakeTranscriber(),
                    asyncio.Lock(),
                )
            return websocket.messages

        messages = asyncio.run(scenario())
        self.assertEqual(
            [message["type"] for message in messages],
            ["partial", "final"],
        )
        self.assertEqual(
            [message["text"] for message in messages],
            ["heard 2", "heard 4"],
        )


@unittest.skipIf(fastapi is None, "streaming server dependencies are not installed")
class StreamingAppTest(unittest.TestCase):
    def test_app_exposes_page_health_and_websocket_routes(self) -> None:
        from qasr.server import create_app

        config = StreamingConfig(
            sample_rate=10,
            partial_interval_seconds=0.2,
            min_audio_seconds=0.2,
            window_seconds=1,
            max_new_tokens=4,
        )
        transcriber = _FakeTranscriber()
        transcriber.sample_rate = 10

        app = create_app(config=config, transcriber=transcriber)
        paths = {route.path for route in app.routes}
        self.assertTrue({"/", "/health", "/ws/transcribe"}.issubset(paths))


if __name__ == "__main__":
    unittest.main()
