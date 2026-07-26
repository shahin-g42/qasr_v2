import unittest
from unittest.mock import patch

import numpy as np

from qasr.data import ResilientAudioDataset


class _Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, token_ids, **kwargs):
        return " ".join(f"token-{token_id}" for token_id in token_ids)


class TranscriptLengthTest(unittest.TestCase):
    def test_overlong_transcript_is_replaced_and_cached(self) -> None:
        dataset = ResilientAudioDataset(
            [
                {"audio_path": "long.wav", "text": "one two three", "line_number": 1},
                {"audio_path": "good.wav", "text": "one two", "line_number": 2},
            ],
            sampling_rate=1_000,
            min_audio_seconds=0.1,
            max_audio_seconds=1.0,
            tokenizer=_Tokenizer(),
            max_target_length=2,
            truncate_long_transcripts=False,
        )
        with patch(
            "qasr.data.load_mono_audio",
            return_value=np.zeros(200, dtype=np.float32),
        ) as load_audio:
            first = dataset[0]
            second = dataset[0]

        self.assertEqual(first["audio_path"], "good.wav")
        self.assertEqual(second["audio_path"], "good.wav")
        self.assertEqual(load_audio.call_count, 2)

    def test_explicit_truncation_updates_text_and_ids(self) -> None:
        dataset = ResilientAudioDataset(
            [{"audio_path": "long.wav", "text": "one two three", "line_number": 1}],
            sampling_rate=1_000,
            min_audio_seconds=0.1,
            max_audio_seconds=1.0,
            tokenizer=_Tokenizer(),
            max_target_length=2,
            truncate_long_transcripts=True,
        )
        with patch(
            "qasr.data.load_mono_audio",
            return_value=np.zeros(200, dtype=np.float32),
        ):
            sample = dataset[0]

        self.assertEqual(sample["transcript_ids"], [0, 1])
        self.assertEqual(sample["text"], "token-0 token-1")


if __name__ == "__main__":
    unittest.main()
