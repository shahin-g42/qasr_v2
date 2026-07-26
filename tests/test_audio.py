import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile as sf

    from qasr.audio import load_mono_audio
except ImportError:
    np = None


@unittest.skipIf(np is None, "audio dependencies are not installed")
class AudioLoadingTest(unittest.TestCase):
    def test_stereo_audio_is_mixed_and_resampled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            source_rate = 8_000
            samples = np.linspace(-0.5, 0.5, source_rate, dtype=np.float32)
            stereo = np.stack([samples, samples], axis=1)
            sf.write(path, stereo, source_rate)

            waveform = load_mono_audio(path, target_sampling_rate=16_000)

            self.assertEqual(waveform.ndim, 1)
            self.assertEqual(waveform.dtype, np.float32)
            self.assertLessEqual(abs(len(waveform) - 16_000), 1)


if __name__ == "__main__":
    unittest.main()

