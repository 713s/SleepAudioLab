import numpy as np
import soundfile as sf

from sleep_audio.io import load_audio


def test_load_audio_mono(tmp_path):
    sample_rate = 8000
    left = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    right = np.linspace(0.5, -0.5, 800, dtype=np.float32)
    stereo = np.column_stack([left, right])

    path = tmp_path / "test.wav"
    sf.write(path, stereo, sample_rate)

    audio = load_audio(path, mono=True)

    assert audio.sample_rate == sample_rate
    assert audio.channels == 2
    assert audio.samples.ndim == 1
    assert audio.samples.shape[0] == 800
    assert np.allclose(audio.samples, 0.0, atol=1e-4)
