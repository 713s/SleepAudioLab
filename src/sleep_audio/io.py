from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def peak_absolute_amplitude(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.max(np.abs(self.samples)))

    @property
    def rms(self) -> float:
        if self.samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(self.samples))))


def load_audio(path: str | Path, *, mono: bool = True) -> AudioData:
    """Load an audio file as floating-point samples.

    Parameters
    ----------
    path:
        Path understood by libsndfile (for example WAV or FLAC).
    mono:
        If True, average channels to produce a one-dimensional baseline signal.
    """
    path = Path(path)
    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    channels = data.shape[1]

    if mono:
        samples = data.mean(axis=1)
    else:
        samples = data

    return AudioData(
        samples=np.asarray(samples),
        sample_rate=int(sample_rate),
        channels=int(channels),
    )
