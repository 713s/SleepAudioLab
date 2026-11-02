from __future__ import annotations

import numpy as np
from scipy.signal import stft


def compute_stft(
    samples: np.ndarray,
    sample_rate: int,
    *,
    nperseg: int = 2048,
    noverlap: int = 1536,
):
    """Compute a Hann-window STFT for a mono signal."""
    frequencies, times, zxx = stft(
        samples,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )
    return frequencies, times, zxx


def power_db(zxx: np.ndarray, *, floor_db: float = -120.0) -> np.ndarray:
    """Convert complex STFT coefficients to relative power in decibels."""
    power = np.abs(zxx) ** 2
    reference = float(np.max(power)) if power.size else 0.0
    if reference <= 0.0:
        return np.full_like(power, floor_db, dtype=float)

    db = 10.0 * np.log10(np.maximum(power / reference, 10 ** (floor_db / 10.0)))
    return db
