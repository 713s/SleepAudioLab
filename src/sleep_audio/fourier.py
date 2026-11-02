"""Fourier-analysis utilities used for learning and validation.

The direct DFT is intentionally implemented from the definition. It is not
intended for long recordings; use NumPy FFT routines for production-scale
analysis.
"""

from __future__ import annotations

import numpy as np


def dft(x: np.ndarray) -> np.ndarray:
    """Compute the discrete Fourier transform directly from its definition.

    Complexity is O(N^2). This implementation is for small educational
    examples and numerical comparison with ``numpy.fft.fft``.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("dft expects a one-dimensional signal")

    n_samples = x.size
    if n_samples == 0:
        return np.empty(0, dtype=np.complex128)

    n = np.arange(n_samples)
    k = n.reshape(-1, 1)
    kernel = np.exp(-2j * np.pi * k * n / n_samples)
    return kernel @ x


def idft(X: np.ndarray) -> np.ndarray:
    """Compute the inverse DFT directly from its definition."""
    X = np.asarray(X, dtype=np.complex128)
    if X.ndim != 1:
        raise ValueError("idft expects a one-dimensional spectrum")

    n_samples = X.size
    if n_samples == 0:
        return np.empty(0, dtype=np.complex128)

    k = np.arange(n_samples)
    n = k.reshape(-1, 1)
    kernel = np.exp(2j * np.pi * n * k / n_samples)
    return (kernel @ X) / n_samples


def frequency_bins(n_samples: int, sample_rate: float) -> np.ndarray:
    """Return the ordinary FFT frequency bins in Hz."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return np.fft.fftfreq(n_samples, d=1.0 / sample_rate)


def rfft_spectrum(x: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """Return nonnegative frequency bins and the real-input FFT spectrum."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("rfft_spectrum expects a one-dimensional signal")
    if x.size == 0:
        raise ValueError("signal must not be empty")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frequencies = np.fft.rfftfreq(x.size, d=1.0 / sample_rate)
    spectrum = np.fft.rfft(x)
    return frequencies, spectrum


def parseval_energies(x: np.ndarray) -> tuple[float, float]:
    """Return time-domain and frequency-domain energies.

    With NumPy's FFT normalization convention, Parseval gives

        sum |x[n]|^2 = (1/N) sum |X[k]|^2.
    """
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1:
        raise ValueError("parseval_energies expects a one-dimensional signal")
    if x.size == 0:
        raise ValueError("signal must not be empty")

    spectrum = np.fft.fft(x)
    time_energy = float(np.sum(np.abs(x) ** 2))
    frequency_energy = float(np.sum(np.abs(spectrum) ** 2) / x.size)
    return time_energy, frequency_energy


def alias_frequency(frequency: float, sample_rate: float) -> float:
    """Map a sinusoidal frequency to its folded discrete-time alias in [0, fs/2]."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    wrapped = frequency % sample_rate
    if wrapped > sample_rate / 2:
        wrapped = sample_rate - wrapped
    return float(abs(wrapped))
