import numpy as np

from sleep_audio.fourier import alias_frequency, dft, idft, parseval_energies


def test_direct_dft_matches_numpy_fft():
    rng = np.random.default_rng(1)
    x = rng.normal(size=16)
    assert np.allclose(dft(x), np.fft.fft(x), atol=1e-10)


def test_idft_recovers_signal():
    rng = np.random.default_rng(2)
    x = rng.normal(size=12)
    reconstructed = idft(dft(x))
    assert np.allclose(reconstructed.real, x, atol=1e-10)
    assert np.allclose(reconstructed.imag, 0.0, atol=1e-10)


def test_parseval_identity():
    rng = np.random.default_rng(3)
    x = rng.normal(size=64)
    time_energy, frequency_energy = parseval_energies(x)
    assert np.isclose(time_energy, frequency_energy, rtol=1e-12, atol=1e-12)


def test_alias_frequency():
    assert alias_frequency(700.0, 1000.0) == 300.0
    assert alias_frequency(1300.0, 1000.0) == 300.0
