"""Generate small Fourier-analysis experiments for v0.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fourier import alias_frequency, parseval_energies, rfft_spectrum


def _plot_spectrum(signal: np.ndarray, sample_rate: float, title: str, output: Path) -> None:
    frequencies, spectrum = rfft_spectrum(signal, sample_rate)
    magnitude = np.abs(spectrum)

    plt.figure(figsize=(9, 4.5))
    plt.plot(frequencies, magnitude)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def run_demo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bin-aligned sinusoid.
    fs = 1024.0
    n_samples = 1024
    t = np.arange(n_samples) / fs
    aligned_frequency = 128.0
    aligned = np.sin(2 * np.pi * aligned_frequency * t)
    _plot_spectrum(
        aligned,
        fs,
        "Bin-aligned sinusoid: 128 Hz at 1024 Hz sampling",
        output_dir / "bin_aligned_sine.png",
    )

    # 2. Spectral leakage from a non-bin-aligned sinusoid.
    leaking_frequency = 130.5
    leaking = np.sin(2 * np.pi * leaking_frequency * t)
    _plot_spectrum(
        leaking,
        fs,
        "Spectral leakage: 130.5 Hz does not align with a DFT bin",
        output_dir / "spectral_leakage.png",
    )

    # 3. Aliasing example.
    alias_fs = 1000.0
    alias_n = 1000
    alias_t = np.arange(alias_n) / alias_fs
    high_frequency = 700.0
    expected_alias = alias_frequency(high_frequency, alias_fs)
    high_signal = np.sin(2 * np.pi * high_frequency * alias_t)
    alias_signal = -np.sin(2 * np.pi * expected_alias * alias_t)

    plt.figure(figsize=(9, 4.5))
    plt.plot(alias_t[:25], high_signal[:25], marker="o", label=f"{high_frequency:.0f} Hz samples")
    plt.plot(alias_t[:25], alias_signal[:25], linestyle="--", label=f"{expected_alias:.0f} Hz alias")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Aliasing after sampling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "aliasing.png", dpi=150)
    plt.close()

    # 4. Parseval identity.
    rng = np.random.default_rng(0)
    test_signal = rng.normal(size=256)
    time_energy, frequency_energy = parseval_energies(test_signal)
    payload = {
        "time_domain_energy": time_energy,
        "frequency_domain_energy": frequency_energy,
        "absolute_difference": abs(time_energy - frequency_energy),
    }
    (output_dir / "parseval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Fourier-analysis demonstrations.")
    parser.add_argument("--out", type=Path, default=Path("reports/fourier_demo"))
    args = parser.parse_args()
    run_demo(args.out)


if __name__ == "__main__":
    main()
