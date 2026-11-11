from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .io import load_audio
from .spectrum import compute_stft, power_db


def _plot_waveform(samples: np.ndarray, sample_rate: int, out_path: Path) -> None:
    time = np.arange(samples.size) / sample_rate
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, samples, linewidth=0.6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_spectrogram(samples: np.ndarray, sample_rate: int, out_path: Path) -> None:
    frequencies, times, zxx = compute_stft(samples, sample_rate)
    db = power_db(zxx)

    fig, ax = plt.subplots(figsize=(12, 5))
    mesh = ax.pcolormesh(times, frequencies, db, shading="auto")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("STFT spectrogram (relative power, dB)")
    fig.colorbar(mesh, ax=ax, label="Relative power (dB)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def inspect_audio(path: Path, out_dir: Path) -> None:
    audio = load_audio(path, mono=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "source": str(path),
        "sample_rate_hz": audio.sample_rate,
        "channels_in_source": audio.channels,
        "samples_after_mono_conversion": int(audio.samples.size),
        "duration_seconds": audio.duration_seconds,
        "peak_absolute_amplitude": audio.peak_absolute_amplitude,
        "rms": audio.rms,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _plot_waveform(audio.samples, audio.sample_rate, out_dir / "waveform.png")
    _plot_spectrogram(audio.samples, audio.sample_rate, out_dir / "spectrogram.png")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote results to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sleep-audio")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Inspect an audio file and generate baseline plots"
    )
    inspect_parser.add_argument("audio", type=Path)
    inspect_parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/inspect"),
        help="Output directory",
    )

    night = subparsers.add_parser(
        "overnight", help="Stream an overnight recording into acoustic event and hourly reports"
    )
    night.add_argument("audio", type=Path)
    night.add_argument("--out", type=Path, default=Path("reports/overnight"))
    night.add_argument("--quiet-min", type=float, default=10.)
    night.add_argument("--event-min", type=float, default=2.)
    night.add_argument("--merge-gap", type=float, default=1.)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "inspect":
        inspect_audio(args.audio, args.out)
    elif args.command == "overnight":
        from .overnight import analyze_overnight
        result = analyze_overnight(
            args.audio, args.out, quiet_min_seconds=args.quiet_min,
            event_min_seconds=args.event_min, merge_gap_seconds=args.merge_gap)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
