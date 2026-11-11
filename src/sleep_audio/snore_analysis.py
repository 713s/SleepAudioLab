"""Exploratory, non-clinical analysis of snoring-like acoustic activity.

All detected events are *candidates* produced by an adaptive energy detector;
they are NOT validated snore episodes or breathing-apnea diagnoses.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

from .io import load_audio


def frame_features(samples: np.ndarray, sample_rate: int, *,
                   frame_seconds: float = 0.20, hop_seconds: float = 0.10):
    """Compute frame-wise RMS and spectral descriptors from a mono recording.

    For every frame, spectral centroid and low-frequency power share the same
    windowed periodogram. Flatness compares geometric vs arithmetic mean power.
    """
    if samples.ndim != 1 or sample_rate <= 0:
        raise ValueError("Expected a mono signal and positive sample rate")
    n = max(32, round(frame_seconds * sample_rate))
    hop = max(1, round(hop_seconds * sample_rate))
    if len(samples) < n:
        raise ValueError("Recording is shorter than one analysis frame")
    starts = np.arange(0, len(samples) - n + 1, hop, dtype=np.int64)
    window = np.hanning(n).astype(np.float64)
    frequencies = np.fft.rfftfreq(n, d=1 / sample_rate)
    rms, centroids, low_ratios, flatness = [], [], [], []
    for start in starts:
        frame = samples[start:start + n].astype(np.float64)
        rms.append(float(np.sqrt(np.mean(frame * frame))))
        power = np.abs(np.fft.rfft((frame - frame.mean()) * window)) ** 2
        total = float(power.sum())
        if total <= 1e-20:
            centroids.append(0.0)
            low_ratios.append(0.0)
            flatness.append(0.0)
        else:
            centroids.append(float(np.dot(frequencies, power) / total))
            low_ratios.append(float(power[frequencies <= 1000].sum() / total))
            positive = np.maximum(power, 1e-20)
            flatness.append(float(np.exp(np.mean(np.log(positive))) /
                                  np.mean(positive)))
    return {
        "time_s": (starts + (n / 2)) / sample_rate,
        "rms": np.asarray(rms),
        "spectral_centroid_hz": np.asarray(centroids),
        "power_ratio_below_1000_hz": np.asarray(low_ratios),
        "spectral_flatness": np.asarray(flatness),
        "frame_seconds": frame_seconds,
        "hop_seconds": hop_seconds,
    }


def autocorrelation_periodicity(samples: np.ndarray, sample_rate: int,
                                *, min_hz: float = 50, max_hz: float = 500):
    """Normalized positive-lag autocorrelation peak, descriptive only.

    A local maximum can reflect tonal noise, equipment hum, or snoring.
    """
    x = np.asarray(samples, dtype=np.float64)
    if len(x) < 32:
        return {"peak_correlation": 0.0, "peak_frequency_hz": None}
    x = x - x.mean()
    variance = np.dot(x, x)
    if variance <= 1e-20:
        return {"peak_correlation": 0.0, "peak_frequency_hz": None}
    lo = max(1, int(sample_rate / max_hz))
    hi = min(len(x) - 2, int(sample_rate / min_hz))
    if hi < lo:
        return {"peak_correlation": 0.0, "peak_frequency_hz": None}
    # FFT autocorrelation avoids an O(n^2) direct evaluation.
    length = 1 << (2 * len(x) - 1).bit_length()
    transform = np.fft.rfft(x, n=length)
    autocorr = np.fft.irfft(transform * np.conjugate(transform), n=length)[:hi + 1]
    lag = lo + int(np.argmax(autocorr[lo:hi + 1]))
    coefficient = float(np.clip(autocorr[lag] / variance, -1, 1))
    return {"peak_correlation": coefficient,
            "peak_frequency_hz": float(sample_rate / lag)}


def _intervals(mask: np.ndarray, times: np.ndarray, hop_seconds: float,
               *, gap_seconds: float, min_seconds: float):
    """Join consecutive candidate frames, closing short gaps."""
    true_indices = np.flatnonzero(mask)
    if true_indices.size == 0:
        return []
    max_gap_frames = int(round(gap_seconds / hop_seconds))
    breaks = np.flatnonzero(np.diff(true_indices) > max_gap_frames + 1) + 1
    groups = np.split(true_indices, breaks)
    out = []
    for group in groups:
        first, last = int(group[0]), int(group[-1])
        start = max(0., float(times[first] - hop_seconds / 2))
        end = float(times[last] + hop_seconds / 2)
        if end - start >= min_seconds:
            out.append((start, end, first, last))
    return out


def detect_acoustic_events(features: dict, duration_seconds: float):
    """Adaptive RMS threshold, followed by temporal grouping.

    'quiet' means only low recorded sound, not physiological absence of breath.
    """
    rms = features["rms"]
    times = features["time_s"]
    hop = features["hop_seconds"]
    baseline = float(np.percentile(rms, 20))
    mad = float(np.median(np.abs(rms - np.median(rms))))
    q90 = float(np.percentile(rms, 90))
    threshold = max(baseline + 2.5 * mad, baseline + 0.18 * (q90 - baseline),
                    1e-5)
    active = rms >= threshold
    candidates = _intervals(active, times, hop, gap_seconds=0.35,
                            min_seconds=0.25)
    events = []
    for i, (start, end, first, last) in enumerate(candidates, 1):
        events.append({
            "event_id": i,
            "type": "candidate_acoustic_burst",
            "start_s": round(start, 3),
            "end_s": round(min(end, duration_seconds), 3),
            "duration_s": round(min(end, duration_seconds) - start, 3),
            "mean_rms": round(float(np.mean(rms[first:last + 1])), 6),
            "mean_centroid_hz": round(float(np.mean(
                features["spectral_centroid_hz"][first:last + 1])), 2),
            "mean_low_freq_ratio": round(float(np.mean(
                features["power_ratio_below_1000_hz"][first:last + 1])), 4),
        })
    quiet = _intervals(~active, times, hop, gap_seconds=0.0,
                       min_seconds=3.0)
    quiet_events = [
        {"start_s": round(start, 3),
         "end_s": round(min(end, duration_seconds), 3),
         "duration_s": round(min(end, duration_seconds) - start, 3)}
        for start, end, _, _ in quiet
    ]
    return threshold, events, quiet_events


def _plot_results(samples: np.ndarray, sample_rate: int, features: dict,
                  threshold: float, events: list, destination: Path):
    # Decimate plotted waveform only; retain full sampling in analysis.
    stride = max(1, len(samples) // 15000)
    plot_t = np.arange(0, len(samples), stride) / sample_rate
    plot_x = samples[::stride]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 2]})
    axes[0].plot(plot_t, plot_x, linewidth=0.35)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Recorded waveform")
    axes[1].plot(features["time_s"], features["rms"], linewidth=0.9)
    axes[1].axhline(threshold, color="tab:red", linestyle="--",
                    label="adaptive RMS threshold")
    axes[1].set_ylabel("RMS")
    axes[1].legend(loc="upper right")
    nperseg = min(2048, len(samples))
    overlap = int(0.75 * nperseg)
    f, t, z = stft(samples, fs=sample_rate, nperseg=nperseg,
                   noverlap=overlap, boundary=None, padded=False)
    power = np.abs(z) ** 2
    ref = max(float(power.max()), 1e-20)
    db = 10 * np.log10(np.maximum(power / ref, 1e-12))
    limit = f <= min(2500, sample_rate / 2)
    mesh = axes[2].pcolormesh(t, f[limit], db[limit],
                              shading="auto", vmin=-90, vmax=0)
    fig.colorbar(mesh, ax=axes[2], label="Relative power (dB)")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_title("STFT spectrogram; orange bands = candidate bursts")
    for event in events:
        for ax in axes:
            ax.axvspan(event["start_s"], event["end_s"], alpha=0.11,
                       color="tab:orange")
    fig.tight_layout()
    fig.savefig(destination, dpi=130)
    plt.close(fig)


def analyze_file(source: str | Path, output_dir: str | Path):
    source = Path(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = load_audio(source)
    features = frame_features(audio.samples, audio.sample_rate)
    threshold, events, quiet = detect_acoustic_events(
        features, audio.duration_seconds)
    # Use highest-RMS candidate burst for an exploratory periodicity estimate.
    if events:
        best = max(events, key=lambda event: event["mean_rms"])
        start = int(best["start_s"] * audio.sample_rate)
        stop = int(min(best["end_s"], best["start_s"] + 1)
                   * audio.sample_rate)
        periodicity = autocorrelation_periodicity(
            audio.samples[start:stop], audio.sample_rate)
    else:
        periodicity = {"peak_correlation": 0.0, "peak_frequency_hz": None}
    summary = {
        "source": source.name,
        "sample_rate_hz": audio.sample_rate,
        "duration_seconds": round(audio.duration_seconds, 3),
        "rms": round(audio.rms, 6),
        "adaptive_rms_threshold": round(threshold, 6),
        "candidate_burst_count": len(events),
        "candidate_burst_total_seconds": round(
            sum(e["duration_s"] for e in events), 3),
        "quiet_interval_count_3s": len(quiet),
        "dominant_burst_autocorrelation": periodicity,
        "interpretation": ("Exploratory acoustic candidate detection only. "
                           "Quiet audio is not apnea and burst events are "
                           "not clinically verified snoring."),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    with (output_dir / "events.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "event_id", "type", "start_s", "end_s", "duration_s",
            "mean_rms", "mean_centroid_hz", "mean_low_freq_ratio"])
        writer.writeheader()
        writer.writerows(events)
    with (output_dir / "quiet_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["start_s", "end_s", "duration_s"])
        writer.writeheader()
        writer.writerows(quiet)
    with (output_dir / "frame_features.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "rms", "spectral_centroid_hz",
                         "power_ratio_below_1000_hz", "spectral_flatness"])
        for vals in zip(features["time_s"], features["rms"],
                        features["spectral_centroid_hz"],
                        features["power_ratio_below_1000_hz"],
                        features["spectral_flatness"]):
            writer.writerow([round(float(v), 6) for v in vals])
    _plot_results(audio.samples, audio.sample_rate, features, threshold,
                  events, output_dir / "analysis.png")
    return summary


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/public_snore"))
    a = p.parse_args()
    print(json.dumps(analyze_file(a.audio, a.out), indent=2))
