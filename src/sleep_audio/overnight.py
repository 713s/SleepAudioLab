"""Stream an entire night into interpretable acoustic reports, not diagnoses.

One-second windows, bounded audio memory, global robust RMS threshold,
candidate activity and recorded quiet intervals. No neural network required.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FEATURES = ("start_s", "end_s", "rms", "peak", "spectral_centroid_hz",
            "low_power_ratio", "spectral_flatness")
EVENT_FIELDS = ("event_id", "type", "start_s", "end_s", "duration_s",
                "mean_rms", "mean_centroid_hz", "mean_low_power_ratio")
QUIET_FIELDS = ("start_s", "end_s", "duration_s")
HOUR_FIELDS = ("hour_index", "start_s", "end_s", "candidate_activity_seconds",
               "quiet_seconds", "candidate_event_count", "mean_rms")
ANALYSIS_RATE = 8000


def _one_second_features(x: np.ndarray, rate: int):
    """Energy in original recording; spectral descriptors after proper resampling."""
    x = np.asarray(x, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x*x)))
    peak = float(np.max(np.abs(x)))
    gcd = math.gcd(rate, ANALYSIS_RATE)
    reduced = resample_poly(x, ANALYSIS_RATE // gcd, rate // gcd)
    # Spectral power from a Hann-windowed demeaned frame.
    spectrum = np.abs(np.fft.rfft((reduced - reduced.mean()) *
                                 np.hanning(len(reduced)))) ** 2
    freq = np.fft.rfftfreq(len(reduced), 1 / ANALYSIS_RATE)
    power = float(spectrum.sum())
    if power < 1e-20:
        return rms, peak, 0.0, 0.0, 0.0, np.zeros(128)
    centroid = float(np.dot(freq, spectrum) / power)
    low_ratio = float(spectrum[freq <= 1000].sum() / power)
    geometric = float(np.exp(np.mean(np.log(np.maximum(spectrum, 1e-20)))))
    flatness = geometric / float(np.mean(spectrum))
    # 128 logarithmic-power frequency buckets (0..2500 Hz) for overview plot.
    edges = np.linspace(0, min(2500, ANALYSIS_RATE / 2), 129)
    spectrum_db = 10 * np.log10(np.maximum(spectrum, 1e-20))
    bucket = np.interp((edges[1:] + edges[:-1]) / 2, freq, spectrum_db)
    return rms, peak, centroid, low_ratio, flatness, bucket


def _write_rows(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _segments(mask, starts, ends, *, gap=0, minimum=1):
    """Merge adjacent/nearby true frames without overlapping intervals."""
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    result = []
    left = int(indices[0])
    right = left
    for idx in indices[1:]:
        idx = int(idx)
        if starts[idx] - ends[right] <= gap + 1e-9:
            right = idx
        else:
            if ends[right] - starts[left] >= minimum:
                result.append((left, right))
            left = right = idx
    if ends[right] - starts[left] >= minimum:
        result.append((left, right))
    return result


def _hours(duration, events, quiet, starts, rms):
    rows = []
    for hour in range(math.ceil(duration / 3600)):
        begin, end = hour * 3600., min(duration, (hour+1)*3600.)
        def overlap(intervals):
            return sum(max(0., min(end, e["end_s"]) - max(begin, e["start_s"]))
                       for e in intervals)
        within = (starts >= begin) & (starts < end)
        rows.append({
            "hour_index": hour,
            "start_s": round(begin, 3),
            "end_s": round(end, 3),
            "candidate_activity_seconds": round(overlap(events), 3),
            "quiet_seconds": round(overlap(quiet), 3),
            "candidate_event_count": sum(begin <= e["start_s"] < end for e in events),
            "mean_rms": round(float(np.mean(rms[within])), 6) if np.any(within) else 0.,
        })
    return rows


def _plot(starts, rms, spec, threshold, events, out):
    # Limit dense scatter points but never change event interval computation.
    max_columns = 1400
    stride = max(1, math.ceil(len(starts) / max_columns))
    t = starts[::stride] / 3600.
    # For long recordings compress spectra by maximum dB in each column.
    matrix = np.stack([spec[i:i+stride].max(axis=0)
                       for i in range(0, len(starts), stride)], axis=1)
    matrix -= np.max(matrix)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(starts / 3600., rms, linewidth=.4)
    axes[0].axhline(threshold, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("1 s RMS")
    axes[0].set_title("All-night acoustic activity (candidates, not confirmed snoring)")
    axes[1].imshow(matrix, origin="lower", aspect="auto", extent=[
        0, max(float(starts[-1]+1)/3600, 1/3600), 0, 2500],
        cmap="viridis", vmin=-80, vmax=0, interpolation="nearest")
    axes[1].set_xlabel("Hours since recording start")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Time-compressed STFT-like 1 s window spectrum, relative dB")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def analyze_overnight(source: str | Path, output: str | Path, *,
                      quiet_min_seconds: float = 10.,
                      event_min_seconds: float = 2.,
                      merge_gap_seconds: float = 1.):
    """Read sequentially with SoundFile; retain only per-second statistics.

    Supports libsndfile decodable WAV/FLAC/OGG. MP3/M4A may require ffmpeg
    conversion to WAV or FLAC before use.
    """
    source, output = Path(source), Path(output)
    if min(quiet_min_seconds, event_min_seconds) <= 0 or merge_gap_seconds < 0:
        raise ValueError("Invalid interval durations")
    if not source.is_file():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    rows, spectra = [], []
    with sf.SoundFile(source) as audio:
        rate, channels, total = audio.samplerate, audio.channels, len(audio)
        if total == 0:
            raise ValueError("Empty recording")
        # Sequential 1-second reads: no full-night recording in RAM.
        index = 0
        while True:
            data = audio.read(frames=rate, dtype="float32", always_2d=True)
            if not len(data):
                break
            samples = data.mean(axis=1)
            rms, peak, centroid, low_ratio, flatness, bucket = _one_second_features(
                samples, rate)
            rows.append({
                "start_s": index / rate,
                "end_s": (index + len(samples)) / rate,
                "rms": rms, "peak": peak,
                "spectral_centroid_hz": centroid,
                "low_power_ratio": low_ratio,
                "spectral_flatness": flatness,
            })
            spectra.append(bucket)
            index += len(samples)
    duration = total / rate
    starts = np.array([r["start_s"] for r in rows])
    ends = np.array([r["end_s"] for r in rows])
    rms = np.array([r["rms"] for r in rows])
    baseline = float(np.percentile(rms, 20))
    median = float(np.median(rms))
    mad = float(np.median(np.abs(rms - median)))
    q90 = float(np.percentile(rms, 90))
    threshold = max(baseline + 2.5*mad, baseline + .18*(q90-baseline), 1e-5)
    active = rms >= threshold
    groups = _segments(active, starts, ends, gap=merge_gap_seconds,
                       minimum=event_min_seconds)
    events = []
    for k, (a, b) in enumerate(groups, 1):
        events.append({
            "event_id": k, "type": "candidate_acoustic_activity",
            "start_s": float(starts[a]), "end_s": float(ends[b]),
            "duration_s": float(ends[b] - starts[a]),
            "mean_rms": float(np.mean(rms[a:b+1])),
            "mean_centroid_hz": float(np.mean(
                [r["spectral_centroid_hz"] for r in rows[a:b+1]])),
            "mean_low_power_ratio": float(np.mean(
                [r["low_power_ratio"] for r in rows[a:b+1]])),
        })
    quiet = [{"start_s": float(starts[a]), "end_s": float(ends[b]),
              "duration_s": float(ends[b] - starts[a])}
             for a, b in _segments(~active, starts, ends, minimum=quiet_min_seconds)]
    hourly = _hours(duration, events, quiet, starts, rms)
    _write_rows(output / "frame_features.csv", FEATURES, rows)
    _write_rows(output / "events.csv", EVENT_FIELDS, events)
    _write_rows(output / "quiet_intervals.csv", QUIET_FIELDS, quiet)
    _write_rows(output / "hourly.csv", HOUR_FIELDS, hourly)
    _plot(starts, rms, spectra, threshold, events, output / "overview.png")
    summary = {
        "source_file": source.name, "duration_seconds": duration,
        "sample_rate_hz": rate, "channels": channels,
        "analysis_window_seconds": 1.0, "rms_threshold": threshold,
        "candidate_event_count": len(events),
        "candidate_activity_total_seconds": sum(e["duration_s"] for e in events),
        "quiet_interval_count": len(quiet),
        "quiet_min_seconds": quiet_min_seconds,
        "event_min_seconds": event_min_seconds,
        "event_merge_gap_seconds": merge_gap_seconds,
        "interpretation": "Energy-based acoustic activity only, not validated snoring, gasping, apnea or diagnosis. Quiet sound is not absence of breathing.",
        "privacy": "Source audio and generated reports stay local; do not publish personal recordings.",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    html_text = """<!doctype html><html lang="en"><meta charset="utf-8"><title>SleepAudioLab night report</title>
<style>body{font:16px system-ui;max-width:980px;margin:40px auto;padding:0 18px;line-height:1.6}
img{max-width:100%}table{border-collapse:collapse}td,th{padding:6px 14px;border:1px solid #aaa}</style>
<h1>SleepAudioLab — overnight acoustic report</h1>
<p><strong>Exploratory activity detection, not snoring or medical diagnosis.</strong>
Quiet periods do not imply apnea.</p>"""
    html_text += f"<p>Source: {html.escape(source.name)}; duration: {duration/3600:.2f} h; detected activity episodes: {len(events)}</p>"
    html_text += '<img src="overview.png" alt="Overview of energy and spectrum">'
    html_text += "<h2>Hourly acoustic activity</h2><table><tr><th>Hour</th><th>Candidate activity (s)</th><th>Quiet (s)</th><th>Episodes starting</th></tr>"
    for row in hourly:
        html_text += f"<tr><td>{row['hour_index']+1}</td><td>{row['candidate_activity_seconds']:.1f}</td><td>{row['quiet_seconds']:.1f}</td><td>{row['candidate_event_count']}</td></tr>"
    html_text += '</table><p>Review events.csv and quiet_intervals.csv for exact intervals. No audio is embedded.</p></html>'
    (output / "report.html").write_text(html_text, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/overnight"))
    parser.add_argument("--quiet-min", type=float, default=10.)
    parser.add_argument("--event-min", type=float, default=2.)
    parser.add_argument("--merge-gap", type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(analyze_overnight(
        args.audio, args.out, quiet_min_seconds=args.quiet_min,
        event_min_seconds=args.event_min, merge_gap_seconds=args.merge_gap), indent=2))


if __name__ == "__main__":
    main()
