"""Apply a trusted, locally trained ICSD 1-second classifier to a full night.

A separate optional second pass after the model-free overnight report.
The model estimates snoring-like clips; it is not a clinical diagnosis.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .classical_classifier import SAMPLE_RATE, extract_features


def _save_csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def predict_overnight(source: str | Path, output_dir: str | Path,
                      model_dir: str | Path, *, batch_size: int = 128):
    """Stream file and predict with an existing 33-feature binary SVM.

    This module never trains, fetches data, or uses restricted ICSD audio.
    Only load your OWN trusted local joblib models.
    """
    source, output_dir, model_dir = Path(source), Path(output_dir), Path(model_dir)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if (metadata.get("sample_rate_hz") != SAMPLE_RATE or
        metadata.get("feature_dimension") != 33 or
        metadata.get("feature_version") != "v0.3-33-mfcc-htk-mel" or
        "svm_rbf" not in metadata.get("models", [])):
        raise ValueError("Incompatible model feature contract")
    model = joblib.load(model_dir / "svm_rbf.joblib")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    predictions = []
    feats = []
    with sf.SoundFile(source) as f:
        rate, total = f.samplerate, len(f)
        while True:
            data = f.read(rate, dtype="float32", always_2d=True)
            if not len(data):
                break
            mono = data.mean(axis=1).astype(np.float64)
            common = math.gcd(rate, SAMPLE_RATE)
            x = resample_poly(mono, SAMPLE_RATE // common, rate // common)
            if len(x) < SAMPLE_RATE:
                x = np.pad(x, (0, SAMPLE_RATE - len(x)))
            feats.append(extract_features(x[:SAMPLE_RATE]))
            if len(feats) >= batch_size:
                predictions.extend(int(v) for v in model.predict(np.stack(feats)))
                feats.clear()
        if feats:
            predictions.extend(int(v) for v in model.predict(np.stack(feats)))
    if not set(predictions).issubset({0, 1}):
        raise ValueError("Model did not return binary predictions")
    starts = np.arange(len(predictions), dtype=np.float64)
    ends = np.minimum(starts + 1, total / rate)
    intervals = []
    open_start = None
    for i, value in enumerate(predictions):
        if value and open_start is None:
            open_start = i
        if open_start is not None and (not value or i == len(predictions)-1):
            last = i if value else i-1
            intervals.append({
                "event_id": len(intervals) + 1,
                "type": "model_snore_candidate",
                "start_s": float(starts[open_start]),
                "end_s": float(ends[last]),
                "duration_s": float(ends[last] - starts[open_start]),
            })
            open_start = None
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(output_dir / "model_frame_predictions.csv",
              ["start_s", "end_s", "model_snore_candidate"],
              [{"start_s": float(a), "end_s": float(b),
                "model_snore_candidate": label}
               for a, b, label in zip(starts, ends, predictions)])
    _save_csv(output_dir / "model_snore_candidates.csv",
              ["event_id", "type", "start_s", "end_s", "duration_s"], intervals)
    rows = []
    for hour in range(math.ceil(total / rate / 3600)):
        a, b = hour*3600, min((hour+1)*3600, total/rate)
        seconds = sum(max(0., min(b, e["end_s"])-max(a, e["start_s"]))
                      for e in intervals)
        rows.append({
            "hour_index": hour,
            "model_snore_candidate_seconds": round(seconds, 3),
            "model_snore_candidate_count": sum(
                a <= e["start_s"] < b for e in intervals),
        })
    _save_csv(output_dir / "model_hourly.csv",
              ["hour_index", "model_snore_candidate_seconds",
               "model_snore_candidate_count"], rows)
    summary_path = output_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["model_snore_candidate_count"] = len(intervals)
    summary["model_snore_candidate_seconds"] = round(
        sum(x["duration_s"] for x in intervals), 3)
    summary["model_type"] = "trusted_local_svm_rbf"
    summary["model_caution"] = (
        "Estimates from ICSD-trained acoustic model; may not generalize to "
        "home recordings. Not validated snoring or a medical diagnosis.")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                            encoding="utf-8")
    report_path = output_dir / "report.html"
    report = report_path.read_text(encoding="utf-8")
    block = (
        f"<h2>Optional model-estimated snore candidates</h2>"
        f"<p>{len(intervals)} candidate events, "
        f"{summary['model_snore_candidate_seconds']:.1f} seconds. "
        "These are unverified predictions, not a clinical measurement. "
        "See model_snore_candidates.csv and model_hourly.csv.</p>"
    )
    report_path.write_text(report.replace("</html>", block+"</html>"),
                           encoding="utf-8")
    return summary
