"""Classical snore-versus-non-snore clip classification: no neural networks.

Exploratory research baseline only; not a sleep or apnea diagnostic tool.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
import platform

import joblib
import sklearn
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import dct
from scipy.signal import resample_poly
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SAMPLE_RATE = 16000


def load_clip(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    data, original_rate = sf.read(path, always_2d=True, dtype="float32")
    x = data.mean(axis=1).astype("float64")
    if original_rate != sample_rate:
        common = math.gcd(int(original_rate), sample_rate)
        x = resample_poly(x, sample_rate // common, original_rate // common)
    if x.size == 0:
        raise ValueError(f"Empty audio: {path}")
    if x.size < sample_rate:
        x = np.pad(x, (0, sample_rate - x.size))
    return x[:sample_rate]


def _hz_to_mel(f):
    return 2595 * np.log10(1 + np.asarray(f) / 700)


def _mel_to_hz(m):
    return 700 * (10 ** (np.asarray(m) / 2595) - 1)


def mel_filters(rate: int, fft_length: int, bands: int = 26) -> np.ndarray:
    """HTK mel scale with triangular filters."""
    freqs = np.fft.rfftfreq(fft_length, d=1 / rate)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(0), _hz_to_mel(rate / 2), bands + 2))
    filters = np.zeros((bands, len(freqs)))
    for j in range(bands):
        left, mid, right = edges[j:j + 3]
        filters[j] = np.maximum(
            0, np.minimum((freqs - left) / (mid - left),
                          (right - freqs) / (right - mid)))
    return filters


def extract_features(samples: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    """33 features: seven descriptors and 13 MFCC means/standard deviations.

    Frames are 25 ms with a 10 ms hop; a Hann window and 512-point FFT.
    MFCC is log-energy, HTK mel triangles, orthonormal DCT-II, coefficients 0..12.
    """
    x = np.asarray(samples, dtype="float64")
    if x.ndim != 1 or x.size < 400:
        raise ValueError("Expected at least 400 mono samples")
    starts = np.arange(0, len(x) - 400 + 1, 160)
    frames = np.stack([x[p:p + 400] for p in starts])
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    centered = (frames - frames.mean(axis=1, keepdims=True)) * np.hanning(400)
    power = np.abs(np.fft.rfft(centered, n=512, axis=1)) ** 2
    freqs = np.fft.rfftfreq(512, 1 / rate)
    norm = np.maximum(power.sum(axis=1), 1e-15)
    centroid = (power @ freqs) / norm
    low_ratio = power[:, freqs <= 1000].sum(axis=1) / norm
    flatness = np.exp(np.mean(np.log(np.maximum(power, 1e-15)), axis=1)) / np.maximum(
        power.mean(axis=1), 1e-15)
    log_mel = np.log(np.maximum(power @ mel_filters(rate, 512).T, 1e-12))
    mfcc = dct(log_mel, axis=1, type=2, norm="ortho")[:, :13]

    # Acoustic waveform periodicity in 50..500 Hz, NOT the breathing rate.
    segment = x[rate // 4:rate // 2].copy()
    segment -= segment.mean()
    variance = np.dot(segment, segment)
    periodicity = 0.0
    if variance > 1e-12:
        lo, hi = int(rate / 500), int(rate / 50)
        n = 1 << (2 * len(segment) - 1).bit_length()
        fft = np.fft.rfft(segment, n)
        acf = np.fft.irfft(fft * fft.conj(), n)
        periodicity = float(max(0., acf[lo:hi + 1].max() / variance))

    primary = [
        np.log10(max(float(rms.mean()), 1e-12)), float(rms.std()),
        float(centroid.mean()), float(centroid.std()),
        float(low_ratio.mean()), float(flatness.mean()), periodicity,
    ]
    return np.array(primary + mfcc.mean(axis=0).tolist() +
                    mfcc.std(axis=0).tolist(), dtype=float)


def read_manifest(manifest: str | Path):
    path = Path(manifest)
    rows = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not {"path", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest must have path,label columns")
        for row in reader:
            label = str(row["label"]).strip().lower()
            if label in {"1", "snore", "snoring"}:
                target = 1
            elif label in {"0", "non_snore", "non-snore", "negative"}:
                target = 0
            else:
                raise ValueError(f"Unexpected label: {label}")
            audio = Path(row["path"])
            if not audio.is_absolute():
                audio = path.parent / audio
            if not audio.is_file():
                raise FileNotFoundError(audio)
            rows.append((audio, target, (row.get("source_id") or "").strip()))
    if not rows:
        raise ValueError("Empty manifest")
    return rows


def train_and_evaluate(manifest: str | Path, output: str | Path, *,
                       allow_clip_split: bool = False, seed: int = 42,
                       save_models: bool = False, dataset_id: str = "khan-snoring"):
    records = read_manifest(manifest)
    has_groups = all(group for _, _, group in records)
    if not has_groups and not allow_clip_split:
        raise ValueError("Missing source_id: to accept possible data leakage, "
                         "pass --allow-clip-split explicitly")
    x = np.stack([extract_features(load_clip(path)) for path, _, _ in records])
    y = np.asarray([label for _, label, _ in records])
    if len(np.unique(y)) != 2:
        raise ValueError("Both classes are required")
    if has_groups:
        groups = np.asarray([group for _, _, group in records])
        train, test = next(GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed).split(x, y, groups))
    else:
        train, test = train_test_split(
            np.arange(len(y)), test_size=0.25, random_state=seed, stratify=y)
    if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
        raise ValueError("Train/test split must each contain both classes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    models = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed),
    }
    for name, model in models.items():
        model.fit(x[train], y[train])
        if save_models:
            joblib.dump(model, output / f"{name}.joblib")
        prediction = model.predict(x[test])
        precision, recall, f1, support = precision_recall_fscore_support(
            y[test], prediction, labels=[1], zero_division=0)
        results[name] = {
            "precision_snore": float(precision[0]),
            "recall_snore": float(recall[0]),
            "f1_snore": float(f1[0]),
            "positive_test_support": int(support[0]),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y[test], prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y[test], prediction, labels=[0, 1],
                zero_division=0, output_dict=True),
        }
    report = {
        "samples": len(records),
        "train_samples": len(train),
        "test_samples": len(test),
        "split": "source_group_disjoint" if has_groups else "exploratory_clip_split",
        "warning": None if has_groups else
            "Original recording independence unknown. Scores may be inflated "
            "by repeated source recordings or correlated audio segments.",
        "models": results,
    }
    if save_models:
        # Freeze the exact fitted pipeline and its feature extraction contract.
        # Hashes allow detecting exact audio reuse across datasets; no audio is saved.
        train_hashes = sorted({
            hashlib.sha256(records[int(i)][0].read_bytes()).hexdigest()
            for i in train
        })
        metadata = {
            "schema_version": 1,
            "dataset_id": dataset_id,
            "source_manifest": str(Path(manifest).name),
            "split": report["split"],
            "seed": seed,
            "sample_rate_hz": SAMPLE_RATE,
            "clip_duration_seconds": 1.0,
            "feature_version": "v0.3-33-mfcc-htk-mel",
            "feature_dimension": int(x.shape[1]),
            "class_labels": {"0": "non_snore", "1": "snore"},
            "models": sorted(models),
            "training_audio_sha256": train_hashes,
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "notes": ("This is a clip-level exploratory baseline; training "
                      "recording provenance may be unknown."),
        }
        (output / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\\n", encoding="utf-8")
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("reports/v03"))
    parser.add_argument("--allow-clip-split", action="store_true")
    parser.add_argument("--save-models", action="store_true",
                        help="Save frozen joblib models and metadata for later inference")
    parser.add_argument("--dataset-id", default="khan-snoring",
                        help="Stable source dataset identifier")
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(
        args.manifest, args.out, allow_clip_split=args.allow_clip_split,
        save_models=args.save_models, dataset_id=args.dataset_id), indent=2))


if __name__ == "__main__":
    main()
