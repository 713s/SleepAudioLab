"""Fit classical models on official ICSD strong TRAIN; evaluate on VALIDATION.

Official ICSD TEST is never read. Restricted data and models remain private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .classical_classifier import SAMPLE_RATE, extract_features, load_clip, read_manifest


def _features(records):
    return np.stack([extract_features(load_clip(path)) for path, _, _ in records])


def train_icsd(train_manifest: str | Path, val_manifest: str | Path,
               output: str | Path, *, seed: int = 42, dataset_id: str = "icsd-strong"):
    train = read_manifest(train_manifest)
    validation = read_manifest(val_manifest)
    # Defensive validation: require the official adapter's split records.
    import csv
    for path, required in ((train_manifest, "train"), (val_manifest, "validation")):
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not {"source_split", "subset", "source_id"}.issubset(reader.fieldnames or []):
                raise ValueError("Use ICSD adapter manifests with explicit split metadata")
            rows = list(reader)
            if len(rows) != (len(train) if required == "train" else len(validation)):
                raise ValueError("Manifest row count mismatch")
            if any(row["source_split"] != required for row in rows):
                raise ValueError("Official train/validation split mismatch")
            if any(not row["source_id"] for row in rows):
                raise ValueError("Missing source_id")
            allowed = {"train_real", "train_synth"} if required == "train" else {"validation"}
            if any(row["subset"] not in allowed for row in rows):
                raise ValueError("Unexpected ICSD subset; test is strictly forbidden")
    # Exact audio overlap guard, including cross-encoded clips only if exact bytes.
    train_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in train}
    val_hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p, _, _ in validation}
    if train_hashes.intersection(val_hashes):
        raise ValueError("Exact clip duplicate across train and validation")
    train_sources = {group for _, _, group in train}
    val_sources = {group for _, _, group in validation}
    if train_sources.intersection(val_sources):
        raise ValueError("Recording source overlaps between train and validation")
    y_train = np.asarray([label for _, label, _ in train])
    y_val = np.asarray([label for _, label, _ in validation])
    if len(np.unique(y_train)) != 2 or len(np.unique(y_val)) != 2:
        raise ValueError("Both splits must contain both positive and negative labels")
    x_train, x_val = _features(train), _features(validation)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    models = {
        "svm_rbf": make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf")),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=2),
    }
    results = {}
    for name, model in models.items():
        model.fit(x_train, y_train)  # ONLY TRAIN samples.
        prediction = model.predict(x_val)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_val, prediction, labels=[1], zero_division=0)
        results[name] = {
            "precision_snore": float(precision[0]),
            "recall_snore": float(recall[0]),
            "f1_snore": float(f1[0]),
            "positive_support": int(support[0]),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y_val, prediction, labels=[0, 1]).tolist(),
            "classification_report": classification_report(
                y_val, prediction, labels=[0, 1],
                zero_division=0, output_dict=True),
        }
        joblib.dump(model, out / f"{name}.joblib")
    metadata = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "source_manifest": Path(train_manifest).name,
        "split": "official_icsd_strong_train_validation",
        "sample_rate_hz": SAMPLE_RATE,
        "clip_duration_seconds": 1.0,
        "feature_version": "v0.3-33-mfcc-htk-mel",
        "feature_dimension": 33,
        "class_labels": {"0": "non_snore", "1": "snore"},
        "models": sorted(models),
        "training_audio_sha256": sorted(train_hashes),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "notes": "Trained on ICSD strong train only; validated on official strong validation; official test untouched.",
    }
    report = {
        "training_clips": len(train),
        "validation_clips": len(validation),
        "test_used": False,
        "results": results,
        "note": "Validation scores are for model development, not official test performance.",
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "validation_results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("models/icsd-strong"))
    args = parser.parse_args()
    print(json.dumps(train_icsd(args.train, args.validation, args.out), indent=2))


if __name__ == "__main__":
    main()
