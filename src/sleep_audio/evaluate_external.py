"""Evaluate frozen SVM/RF on a separate labeled dataset, with NO fitting.

Security: joblib uses pickle, so load only model files YOU produced/trust.
Expected CSV: path,label[,source_id]. The external dataset must have a
different dataset_id from the model training set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from .classical_classifier import SAMPLE_RATE, extract_features, load_clip, read_manifest


def evaluate_external(manifest: str | Path, models_dir: str | Path, output: str | Path,
                      *, dataset_id: str):
    models_dir = Path(models_dir)
    output = Path(output)
    if not dataset_id.strip():
        raise ValueError("An explicit --dataset-id is required")
    metadata = json.loads((models_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1 or metadata["feature_dimension"] != 33:
        raise ValueError("Unsupported model format or feature schema")
    if metadata["feature_version"] != "v0.3-33-mfcc-htk-mel":
        raise ValueError("Feature version mismatch")
    if metadata["sample_rate_hz"] != SAMPLE_RATE or metadata["clip_duration_seconds"] != 1.0:
        raise ValueError("Audio preprocessing contract mismatch")
    if metadata["dataset_id"].strip().casefold() == dataset_id.strip().casefold():
        raise ValueError("External dataset_id must differ from the training dataset_id")

    records = read_manifest(manifest)
    existing = set(metadata["training_audio_sha256"])
    overlapping = [
        str(path) for path, _, _ in records
        if hashlib.sha256(path.read_bytes()).hexdigest() in existing
    ]
    if overlapping:
        raise ValueError(
            f"External dataset contains {len(overlapping)} exact duplicate(s) "
            "of training files; aborting evaluation")
    x = np.stack([extract_features(load_clip(path)) for path, _, _ in records])
    if x.shape[1] != metadata["feature_dimension"]:
        raise ValueError("Feature dimension mismatch")
    y = np.asarray([label for _, label, _ in records])
    if len(np.unique(y)) != 2:
        raise ValueError("External evaluation requires BOTH positive and negative labels")
    results = {}
    for name in metadata["models"]:
        # No fit()/fit_transform() anywhere in this evaluation path.
        # joblib loading is unsafe for untrusted files.
        model = joblib.load(models_dir / f"{name}.joblib")
        predictions = model.predict(x)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(x)
        elif hasattr(model, "predict_proba"):
            scores = model.predict_proba(x)[:, list(model.classes_).index(1)]
        else:
            scores = None
        precision, recall, f1, support = precision_recall_fscore_support(
            y, predictions, labels=[1], zero_division=0)
        results[name] = {
            "precision_snore": float(precision[0]),
            "recall_snore": float(recall[0]),
            "f1_snore": float(f1[0]),
            "positive_support": int(support[0]),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y, predictions, labels=[0, 1]).tolist(),
            "average_precision": float(average_precision_score(y, scores))
                if scores is not None else None,
            "roc_auc": float(roc_auc_score(y, scores))
                if scores is not None else None,
            "classification_report": classification_report(
                y, predictions, labels=[0, 1], zero_division=0, output_dict=True),
        }
    report = {
        "evaluation_type": "frozen_model_external_dataset",
        "train_dataset_id": metadata["dataset_id"],
        "test_dataset_id": dataset_id,
        "test_clip_count": len(records),
        "positive_count": int((y == 1).sum()),
        "negative_count": int((y == 0).sum()),
        "exact_train_file_overlap": 0,
        "model_schema_version": metadata["schema_version"],
        "feature_version": metadata["feature_version"],
        "models": results,
        "limitations": (
            "File SHA256 removes exact duplicates only, not re-encoded or overlapping "
            "audio sources. Dataset independence and clinical relevance require "
            "independent provenance review."),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "external_evaluation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True,
                        help="Dataset identifier distinct from the training data")
    parser.add_argument("--out", type=Path, default=Path("reports/external"))
    args = parser.parse_args()
    print(json.dumps(evaluate_external(
        args.manifest, args.models, args.out, dataset_id=args.dataset_id), indent=2))


if __name__ == "__main__":
    main()
