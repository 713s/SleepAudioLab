"""Synthetic numerical tests do not claim real snoring classification accuracy."""
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sleep_audio.classical_classifier import train_and_evaluate
from sleep_audio.evaluate_external import evaluate_external


def _manifest(root: Path, prefix: str, *, seed: int):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lines = ["path,label,source_id"]
    fs = 16000
    t = np.arange(fs) / fs
    for i in range(24):
        label = i % 2
        frequency = 130 if label else 1450
        signal = (0.25 * np.sin(2 * np.pi * frequency * t) +
                  rng.normal(0, 0.035, fs))
        path = root / f"{prefix}_{i:02}.wav"
        sf.write(path, signal, fs)
        lines.append(f"{path.name},{label},{prefix}_source_{i}")
    manifest = root / "manifest.csv"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def test_frozen_models_evaluate_without_refitting(tmp_path, monkeypatch):
    train = _manifest(tmp_path / "train", "train", seed=4)
    external = _manifest(tmp_path / "external", "external", seed=75)
    models = tmp_path / "models"
    train_and_evaluate(train, models, save_models=True, dataset_id="synthetic-train")
    metadata = json.loads((models / "model_metadata.json").read_text())
    assert metadata["feature_dimension"] == 33
    assert len(metadata["training_audio_sha256"]) == 18
    for name in ("svm_rbf", "random_forest"):
        assert (models / f"{name}.joblib").exists()

    # Patch global sklearn fit methods: external evaluation must never fit.
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("External evaluation attempted to train a model")
    monkeypatch.setattr(Pipeline, "fit", forbidden_fit)
    monkeypatch.setattr(RandomForestClassifier, "fit", forbidden_fit)
    report = evaluate_external(
        external, models, tmp_path / "results", dataset_id="new-source")
    assert report["evaluation_type"] == "frozen_model_external_dataset"
    assert report["test_clip_count"] == 24
    for result in report["models"].values():
        assert 0 <= result["f1_snore"] <= 1
        assert 0 <= result["average_precision"] <= 1
    assert (tmp_path / "results" / "external_evaluation.json").exists()


def test_guard_against_same_dataset_and_exact_audio_overlap(tmp_path):
    train = _manifest(tmp_path / "training", "tr", seed=1)
    external = _manifest(tmp_path / "new", "new", seed=2)
    models = tmp_path / "models"
    train_and_evaluate(train, models, save_models=True, dataset_id="set-one")
    with pytest.raises(ValueError, match="differ"):
        evaluate_external(external, models, tmp_path / "report", dataset_id="set-one")

    # Use one existing TRAIN file under a different label manifest: block exact reuse.
    training_files = sorted((tmp_path / "training").glob("tr_*.wav"))
    fresh_files = sorted((tmp_path / "new").glob("new_*.wav"))
    colliding = tmp_path / "collision.csv"
    colliding.write_text(
        "path,label,source_id\n" +
        f"{training_files[0]},0,x\n{fresh_files[1]},1,y\n",
        encoding="utf-8")
    # One of the 24 training files might fall in held-out validation; insist on a hash
    # in the actual model training set rather than assuming index 0 was in it.
    hashes = set(json.loads((models / "model_metadata.json").read_text())
                 ["training_audio_sha256"])
    import hashlib
    matched = next(p for p in training_files
                   if hashlib.sha256(p.read_bytes()).hexdigest() in hashes)
    colliding.write_text(
        "path,label,source_id\n" +
        f"{matched},0,x\n{fresh_files[1]},1,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact duplicate"):
        evaluate_external(colliding, models, tmp_path / "report", dataset_id="new-set")
