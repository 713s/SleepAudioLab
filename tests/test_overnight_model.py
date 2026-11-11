"""Synthetic-only tests for optional frozen-model overnight inference."""
from __future__ import annotations

import csv
import json

import joblib
import numpy as np
import pytest
import soundfile as sf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sleep_audio.overnight import analyze_overnight
from sleep_audio.overnight_model import predict_overnight


def test_model_inference_and_hourly_export(tmp_path):
    wave = tmp_path / "sample.wav"
    rate = 8000
    t = np.arange(rate * 3)/rate
    sf.write(wave, (.2*np.sin(2*np.pi*130*t)).astype("float32"), rate)
    output = tmp_path/"reports"
    analyze_overnight(wave, output, quiet_min_seconds=1)
    model_dir = tmp_path/"my_models"
    model_dir.mkdir()
    # Training only on artificial tabular features: no third-party audio.
    model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    model.fit(np.array([[0.]*33, [1.]*33, [2.]*33, [3.]*33]), [0, 0, 1, 1])
    joblib.dump(model, model_dir/"svm_rbf.joblib")
    (model_dir/"model_metadata.json").write_text(json.dumps({
        "sample_rate_hz": 16000, "feature_dimension": 33,
        "feature_version": "v0.3-33-mfcc-htk-mel",
        "models": ["svm_rbf"]}))
    summary = predict_overnight(wave, output, model_dir)
    assert "model_snore_candidate_seconds" in summary
    assert (output/"model_hourly.csv").is_file()
    with (output/"model_frame_predictions.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 3
    assert "Optional model-estimated snore candidates" in (output/"report.html").read_text()


def test_reject_incompatible_model_without_loading_pickle(tmp_path):
    model_dir = tmp_path/"model"
    model_dir.mkdir()
    (model_dir/"model_metadata.json").write_text(json.dumps({
        "sample_rate_hz": 44100, "feature_dimension": 33,
        "feature_version": "unknown", "models": ["svm_rbf"]}))
    with pytest.raises(ValueError, match="Incompatible"):
        predict_overnight(tmp_path/"missing.wav", tmp_path/"out", model_dir)
