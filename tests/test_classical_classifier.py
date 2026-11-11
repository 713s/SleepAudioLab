import numpy as np
import soundfile as sf
import pytest

from sleep_audio.classical_classifier import (
    extract_features, mel_filters, read_manifest, train_and_evaluate
)


def test_mel_filter_bank_and_silence():
    assert mel_filters(16000, 512).shape == (26, 257)
    feature = extract_features(np.zeros(16000))
    assert feature.shape == (33,)
    assert np.all(np.isfinite(feature))


def test_tones_have_different_low_frequency_ratios():
    time = np.arange(16000) / 16000
    a = extract_features(0.4 * np.sin(2 * np.pi * 200 * time))
    b = extract_features(0.4 * np.sin(2 * np.pi * 1700 * time))
    assert a[4] > b[4]


def test_group_disjoint_training_and_metrics(tmp_path):
    rng = np.random.default_rng(123)
    rows = ["path,label,source_id"]
    for i in range(24):
        label = i % 2
        time = np.arange(16000) / 16000
        freq = 150 if label else 1200
        signal = 0.3 * np.sin(2 * np.pi * freq * time) + rng.normal(0, 0.015, 16000)
        path = tmp_path / f"{i}.wav"
        sf.write(path, signal, 16000)
        rows.append(f"{path.name},{label},source_{i}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows))
    assert len(read_manifest(manifest)) == 24
    report = train_and_evaluate(manifest, tmp_path / "results")
    assert report["split"] == "source_group_disjoint"
    assert (tmp_path / "results" / "evaluation.json").is_file()
    assert 0 <= report["models"]["svm_rbf"]["f1_snore"] <= 1

    manifest.write_text("\n".join(
        [rows[0]] + [line.rsplit(",", 1)[0] + "," for line in rows[1:]]
    ))
    with pytest.raises(ValueError, match="source_id"):
        train_and_evaluate(manifest, tmp_path / "untrusted")
