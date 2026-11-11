"""No gated data used: tests generate synthetic audio and strong TSV."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sleep_audio.icsd_training_adapter import convert_icsd_train_validation, union_coverage
from sleep_audio.train_icsd import train_icsd


def build_fake_dataset(root: Path):
    locations = {
        "train_real": ("audio/train/real_strong_train", "metadata/train/real_strong_train.tsv"),
        "train_synth": ("audio/train/synth_strong_train", "metadata/train/synth_strong_train.tsv"),
        "validation": ("audio/validation/strong_val", "metadata/validation/strong_val.tsv"),
    }
    for label, (relative_audio, relative_tsv) in locations.items():
        folder = root / relative_audio
        folder.mkdir(parents=True)
        metadata = root / relative_tsv
        metadata.parent.mkdir(parents=True, exist_ok=True)
        lines = ["filename\tonset\toffset\tevent_label"]
        for number in range(3):
            rate = 8000
            t = np.arange(rate * 4) / rate
            frequency = {"train_real": 100, "train_synth": 135, "validation": 160}[label] + number
            waveform = .3 * np.sin(2 * np.pi * frequency * t)
            name = f"{label}_{number}.wav"
            sf.write(folder / name, waveform, rate)
            lines.extend([
                f"{name}\t0\t1\tSnoring",
                f"{name}\t2\t3\tInfantcry",
            ])
        metadata.write_text("\n".join(lines), encoding="utf-8")


def test_window_union():
    assert union_coverage([(0, .8), (.1, .9)], 0, 1) == pytest.approx(.9)


def test_train_validation_conversion_and_frozen_training(tmp_path):
    root = tmp_path / "Dataset"
    build_fake_dataset(root)
    out = tmp_path / "processed"
    audit = convert_icsd_train_validation(root, out, assume_exhaustive=True)
    assert audit["official_test_policy"] == "never opened"
    assert audit["subsets"]["train_real"]["positive"] == 3
    assert audit["subsets"]["train_synth"]["negative"] == 6
    assert audit["subsets"]["validation"]["positive"] == 3
    with (out / "train_manifest.csv").open() as file:
        train = list(csv.DictReader(file))
    with (out / "validation_manifest.csv").open() as file:
        val = list(csv.DictReader(file))
    assert len(train) == 18
    assert len(val) == 9
    assert {row["source_split"] for row in train} == {"train"}
    assert {row["source_split"] for row in val} == {"validation"}
    result = train_icsd(out / "train_manifest.csv", out / "validation_manifest.csv",
                        tmp_path / "models")
    assert result["test_used"] is False
    assert (tmp_path / "models" / "svm_rbf.joblib").is_file()
    assert (tmp_path / "models" / "random_forest.joblib").is_file()
    assert (tmp_path / "models" / "model_metadata.json").is_file()


def test_default_does_not_invent_negatives(tmp_path):
    root = tmp_path / "Dataset"
    build_fake_dataset(root)
    out = tmp_path / "processed"
    audit = convert_icsd_train_validation(root, out)
    assert audit["subsets"]["train_real"]["negative"] == 0
    with pytest.raises(ValueError, match="both positive and negative"):
        train_icsd(out / "train_manifest.csv", out / "validation_manifest.csv",
                   tmp_path / "models")


def test_rejects_test_substitution(tmp_path):
    root = tmp_path / "Dataset"
    build_fake_dataset(root)
    out = tmp_path / "processed"
    convert_icsd_train_validation(root, out, assume_exhaustive=True)
    manifest = out / "validation_manifest.csv"
    text = manifest.read_text().replace(",validation,validation", ",test,test")
    manifest.write_text(text)
    with pytest.raises(ValueError, match="split mismatch"):
        train_icsd(out / "train_manifest.csv", manifest, tmp_path / "models")


def test_never_reads_test_directory(tmp_path):
    root = tmp_path / "Dataset"
    build_fake_dataset(root)
    testdir = root / "audio/test/real_strong_test"
    testdir.mkdir(parents=True)
    (testdir / "unreadable.wav").write_bytes(b"not an audio file")
    audit = convert_icsd_train_validation(root, tmp_path / "processed",
                                          assume_exhaustive=True)
    assert audit["official_test_policy"] == "never opened"
