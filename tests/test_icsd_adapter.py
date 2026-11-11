"""Synthetic fixtures only: no restricted ICSD content is included in Git."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from sleep_audio.icsd_adapter import convert_icsd_test


def _setup(root: Path):
    wavdir = root / "audio/test/real_strong_test"
    metadir = root / "metadata/test"
    wavdir.mkdir(parents=True)
    metadir.mkdir(parents=True)
    # 5 seconds: 0.0-1.0 snore, 1.0-2.0 quiet, 2.0-3.0 partial,
    # 3.0-4.0 infant cry, 4.0-5.0 snore
    t = np.arange(8000 * 5) / 8000
    sf.write(wavdir / "sample.wav", 0.2 * np.sin(2 * np.pi * 135 * t), 8000)
    (metadir / "real_strong_test.tsv").write_text(
        "filename\tonset\toffset\tevent_label\n"
        "sample.wav\t0\t1\tSnoring\n"
        "sample.wav\t2.2\t2.45\tSnoring\n"
        "sample.wav\t3\t4\tInfantcry\n"
        "sample.wav\t4\t5\tSnoring\n",
        encoding="utf-8",
    )


def test_strong_test_only_and_ambiguous_windows(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    out = tmp_path / "PRIVATE" / "icsd_test"
    result = convert_icsd_test(root, out, assume_exhaustive=True)
    assert result["official_split"] == "test"
    assert result["counts"]["positive"] == 2
    assert result["counts"]["negative"] == 1
    assert result["counts"]["ambiguous"] == 2
    with (out / "test_manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(r["label"]) for r in rows] == [1, 0, 1]
    assert len(set(r["source_id"] for r in rows)) == 1
    for row in rows:
        audio, sample_rate = sf.read(row["path"])
        assert sample_rate == 16000
        assert len(audio) == 16000


def test_default_does_not_invent_negative_labels(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    result = convert_icsd_test(root, tmp_path / "PRIVATE")
    assert result["counts"]["positive"] == 2
    assert result["counts"]["negative"] == 0


def test_rejects_missing_annotations_and_non_test_layout(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    (root / "audio/test/real_strong_test" / "extra.wav").write_bytes(b"test")
    with pytest.raises(ValueError, match="without TSV"):
        convert_icsd_test(root, tmp_path / "PRIVATE")
    with pytest.raises(FileNotFoundError):
        convert_icsd_test(tmp_path / "other", tmp_path / "PRIVATE2")


def test_rejects_unexpected_class_and_unsafe_filename(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    tsv = root / "metadata/test/real_strong_test.tsv"
    tsv.write_text("filename\tonset\toffset\tevent_label\n"
                   "sample.wav\t0\t1\tother\n")
    with pytest.raises(ValueError, match="Unknown event"):
        convert_icsd_test(root, tmp_path / "PRIVATE")
    tsv.write_text("filename\tonset\toffset\tevent_label\n"
                   "../outside.wav\t0\t1\tSnoring\n")
    with pytest.raises(ValueError, match="Unsafe"):
        convert_icsd_test(root, tmp_path / "PRIVATE")


def test_rejects_output_inside_restricted_source(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    with pytest.raises(ValueError, match="outside"):
        convert_icsd_test(root, root / "output")


def test_refuses_exact_test_train_overlap(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    training = root / "audio/train/real_strong_train"
    training.mkdir(parents=True)
    (training / "same_recording.wav").write_bytes(
        (root / "audio/test/real_strong_test/sample.wav").read_bytes())
    with pytest.raises(ValueError, match="exact copy"):
        convert_icsd_test(root, tmp_path / "PRIVATE")


def test_refuses_reusing_frozen_test_manifest(tmp_path):
    root = tmp_path / "Dataset"
    _setup(root)
    output = tmp_path / "PRIVATE"
    convert_icsd_test(root, output, assume_exhaustive=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_icsd_test(root, output, assume_exhaustive=True)
