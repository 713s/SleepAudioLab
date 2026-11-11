"""End-to-end private overnight report tests on generated WAVs only."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from sleep_audio.overnight import analyze_overnight


def _csv(path):
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_overnight_report_two_hours_and_boundary(tmp_path):
    # Two hours at a low sample rate proves streaming and interval accounting.
    rate = 1000
    audio = tmp_path / "night.wav"
    with sf.SoundFile(audio, mode="w", samplerate=rate, channels=1,
                      subtype="PCM_16") as out:
        zero = np.zeros(rate * 60, dtype=np.float32)
        tone = .4 * np.sin(2 * np.pi * 120 * np.arange(rate)/rate).astype("float32")
        for minute in range(121):
            block = zero.copy()
            if minute in {59, 60, 119}:
                block[:rate * 3] = np.tile(tone, 3)
            out.write(block)
    folder = tmp_path / "results"
    summary = analyze_overnight(audio, folder)
    assert summary["duration_seconds"] == pytest.approx(121*60)
    assert summary["candidate_event_count"] == 3
    assert len(_csv(folder / "hourly.csv")) == 3
    assert _csv(folder / "hourly.csv")[0]["candidate_event_count"] == "1"
    assert _csv(folder / "hourly.csv")[1]["candidate_event_count"] == "2"
    assert _csv(folder / "hourly.csv")[2]["candidate_event_count"] == "0"
    assert (folder / "overview.png").stat().st_size > 1000
    assert "not snoring or medical diagnosis" in (folder / "report.html").read_text()
    assert json.loads((folder / "summary.json").read_text())["candidate_event_count"] == 3


def test_short_and_silent_audio(tmp_path):
    audio = tmp_path / "silent.wav"
    sf.write(audio, np.zeros(1500), 1000)
    out = tmp_path / "out"
    summary = analyze_overnight(audio, out, quiet_min_seconds=1)
    assert summary["candidate_event_count"] == 0
    assert summary["quiet_interval_count"] == 1
    assert len(_csv(out / "frame_features.csv")) == 2
    assert _csv(out / "quiet_intervals.csv")[0]["duration_s"] == "1.5"


def test_empty_is_error(tmp_path):
    audio = tmp_path / "empty.wav"
    sf.write(audio, np.empty(0), 1000)
    with pytest.raises(ValueError, match="Empty"):
        analyze_overnight(audio, tmp_path / "out")
