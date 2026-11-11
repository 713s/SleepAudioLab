import numpy as np
from sleep_audio.snore_analysis import (
    autocorrelation_periodicity, detect_acoustic_events, frame_features
)


def test_periodicity_detects_clean_tone():
    fs = 4000
    t = np.arange(fs) / fs
    x = np.sin(2 * np.pi * 200 * t)
    result = autocorrelation_periodicity(x, fs)
    assert abs(result["peak_frequency_hz"] - 200) <= 1
    assert result["peak_correlation"] > 0.9


def test_energy_bursts_are_grouped():
    fs = 4000
    x = np.zeros(5 * fs)
    t = np.arange(fs) / fs
    x[fs:2 * fs] = 0.5 * np.sin(2 * np.pi * 160 * t)
    x[3 * fs:4 * fs] = 0.6 * np.sin(2 * np.pi * 160 * t)
    features = frame_features(x, fs)
    threshold, events, quiet = detect_acoustic_events(features, 5.0)
    assert threshold > 0
    assert len(events) == 2
    assert events[0]["start_s"] >= 0.8
    assert events[1]["start_s"] >= 2.8
    assert isinstance(quiet, list)


def test_silence_has_no_candidates():
    fs = 2000
    x = np.zeros(5 * fs)
    f = frame_features(x, fs)
    _, events, quiet = detect_acoustic_events(f, 5)
    assert not events
    assert len(quiet) == 1
    assert autocorrelation_periodicity(x, fs)["peak_frequency_hz"] is None
