# Exploratory report: one minute of a CC0 snore recording

Source: `data/samples/snoring_cc0.ogg`. The recording is publicly available from Wikimedia Commons and has no clinically validated event annotations.

## Reproducibility

The GitHub Actions workflow `analyze-public-snore.yml` executes `pytest -q` and then computes these outputs from the repository's original audio file.

Run locally (optional):

```bash
python -m sleep_audio.snore_analysis data/samples/snoring_cc0.ogg --out reports/public_snore
```

Results are in `summary.json`, `events.csv`, `quiet_intervals.csv`, `frame_features.csv` and `analysis.png`.

## Actual measurements

| Quantity | Result |
| --- | ---: |
| Recording length | 59.925 seconds |
| Sample rate | 44,100 Hz |
| Overall digital RMS | 0.011896 |
| Adaptive frame RMS threshold | 0.004699 |
| Candidate above-threshold sound bursts | 17 |
| Total candidate-burst duration | 15.2 seconds |
| Low-recorded-energy intervals lasting >= 3 s | 3 |
| Autocorrelation peak on the highest-RMS burst | 0.2875 |
| Frequency corresponding to that autocorrelation lag | 56.76 Hz |

The durations and counts are **properties of the current detector settings**, not ground-truth snore or clinical event measurements.

The bursts begin approximately every **3.49 seconds on average** (16 inter-onset intervals, sample standard deviation not reported). This is a recurring *burst envelope* on the seconds time scale. It is entirely distinct from the 56.76 Hz peak found when calculating the autocorrelation of an audio waveform inside one burst. The first may plausibly relate to a recurring respiratory sound, but the recording provides no airflow, respiratory effort, or clinical annotation to prove this interpretation. The latter is simply the strongest autocorrelation lag searched within the specified 50–500 Hz band, with a relatively weak normalized peak of 0.2875.

Most candidate bursts have mean spectral centroids around 236–344 Hz and mean power fractions below 1000 Hz around 0.99. These are calculated from finite Hann-window periodograms, so spectral centroid depends on the recording chain, bandwidth, noise and windowing; the low-frequency ratio is **not** a universal criterion for snoring.

## Candidate acoustic event excerpts

| ID | Start (s) | End (s) | Duration (s) |
| --- | ---: | ---: | ---: |
| 1 | 0.85 | 1.65 | 0.80 |
| 5 | 15.45 | 16.35 | 0.90 |
| 9 | 29.05 | 30.05 | 1.00 |
| 13 | 43.05 | 44.25 | 1.20 |
| 17 | 56.65 | 57.85 | 1.20 |

All 17 boundaries are in `events.csv`. The three quiet intervals are 5.15–8.15 s, 8.95–11.95 s, and 16.35–19.35 s. "Quiet" means *recorded energy below the algorithm's threshold*, not absent breathing; these intervals must not be used to diagnose apnea.

## Core caveats

1. The input has a class-level descriptive label from the uploader ("snoring"), but **no timestamp-level annotations**. True-positive rates, event precision, and recall cannot yet be calculated.
2. The detector uses short-time energy and temporal grouping; it **cannot distinguish** snoring from an equally loud environmental transient without additional validation.
3. Amplitudes are digital sample values, not calibrated sound pressure level. Relative dB on the spectrogram should never be presented as measured room dB SPL.
4. This 60-second clip cannot establish sleep quality, sleep stage, overnight severity, or sleep-disordered breathing.

## What to do next

Manually listen to all 17 highlighted bursts and create timestamp annotations in a CSV. Then calculate event matching precision and recall at a declared temporal tolerance, and compare several adaptive thresholds. A separate baseline that uses spectral shape and autocorrelation may then be evaluated against that labeled reference.
