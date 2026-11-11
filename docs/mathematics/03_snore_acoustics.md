# From a real recording to interpretable acoustic events

This experiment uses `data/samples/snoring_cc0.ogg` (CC0; details in `data/samples/README.md`).

## 1. Time-domain waveform and short-time RMS

The samples `x[n]` represent microphone amplitude, not calibrated sound pressure. The short-time root mean square is

`RMS_m = sqrt((1/L) sum_{n=0}^{L-1} x[mH+n]^2)`.

Use a 200 ms frame and 100 ms hop. A high RMS indicates a louder recorded signal, but this can be speech, movement or environmental sound as well as snoring.

## 2. Fourier power and time-frequency structure

For a Hann-windowed frame, compute the DFT `X_m[k]`. The unnormalized bin power is `P_m[k] = |X_m[k]|^2`. An STFT spectrogram displays this quantity versus time and frequency. The figure in this project uses decibels relative to the strongest plotted power value, **not calibrated dB SPL**.

Spectral centroid: `sum(f[k] P[k]) / sum(P[k])`.

Low-frequency power ratio (below 1000 Hz): `sum(P[k] for f[k] <= 1000) / sum(P[k])`.

Spectral flatness: geometric mean of `P[k]` divided by its arithmetic mean. It distinguishes broad-band, noise-like power from concentrated tonal power, but cannot identify snoring by itself.

## 3. Periodicity via the autocorrelation function

Use the finite-sample autocorrelation `R[l] = sum_n (x[n] - mean(x)) (x[n+l] - mean(x))`. Its local peaks can describe approximately periodic oscillations. If a peak occurs at lag `l`, an associated approximate frequency is `f_s/l`. The implementation restricts frequency to 50-500 Hz and uses an FFT for autocorrelation computation.

**Caution:** a stable tone from room ventilation can also produce a very strong autocorrelation peak. The estimated frequency is an *acoustic periodicity*, not an estimated breathing rate or necessarily a vocal-fold fundamental frequency.

## 4. Candidate events

Estimate an adaptive RMS threshold from the 20th and 90th RMS percentiles and median absolute deviation (MAD). Mark above-threshold frames; join bursts separated by at most 350 ms; retain events lasting at least 250 ms. Separately list quiet recordings spanning at least 3 seconds.

The exact thresholds are heuristic and can be tuned once representative manual annotations exist. In particular, **silence in a microphone recording does not imply physiological apnea**. The output event labels are therefore `candidate_acoustic_burst` and `quiet_interval`, not confirmed snore/apnea labels.

## 5. Reproduction

Run from the project root:

```bash
pip install -e '.[dev]'
python -m sleep_audio.snore_analysis data/samples/snoring_cc0.ogg --out reports/public_snore
pytest -q
```

Outputs: `analysis.png`, `frame_features.csv`, `events.csv`, `quiet_intervals.csv`, `summary.json`.

The project also saves a **public sample report** under `examples/public_snore_analysis/` via GitHub Actions, without including private recordings.

## 6. Next experiments

- Manually annotate several audible bursts, then calculate precision, recall and F1 for candidate detection.
- Compare 100 ms vs 200 ms vs 500 ms frames, and vary the activity threshold.
- Inspect differences between background hum and snore-like energy envelopes.
- Extend to longer recordings with streaming/chunked STFT and hourly histograms.
