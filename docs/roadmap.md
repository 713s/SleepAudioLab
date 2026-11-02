# Roadmap

SleepAudioLab is both a working tool and a learning project. Each version should add one clearly testable capability and one corresponding mathematical topic.

## v0.1 — Sampling and basic inspection

Engineering:

- read WAV/FLAC audio;
- inspect sample rate, duration, channels, peak amplitude, and RMS;
- convert multi-channel recordings to mono for baseline analysis;
- draw waveform;
- compute a basic STFT spectrogram.

Mathematics:

- discrete-time signals;
- sampling frequency and sample interval;
- amplitude normalization;
- RMS as an energy-like statistic.

## v0.2 — Fourier spectrum

Engineering:

- implement a direct DFT for short toy signals;
- compare DFT with `numpy.fft.rfft`;
- draw amplitude and power spectra;
- add simple synthetic tests.

Mathematics:

- complex exponentials;
- orthogonality;
- Fourier coefficients;
- Parseval identity;
- Nyquist frequency and aliasing.

## v0.3 — Time-frequency analysis

Engineering:

- expose STFT parameters;
- compare window functions;
- support long recordings by block/chunk processing;
- generate an overnight overview spectrogram.

Mathematics:

- windowing;
- spectral leakage;
- time-frequency resolution trade-off;
- convolution viewpoint.

## v0.4 — Acoustic features

- RMS;
- zero-crossing rate;
- spectral centroid/bandwidth/rolloff;
- spectral entropy;
- robust summary statistics.

## v0.5 — Generic event candidates

- envelope extraction;
- adaptive thresholds;
- event merging;
- change detection;
- candidate clip export.

## v0.6 — Rule-based snore detection

- periodicity;
- autocorrelation;
- dominant-frequency features;
- interpretable rule baseline.

## v0.7 — Classical machine learning

- MFCC;
- manual labels;
- train/test separation;
- logistic regression and SVM;
- precision, recall, ROC/AUC.

## v0.8 — Small neural model

- log-Mel spectrograms;
- convolutional neural networks;
- compact inference suitable for eventual mobile deployment.

## v0.9 — Temporal patterns

- quiet intervals;
- gasp-like recovery events;
- finite-state model first;
- HMM/temporal models only if justified.

## v1.0 — Reproducible overnight report

Target outputs:

- overnight spectrogram;
- snore event table;
- hourly snoring duration;
- candidate long-quiet intervals;
- candidate gasp/recovery clips;
- parameters and model version recorded in report metadata.
