# SleepAudioLab

SleepAudioLab is an offline-first research and learning project for analyzing overnight audio recordings.

The project has two goals:

1. extract interpretable acoustic indicators from sleep recordings; and
2. learn the mathematics and algorithms of digital signal processing, time-frequency analysis, statistics, and audio event detection by implementing them step by step.

## Scope

Initial target outputs:

- overnight waveform and spectrogram;
- snore event list;
- snoring duration by hour;
- candidate long-quiet intervals;
- candidate gasp-like or recovery-breath events;
- CSV/HTML reports suitable for manual review.

This project is **not a medical diagnostic device**. Acoustic events can be useful for self-observation and screening, but microphone recordings alone cannot diagnose obstructive sleep apnea or determine clinical sleep stages.

## Hardware workflow

```text
Offline Sony phone
      |
      | overnight recording
      v
WAV / FLAC / M4A
      |
      | USB transfer
      v
Linux Mint laptop
      |
      +-- signal processing
      +-- visualization
      +-- event detection
      +-- statistics / ML
      `-- report generation
```

For the first stages, the phone is only a recorder. Analysis is performed on the laptop. On-device inference can be explored later with a compact model.

## Learning roadmap

| Version | Engineering goal | Mathematics / algorithms |
| --- | --- | --- |
| v0.1 | Read audio, inspect metadata, draw waveform | sampling, discrete-time signals |
| **v0.2** | **DFT/FFT experiments and power-spectrum foundations** | **Fourier basis, leakage, Nyquist, aliasing, Parseval** |
| v0.3 | STFT and overnight spectrogram | windowing, time-frequency resolution |
| v0.4 | RMS/ZCR/spectral features | descriptive and robust statistics |
| v0.5 | Generic sound-event candidates | thresholding, envelope, change detection |
| v0.6 | Rule-based snore detector | autocorrelation, periodicity |
| v0.7 | MFCC + classical classifier | feature spaces, logistic regression/SVM |
| v0.8 | Small CNN snore detector | convolution, learned representations |
| v0.9 | quiet -> gasp -> recovery sequence | time-series / state models |
| v1.0 | reproducible overnight report | end-to-end pipeline |

## Project structure

```text
SleepAudioLab/
├── README.md
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── roadmap.md
│   ├── medical_limits.md
│   └── mathematics/
├── src/
│   └── sleep_audio/
├── tests/
├── data/
│   └── README.md
└── reports/
```

Raw personal sleep recordings should remain local and are excluded from Git by default.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run tests:

```bash
pytest
```

## v0.1 — inspect an audio recording

```bash
sleep-audio inspect path/to/test.wav --out reports/test
```

This produces a waveform, an STFT spectrogram, and a small JSON summary.

## v0.2 — learn Fourier analysis from the definition

The v0.2 code contains a direct \(O(N^2)\) implementation of the DFT,

\[
X_k=\sum_{n=0}^{N-1}x_n e^{-2\pi i kn/N},
\]

plus an inverse DFT and NumPy FFT comparison tests. The notes are in:

```text
docs/mathematics/02_fourier_transform.md
```

Generate the experiments:

```bash
python -m sleep_audio.fourier_demo --out reports/fourier_demo
```

Outputs:

```text
reports/fourier_demo/
├── bin_aligned_sine.png
├── spectral_leakage.png
├── aliasing.png
└── parseval.json
```

The experiments are designed to answer four concrete questions:

1. Why does a sinusoid on an exact DFT bin produce a sharp spectral line?
2. Why does a slightly off-bin sinusoid leak into neighboring frequencies?
3. Why can a sampled 700 Hz sinusoid at 1000 Hz sampling appear as 300 Hz?
4. Why does Parseval's identity give the same total energy in time and frequency domains?

## Development principles

- Prefer interpretable signal-processing baselines before deep learning.
- Implement important algorithms from first principles for learning, then compare with SciPy/librosa.
- Keep raw recordings and derived private clips out of the public repository.
- Separate measurement, preprocessing, inference, and reporting.
- Record assumptions, thresholds, and validation procedures explicitly.

## License

A license has not yet been selected. Until one is added, the repository is publicly viewable but no open-source license is granted automatically.
