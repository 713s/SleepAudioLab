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

## Real snore acoustics experiment (public CC0 sample)

The repository includes `data/samples/snoring_cc0.ogg`, a freely reusable snoring recording from Wikimedia Commons. The third mathematics lesson is [Snore acoustics](docs/mathematics/03_snore_acoustics.md).

To compute the waveform/spectrogram, RMS, spectral centroid, low-frequency power ratio, spectral flatness, autocorrelation periodicity, and **candidate** acoustic events:

```bash
python -m sleep_audio.snore_analysis data/samples/snoring_cc0.ogg --out reports/public_snore
```

Outputs include `analysis.png`, `summary.json`, `frame_features.csv`, `events.csv`, and `quiet_intervals.csv`. An automatically generated reproducible report for this public recording is available in [examples/public_snore_analysis](examples/public_snore_analysis) after the corresponding GitHub Action finishes.

**Important:** these are heuristic, unlabeled sound candidates. High energy does not prove snoring, silence does not prove apnea, and the audio is not a validated medical benchmark.

## v0.3 — classical snore classification with labeled data

We use a dedicated public clip-level snoring dataset for the **training experiment**, not the 17 acoustic bursts in the initial one-minute recording. A reproducible, **non-deep-learning** baseline compares a 33-feature vector (RMS, spectral descriptors, autocorrelation and 13 MFCC means/standard deviations) using SVM and random forest. The evaluation reports precision, recall, F1 and confusion matrices.

- [Labeled dataset license/provenance audit](docs/datasets_v03.md)
- [Classifier implementation](src/sleep_audio/classical_classifier.py)
- [Khan dataset manifest importer](scripts/import_khan_dataset.py)
- [Automated baseline with ephemeral Kaggle download](.github/workflows/v03-classical-baseline.yml)

Training audio is **not mirrored or committed**: it is downloaded from its source only into the temporary GitHub Actions runner. Actions uploads `evaluation.json` as a short-lived artifact. Original recording/subject IDs are unavailable for the default Khan clips, so the exploratory random clip split **must not** be reported as a trustworthy subject-independent test. To use a source-disjoint test, supply verified original recording IDs through the manifest's `source_id` column.

## v0.3.1 — frozen models and external evaluation (no retraining)

The Kaggle/Khan SVM and random forest can be saved as complete fitted `joblib` models with the exact feature contract and train-file checksums. For a separate, labeled 1-second-audio manifest, run the **inference-only** evaluator:

```bash
python -m sleep_audio.evaluate_external data/processed/external_manifest.csv \
  --models models/khan-v031 --dataset-id independent-dataset-name \
  --out reports/external
```

[Full guide to frozen models, independent evaluation and ICSD access restrictions](docs/v031_external_evaluation_and_icsd.md). You may download the successfully trained frozen model archive from [GitHub Actions](https://github.com/713s/SleepAudioLab/actions/workflows/v03-classical-baseline.yml), under **Artifacts** in a successful run. Model files and restricted training audio are not committed to the public Git tree. Only load trusted model artifacts: `joblib` uses pickle under the hood.

**ICSD caution:** although ICSD is labeled, its actual access agreement prohibits unauthorized redistribution of originals **and derivatives**. Keep local audio, event labels, clips, feature files, and ICSD-derived models off public GitHub unless permission is secured.

## v0.3.2 — gated ICSD test adapter

[ICSD real-strong test adapter guide](docs/v032_icsd_test_adapter.md) documents how to convert **authorized, privately stored** strong test annotations into one-second audio clips and a `path,label,source_id` manifest for the v0.3.1 frozen-model evaluator. The code is [`src/sleep_audio/icsd_adapter.py`](src/sleep_audio/icsd_adapter.py). The public GitHub repo contains **only adapter source code, documentation and synthetic tests**, never gated ICSD audio or annotations.

```bash
python -m sleep_audio.icsd_adapter data/icsd/Dataset \
  --out data/processed/icsd/test --assume-exhaustive
python -m sleep_audio.evaluate_external \
  data/processed/icsd/test/test_manifest.csv \
  --models models/khan-v031 --dataset-id icsd-real-strong-test \
  --out reports/icsd-test-v031
```

The assumption of exhaustive strong annotations must be checked before enabling negative-label creation. Test-set isolation from ICSD training/validation is enforced by fixed paths and a byte-identical recording overlap audit, but the authors cite **Khan Snoring among their source collections**. A further source-level fingerprint audit is required before calling the evaluation independent of Khan training.
