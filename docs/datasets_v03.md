# v0.3: labeled audio datasets, provenance and evaluation

**Research scope:** binary clip-level snore versus non-snore classification using
MFCC, spectrum descriptors, autocorrelation, SVM and random forest. There is no
deep learning in this milestone.

## Dataset audit (2026-11-11)

| Dataset | Type of label | Amount | Source | Access and restrictions | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Khan Snoring | Clip-level 0/1: 500 snoring + 500 non-snoring, 1 s each | 1000 clips, 16 kHz | https://www.kaggle.com/datasets/tareqkhanemu/snoring | Public download, but several original sounds come from third-party stock-effects sites/YouTube; **redistribution clearance is not established** | Primary exploratory binary baseline; keep raw files outside Git |
| ESC-50 | 50 sound categories, 2000 clips of 5 s, original 5-fold splits | 2000 clips | https://github.com/karolpiczak/ESC-50 | CC BY-NC 3.0 (audio); verify all license terms | Background hard negatives only, NOT labeled snoring positives; beware confounding |
| ICSD (infant cry/snoring) | Mix of strong event annotations and weak labels | >4 h | https://huggingface.co/datasets/QingyuLiu1/ICSD | Gated, explicitly **no redistribution** without permission, non-commercial | Later event-detection benchmark, not a freely mirrored training set |
| APSAA | Synchronized overnight audio, polygraph signals and manual polygraph event annotations | 32 subjects | https://zenodo.org/records/14096541 | Academic/noncommercial with **no redistribution** | Later physiologically grounded work; annotations are not automatically snore event boundaries |
| MPSSC | Snore event class denotes anatomical excitation site (V/O/T/E), not non-snore | 828 snore events from 219 subjects | https://pubmed.ncbi.nlm.nih.gov/29407995/ | Data access and redistribution need verification | Later multiclass research, not the initial snore/non-snore task |
| FSD50K | Human-verified classes incl. breathing and other sound events; no verified dedicated snoring class in the 200-class release | 51,197 clips | https://fsannotator.upf.edu/fsd/release/FSD50K/ | Per-clip Creative Commons licenses | Negative classes or transfer-study exploration, **not** a clean binary snore dataset |

Do **not** assume dataset metadata or platform visibility authorizes republishing
third-party audio. The full Khan data is never committed to the public repo.

### Khan provenance warnings

Source inventory: https://github.com/minhdenthedev/snoring_detection/blob/master/raw_data/Snoring_dataset.txt

The 500 snore clips are not necessarily 500 independent sleepers or original
recordings. Some sounds could be segments from the same recording or sound-effects
track. A random clip split is an **exploratory**, possibly optimistic estimate.
Do not describe it as unseen-subject generalization.

## Running the baseline

Acquire the Khan dataset separately from the original Kaggle host, check its
terms, and locate the folders `0/` (negative) and `1/` (positive).

```bash
pip install -e '.[dev]'
python scripts/import_khan_dataset.py /path/to/snoring-dataset --out data/processed/khan_manifest.csv
python -m sleep_audio.classical_classifier data/processed/khan_manifest.csv \
    --out reports/v03 --allow-clip-split
```

This explicit flag acknowledges that true independence cannot be guaranteed.
Output: `reports/v03/evaluation.json` with class-1 precision, recall, F1 and
confusion matrices for SVM and random forest.

When original source IDs / subjects become available, provide
`filename,source_id` CSV to `--source-map`, re-import and run **without**
`--allow-clip-split`. Splitting by source avoids recording-level leakage,
but the split still needs a manual audit for duplicates and class balance.

## Features

- Frame window: 25 ms Hann, 10 ms hop, 16 kHz mono, 512-point RFFT
- Time domain: RMS log mean and standard deviation
- Spectrum: mean/std centroid, low-frequency ratio (<=1 kHz), flatness
- Waveform autocorrelation peak in 50-500 Hz (not respiratory frequency)
- MFCC: 26 triangular HTK-mel filters, log power, DCT-II; first 13
  coefficients, mean/std = 26 features
- In total: 33 features per 1-second clip
- SVM: standardize on training set only, RBF kernel
- RandomForest: 200 trees, balanced class weighting

For class `snore=1`, precision = TP/(TP+FP), recall = TP/(TP+FN),
F1 = 2PR/(P+R). Report the test-set support and confusion matrix alongside
the scores. Data labels are *clip-level*, not time-of-onset clinical events.

## Next validation requirements

1. Obtain original recording provenance and implement grouped independent
   train/test splits; do not count augmented crops as independent samples.
2. Deduplicate by exact hashes and near-duplicate acoustic fingerprints.
3. Test the trained classifier against unseen, longer overnight recordings.
4. Annotate a **separate** evaluation set for temporal event precision/recall.
5. Estimate confidence intervals by original recording or subject, never by
   treating adjacent frames as independent observations.
