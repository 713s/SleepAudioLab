# ICSD local CPU training (v0.3.3)

Use **only on the computer/server where you have legally downloaded ICSD**.
No dataset files, derived clips, labels, feature matrices, or ICSD-trained models
may be copied into the public Git tree under the dataset access agreement.

## Verified official strong-label directory names

The [upstream ICSD configuration](https://github.com/QingyuLiu0521/ICSD/blob/main/ICSD_task/confs/default.yaml) defines:

| Role | Audio | Strong label TSV |
| --- | --- | --- |
| Training, real | `audio/train/real_strong_train/` | `metadata/train/real_strong_train.tsv` |
| Training, synthetic | `audio/train/synth_strong_train/` | `metadata/train/synth_strong_train.tsv` |
| Validation, real | `audio/validation/strong_val/` | `metadata/validation/strong_val.tsv` |
| Test (held out) | `audio/test/real_strong_test/` | `metadata/test/real_strong_test.tsv` |
| Weak training | `audio/train/weak/` | `metadata/train/weak.tsv` |

The label vocabulary is `Snoring` and `Infantcry`. Strong TSVs use
`filename`, `onset`, `offset`, `event_label`.

The current adapter intentionally handles **strong labels only**.
Weak labels specify clip-level presence, **not event onset or every one-second window**;
a later multiple-instance learning or weak-supervision method is needed.
Never fabricate weak-label timestamps.

## 1. Verify local dataset

From the project root on your authorized machine:

```bash
test -f data/icsd/Dataset/metadata/train/real_strong_train.tsv
test -f data/icsd/Dataset/metadata/train/synth_strong_train.tsv
test -f data/icsd/Dataset/metadata/validation/strong_val.tsv
```

Check actual extraction location if one test fails. **Do not upload ICSD TSVs
for diagnosis**; share only directory names, counts, or sanitized error messages.

## 2. Build private 1-second training and validation clips

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python -m sleep_audio.icsd_training_adapter data/icsd/Dataset \
  --out data/processed/icsd/strong-v1 \
  --assume-exhaustive
```

**Critical assumption:** use `--assume-exhaustive` only if you judge the
strong-event annotation convention to mark every target event. Without it,
unannotated windows are not labeled negative, so binary training is not
possible. The script does not assign any label to 1-second windows with
partial snore overlap (0 < coverage < 0.5 s), infant-cry overlap, or a truncated tail.

For a lower-cost first experiment use `--no-synthetic` to exclude synthetic
training audio; validation remains independent.

The adapter decodes one source recording at a time and never reads ICSD official
test. It writes private files:

```text
data/processed/icsd/strong-v1/
  train_manifest.csv
  validation_manifest.csv
  audit.json
  clips/train/*.wav
  clips/validation/*.wav
```

Repeated windows from one recording retain a common source ID.
The converter checks exact original-audio hashes across train/validation.
It does not establish independence of underlying subjects or of separately
re-encoded sources.

## 3. Train ICSD-only models; validate without fitting on validation

```bash
python -m sleep_audio.train_icsd \
  --train data/processed/icsd/strong-v1/train_manifest.csv \
  --validation data/processed/icsd/strong-v1/validation_manifest.csv \
  --out models/icsd-strong
```

Saved locally: `svm_rbf.joblib`, `random_forest.joblib`,
`model_metadata.json`, `validation_results.json`.

The feature set is the existing 33-dimensional RMS/spectrum/autocorrelation/
MFCC baseline at 16 kHz with one-second windows. This baseline is for comparison:
the project's research goal can subsequently use rough path signatures and
log-signatures of time-augmented or feature paths.

The program checks manifest `source_split` and `subset` values, rejects
known source/byte overlap, calls `fit` **only on TRAIN**, and evaluates on
validation. Do not interpret validation scores as official test scores.

## 4. Model combination later

Do not average SVM decision_function scores with random-forest probabilities.
A future ensemble can calibrate each model using *validation only* and freeze
ensemble weights before using the official ICSD test once. If data from
Khan appears in ICSD synthetic sources, model combination remains possible
but does not create an independent benchmark.

## Security

All original ICSD audio/labels, derivative windows, per-file predictions,
and ICSD-trained models should stay local per the access agreement.
`.gitignore` covers `data/icsd/`, `data/processed/`, `models/`, and `reports/`.
Avoid `git add -f`, public GitHub Actions artifacts or public cloud uploads
containing ICSD-derived files.
