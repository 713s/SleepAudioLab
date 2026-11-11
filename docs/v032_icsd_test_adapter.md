# v0.3.2 — ICSD real-strong TEST adapter (restricted-data workflow)

This adapter converts the **official ICSD real_strong_test** dataset only into local, labeled 1-second clips compatible with the frozen v0.3.1 binary SVM and random forest.

## Verified official layout

The [ICSD upstream config](https://github.com/QingyuLiu0521/ICSD/blob/main/ICSD_task/confs/default.yaml) names the following paths relative to the extracted `Dataset` root:

```text
Dataset/
├── audio/
│   ├── train/...
│   ├── validation/...
│   └── test/real_strong_test/*.wav
└── metadata/
    └── test/real_strong_test.tsv
```

The original event vocabulary includes `Snoring` and `Infantcry` (see [official classes](https://github.com/QingyuLiu0521/ICSD/blob/main/ICSD_task/local/classes_dict.py)); strong TSV fields are `filename`, `onset`, `offset`, `event_label`. The adapter **does not** interpret weak labels as event timestamps and **does not** mix synthetic test data with real test data.

## Permission prerequisite

[Request ICSD access through the Hugging Face dataset page](https://huggingface.co/datasets/QingyuLiu1/ICSD). Only download after receiving authorized access. Keep the original audio and labels **locally** under `data/icsd/`, which is excluded from the public repository. Do not commit any ICSD source archive, audio, TSV/CSV annotations, clipped audio, derived signal features, or per-sample predictions. ICSD prohibits distribution of all or part of the dataset and its derivatives without permission.

## Conversion

Once you have legally extracted `Dataset.zip` into `data/icsd/Dataset/` (or another private location), run:

```bash
python -m sleep_audio.icsd_adapter data/icsd/Dataset \
  --out data/processed/icsd/test --assume-exhaustive
```

**Important**: `--assume-exhaustive` is an *explicit research assumption* that missing `Snoring` and `Infantcry` intervals in the real strongly annotated test imply absence of those target events. If completeness is unconfirmed, **do not use the flag**, which defaults to a **positive-only** export not suitable for calculating meaningful precision/F1. This prevents silently labeling all unannotated audio as negative. Use the flag only after reviewing the dataset's event annotation protocol.

Mapping on non-overlapping one-second windows:
- Positive (`1`): at least 0.5 seconds overlap with a Snoring event, **and no Infantcry overlap**.
- Negative (`0`): no target event overlap, only if `--assume-exhaustive` is explicitly provided.
- Excluded (ambiguous): any Infantcry overlap; Snoring overlap greater than zero but below 0.5 seconds; truncated window shorter than 1 second.
- No weak labels, synthetic labels, or non-test training partitions are imported.

The adapter reads the official real-strong test path only, refuses unexpected event classes/filenames, validates times against recording duration, checks exact audio file hashes against optional `audio/train` and `audio/validation` trees, and refuses to overwrite an existing test manifest. It neither trains a model nor chooses any threshold using official test predictions. **Byte-level hashes cannot detect recompressed/trimmed same-source audio; manual provenance audits remain necessary.**

Private output:

```text
data/processed/icsd/test/
├── clips/                     # derived 1-second ICSD WAVs (PRIVATE)
├── test_manifest.csv          # path,label,source_id (PRIVATE)
└── audit.json                 # conversion settings and counts (PRIVATE)
```

After the first conversion, freeze the manifest and keep `audit.json` in your local experiment notebook. Do not experiment with thresholds against test-set performance: changing the 0.5-second overlap rule after looking at the test results **invalidates its holdout status**.

## Frozen model external evaluation

Download the saved **Khan-only** models from the successful GitHub Actions `v0.3 labeled snore classifier baseline` run (artifact `v031-khan-frozen-models`). Extract the archive into a local `models/khan-v031/` directory such that the `*.joblib` files and `model_metadata.json` are immediately inside that directory.

```bash
python -m sleep_audio.evaluate_external \
  data/processed/icsd/test/test_manifest.csv \
  --models models/khan-v031 \
  --dataset-id icsd-real-strong-test \
  --out reports/icsd-test-v031
```

`evaluate_external` only loads frozen estimators and calls inference methods; it does not fit/recalibrate models. It reports precision, recall, F1, average precision and confusion matrix. If the conversion produced only one class, the evaluator will reject the manifest instead of fabricating metrics.

## Do not overclaim independence

The ICSD authors list Khan Snoring among source acknowledgments. Therefore even when the official **ICSD test split** is disjoint from **ICSD train/validation**, this does **not** automatically establish source independence from a model trained on the **Khan** dataset. Exact file-byte duplicate checks are inadequate for resampled, compressed, cropped, or remixed versions of a Khan recording.

Before calling results an independent-dataset benchmark, perform source recording matching or acoustic fingerprinting against Khan training audio. If provenance cannot be confirmed, report this as a *cross-dataset evaluation with unresolved source overlap*, not verified external generalization.

Source: [ICSD official repository](https://github.com/QingyuLiu0521/ICSD), [ICSD access agreement](https://huggingface.co/datasets/QingyuLiu1/ICSD).
