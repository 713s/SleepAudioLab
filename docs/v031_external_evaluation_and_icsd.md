# v0.3.1 — Frozen models, external validation and restricted ICSD data

## What was implemented

Training can now persist two fitted estimators, not just report scores:

```bash
python -m sleep_audio.classical_classifier data/processed/khan_manifest.csv \
  --out models/khan-v031 --allow-clip-split --save-models --dataset-id khan-snoring
```

Output in the **local** `models/khan-v031/` directory:
- `svm_rbf.joblib`: the entire fitted StandardScaler + RBF SVC pipeline.
- `random_forest.joblib`: the fitted random forest.
- `model_metadata.json`: sample rate, exactly one-second input, feature schema/dimension, class labels, training dataset name, source-split type, sklearn version, and SHA-256 of *training* audio files (no raw audio).
- `evaluation.json`: baseline internal metrics.

The GitHub Actions workflow also trains models and uploads these four files as a **30-day run artifact**, but does **not commit models or training audio** to the public Git tree. Model artifact link is available from the workflow's latest successful run.

For external, separately labeled one-second clips, create a local CSV manifest (`path,label,source_id`) with label 1 = snore, 0 = non-snore and run:

```bash
python -m sleep_audio.evaluate_external data/processed/external_manifest.csv \
  --models models/khan-v031 --dataset-id icsd-real-test \
  --out reports/icsd-v031
```

This evaluator performs only `joblib.load`, `predict`, and optional decision scores. **No model fitting or threshold tuning occurs** during evaluation. It rejects the identical dataset ID and exact-byte duplicates of training files. It reports precision, recall, F1, average precision, ROC-AUC, and confusion matrices. **Only load joblib files from a trusted run**: deserializing arbitrary joblib/pickle is unsafe. Exact-file SHA-256 matching does **not** detect near-duplicate recodings or overlapping excerpts.

At present, the evaluator expects binary **clip-level labels for one-second segments**. ICSD is an event-detection dataset: its strong labels are intervals, often in longer recordings, and its weak labels do not provide boundaries. Correct mapping from events to clip labels, including negative windows, partial overlaps and ambiguous boundaries, requires a separate adapter and defined policy. **Do not point this evaluator directly at raw ICSD metadata and pretend the labels match.**

## ICSD license & access audit (2026-11-11)

Authoritative original sources:
- Dataset terms: https://huggingface.co/datasets/QingyuLiu1/ICSD
- Authors' instructions: https://github.com/QingyuLiu0521/ICSD

The dataset page has a `cc-by-nc-sa-4.0` license tag **AND additional access terms**. The actual access agreement explicitly requires:
1. non-commercial research/education;
2. **no redistribution of dataset data, in whole or part, without prior written provider consent**;
3. derivatives/modifications used only for permitted purposes and **not shared or distributed**;
4. acknowledgment of the dataset and providers in publications and presentations.

The dataset authors also state that copyright in the underlying audio remains with its original owners. The license tag does **not** override the explicit no-redistribution condition; do not treat it as blanket authorization to publish files.

### Keep local/private or on your authorized secure compute environment

| Material | Public GitHub? | Safe project location |
| --- | --- | --- |
| `Dataset.zip`, `Materials.zip`, original ICSD WAV/FLAC/MP3 audio | **No** without explicit written permission | `data/icsd/` |
| ICSD strong label time-stamps, weak label CSV/TSV, official metadata | **No**: dataset content cannot be mirrored | `data/icsd/` or `data/labels/` |
| Cropped clips, resampled WAV, synthesized mixtures, spectrogram images made from ICSD | **No**: derivatives/modified content restricted by access terms | `data/processed/` |
| Per-clip MFCC/embeddings/features, file-level predictions and per-recording identifying metadata | **Keep private**: may be restricted derivatives or enable reconstruction/linkage | `data/processed/` or `reports/` |
| Models trained/fine-tuned on ICSD | **Keep private unless separately cleared**; sharing may count as restricted derivative | `models/icsd/` |
| Your Hugging Face token, Kaggle credentials, .env files | **Never**, independent of dataset licensing | environment variables or GitHub Actions secrets |
| ICSD train/validation/test manifests with private file paths | **No** | `data/processed/` |
| Independently written code, algorithms, empty manifest templates and public source URL/license note | **Yes** | `src/`, `scripts/`, `docs/` |
| High-level aggregated benchmark scores from authorized research | **Potentially yes** as research results with attribution, subject to agreement and no release of dataset-derived records or reconstructible content | `docs/` after checking terms |

If you obtain specific written redistribution consent, file-by-file rights may differ. Until then, use the restrictive default.

### Obtaining ICSD and handling labels

1. Request access using your Hugging Face account at the dataset page and agree to its terms; do not bypass access controls.
2. Download on your own authorized machine using the authors' script and a token stored in a private environment variable, or follow the official instructions.
3. Preserve original metadata and train/validation/test split locally. **Do not use test recordings for training, feature selection, or threshold calibration.**
4. For **strong** event labels (start/end timestamps), use a documented 1-second overlap policy, e.g., positive if snoring covers >=50% of the window, negative if no relevant annotated sound intersects the window, and **exclude** partial/ambiguous cases. This is a *proposed* protocol, not a claim about the data's original label semantics.
5. Weak clip labels cannot be converted to true event onset/offset labels. Do not generate fake timestamps.
6. Check source-recording overlap with Khan using hashes and acoustic fingerprints before calling the experiment truly independent.
7. Keep raw data and derivative model outputs private. Publish only permissible aggregate evaluation and cite Liu et al., *ICSD: An Open-source Dataset for Infant Cry and Snoring Detection*, arXiv:2408.10561.

## Security and reproducibility

- Private audio is ignored by `.gitignore`, but `git add -f` can override ignores. Do not force-add it.
- The app needs Python 3.10+, and matching scikit-learn/numpy versions are advisable when loading joblib artifacts.
- The model currently learns only clip classification, **not** breathing cessation, apnea, sleep stage or medical severity.
- A dataset-independent evaluation has **not yet been performed** until a separate labeled dataset is imported through a verified adapter; synthetic unit tests and Khan internal testing are not sufficient.
