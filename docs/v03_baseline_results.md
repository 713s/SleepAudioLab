# v0.3 first labeled-data baseline (2026-11-11)

**Status: successfully executed in GitHub Actions.**
- [CI run and full logs](https://github.com/713s/SleepAudioLab/actions/runs/42955674914)
- [Metrics artifact (evaluation.json)](https://github.com/713s/SleepAudioLab/actions/runs/42955674914/artifacts/13826922161)
- Dataset: [Tareq Khan Snoring (Kaggle)](https://www.kaggle.com/datasets/tareqkhanemu/snoring)
- Class 1 = snoring, class 0 = non-snoring.
- Dataset files stay in the ephemeral runner and are **not** mirrored in this GitHub repository.

## Experiment

1,000 labeled 1-second clips successfully downloaded: 500 positive + 500 negative. Clip-level stratified random 75%/25% split with seed 42: training = 750, test = 250 (125 positive and 125 negative test clips).

A fixed 33-dimensional acoustic feature vector is extracted from each 1-second audio:
RMS, spectrum statistics, short-lag autocorrelation and MFCC means/standard deviations. Classifiers: SVM with RBF kernel and random forest with 200 trees.

## Observed metrics: exploratory clip split ONLY

| Model | Precision, snore | Recall, snore | F1, snore | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| SVM (RBF) | 0.9609 | 0.9840 | 0.9723 | 0.9720 |
| Random forest | 0.9615 | 1.0000 | 0.9804 | 0.9800 |

Confusion matrices use row = true label, column = predicted label, with order [non-snore, snore]:

**SVM:** `[[120, 5], [2, 123]]`.

**Random forest:** `[[120, 5], [0, 125]]`.

## Why these are NOT validated generalization results

The data contains 1,000 short *clips* but does not provide verified source recording or subject IDs. Related segments or different excerpts from one sound-effect recording may land in both train and test. Therefore, the reported metrics may be inflated by acoustic similarity, recording artifacts or source overlap.

This is a **pipeline smoke test on real, labeled data**, not a validated estimate of subject-independent performance, real-night snore duration estimation, breathing interruption detection, or clinical sleep quality.

A 98% random-clip accuracy is especially not a license to claim that the detector will accurately recognize snoring in Sony-phone overnight recordings.

## Next experiment for credible evaluation

- Reconstruct original source group IDs where possible, remove duplicates, and use group-disjoint splitting.
- Add independent CC-licensed snoring recordings not appearing in training data.
- Compare normal audio and hard negatives such as breathing, fan hum, rain, cough and speech.
- Annotate time boundaries on a few **separate** longer recordings, and score event-level precision/recall; clip-level F1 does not replace event-level accuracy.
- Where possible, evaluate by recording participant and microphone device.

The source-dataset inventory and access restrictions are documented in [datasets_v03.md](datasets_v03.md).
