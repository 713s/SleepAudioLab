# Overnight audio analysis: first usable application

The **initial project goal** is an offline tool that extracts interpretable
acoustic indicators and an auditable event timeline from personal recordings.
Rough path theory is a later research direction, **not a prerequisite**.

## Run on Linux Mint or your authorized private server

```bash
cd /opt/sleep-audio-lab   # adjust to your actual checkout
git pull
source .venv/bin/activate
pip install -e '.[dev]'

sleep-audio overnight /path/to/personal-night.wav --out reports/my-night
```

Supported directly through libsndfile: WAV, FLAC and OGG where available.
For M4A/MP3 that your soundfile installation cannot decode, convert locally:

```bash
ffmpeg -i recording.m4a -vn -c:a pcm_s16le recording.wav
sleep-audio overnight recording.wav --out reports/my-night
```

Output:

- `report.html`: open locally in a browser (no hosted service)
- `overview.png`: RMS activity and compressed frequency overview
- `summary.json`: duration, recording format and detector assumptions
- `frame_features.csv`: 1-second RMS, peak amplitude, spectral centroid,
  low-frequency power ratio (0–1000 Hz), spectral flatness
- `events.csv`: candidate acoustic bursts with onset/offset and descriptors
- `quiet_intervals.csv`: intervals with low recorded energy
- `hourly.csv`: per-hour activity seconds, quiet seconds and count of
  candidate bursts *starting* in that hour

Times are **seconds from recording start**, not wall-clock time.
If a detected event crosses an hour boundary, activity seconds are split between
hours while event count is assigned to its **starting** hour.

## Mathematics and algorithm choices

1. For each 1-second nonoverlapping frame, calculate RMS
   `sqrt(mean(x**2))` and the peak magnitude.
2. Properly polyphase-resample the frame to 8 kHz and apply a Hann window.
   The periodogram supplies spectral centroid, 0–1 kHz power ratio,
   spectral flatness, and a compressed spectral overview.
3. Compute the full-recording adaptive energy threshold using the
   20th and 90th RMS percentiles plus median absolute deviation.
4. Group above-threshold one-second frames into candidate activity intervals.
   A user-configurable gap may bridge brief interruptions.
5. Group consecutive below-threshold frames into long quiet intervals.

The detector works globally across the recording, not by normalizing every
block separately. It streams audio and stores per-second feature vectors
in memory (not audio samples). The visualization aggregates many seconds per
pixel to keep output manageable for overnight recordings.

## Tuning

```bash
sleep-audio overnight night.wav --out reports/night2 \
  --event-min 2 --merge-gap 1 --quiet-min 10
```

Defaults: candidate minimum 2 seconds, gap merge <=1 second,
quiet minimum 10 seconds. Avoid calling these parameters clinically validated.

## Important limitations

- **Candidate acoustic activity is not confirmed snoring.** A fan, cough,
  speech or another sound can be detected. Conversely soft snores can be missed.
- **Quiet sound does not imply cessation of breathing or sleep apnea.**
  No claims about oxygen saturation, airflow, sleep stages or sleep disorders.
- This first version does **not** classify gasp or recovery breathing; those
  remain future research tasks. A trained ICSD model is not yet connected
  to this overnight entrypoint. The CSV `candidate_activity_seconds` is
  not yet a validated *snoring duration* estimate.
- If source recording level changes substantially during the night,
  one global energy threshold may be too coarse; review before interpreting.
- The 1-second windows have 1-second boundary uncertainty and may merge
  separate events. This is a coarse initial working implementation.
- The low-frequency ratio is a descriptive statistic, not a unique snore marker.
- All personal audio and reports remain local. Do not publish identifiable
  recordings or ICSD derivatives.

## Next practical milestone

Integrate the locally trained ICSD classifier as a *selectable* detector
with matching one-second audio and feature contracts; then compare outputs
against manually labeled portions of the user's own recordings. Only after
that should the report claim model-estimated `snore_candidate_seconds`.

## Optional frozen ICSD SVM for snore candidates

After training the ICSD strong-label SVM locally, run:

    sleep-audio overnight /path/to/night.wav --out reports/my-night-model --model-dir models/icsd-strong

This adds model_frame_predictions.csv (one-second labels), model_snore_candidates.csv (candidate event intervals) and model_hourly.csv (candidate seconds per hour). The heuristic acoustic activity output stays separate.

The model is not clinically validated and may not generalize to your microphone or sleeping environment. Review samples manually before treating the outputs as snoring estimates. Never treat quiet audio as proven apnea, or acoustic activity as confirmed snoring. Only load your OWN trusted local joblib models: untrusted joblib/pickle files may execute arbitrary code. Model inference is optional; the original signal-analysis report works without any classifier.
