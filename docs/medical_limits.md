# Medical and interpretation limits

SleepAudioLab analyzes **acoustic recordings**, not clinical sleep-study signals.

The project may estimate or summarize observable acoustic quantities such as:

- snore-like sound duration;
- sound-event density;
- relative loudness changes;
- long quiet intervals;
- gasp-like or recovery-breath candidates;
- night-to-night trends.

It should not present microphone-only results as:

- a diagnosis of obstructive sleep apnea (OSA);
- a clinical apnea-hypopnea index (AHI);
- EEG-derived sleep stages;
- a medical-grade sleep quality score.

A silent interval in an audio recording is not equivalent to absence of airflow. It may be caused by quiet breathing, body position, microphone placement, room noise, automatic gain control, compression, or other recording effects.

Any future risk score must keep raw/intermediate evidence available for manual review and must clearly state its assumptions and validation data.
