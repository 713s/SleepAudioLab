# 01 — Sampling and discrete-time audio

A microphone produces a continuous-time physical signal that is converted into a digital sequence.

If the sampling frequency is

\[
f_s \;\text{samples/second},
\]

then the sampling interval is

\[
\Delta t = \frac{1}{f_s}.
\]

The discrete samples are

\[
x[n] = x(n\Delta t), \qquad n=0,1,\dots,N-1.
\]

For example, if

\[
f_s=16000\ \text{Hz},
\]

then

\[
\Delta t = 62.5\ \mu s.
\]

A recording with `N` samples has duration approximately

\[
T = \frac{N}{f_s}.
\]

## Why sampling rate matters

The Nyquist frequency is

\[
f_N=\frac{f_s}{2}.
\]

Frequencies above this limit cannot be represented uniquely after sampling and may alias into lower frequencies unless they are sufficiently filtered before sampling.

For sleep-audio work, the first practical questions are therefore:

1. What sample rate did the phone recorder use?
2. Is the recording mono or stereo?
3. Does the recorder apply automatic gain control or compression?
4. Is the chosen rate unnecessarily high for the acoustic events we want to study?

## Amplitude normalization

Audio libraries commonly expose floating-point samples in a range close to

\[
[-1,1].
\]

The absolute numerical amplitude is not automatically a calibrated sound-pressure level. Unless the microphone has been calibrated, amplitude should initially be interpreted as a **relative digital amplitude**.

## RMS

For a block of `N` samples, root-mean-square amplitude is

\[
\operatorname{RMS}(x)
=\sqrt{\frac{1}{N}\sum_{n=0}^{N-1}x[n]^2}.
\]

RMS is useful as a compact measure of signal magnitude and later becomes a basic feature for detecting sound activity.

## First experiment

Take a short recording containing:

- 5–10 seconds of room noise;
- normal breathing;
- speech;
- one deliberately loud sound.

Inspect the waveform and compare peak amplitude with RMS. The purpose is to learn what the samples actually look like before using any classifier.
