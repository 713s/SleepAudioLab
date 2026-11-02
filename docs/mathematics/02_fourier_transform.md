# 02. Fourier transform for sampled audio

This chapter develops the discrete Fourier transform (DFT) from first principles and connects it to audio analysis.

## 1. From sampled audio to a vector

A sampled audio segment is a finite sequence

\[
x_0,x_1,\ldots,x_{N-1}.
\]

If the sampling rate is \(f_s\) Hz, then sample \(n\) corresponds to time

\[
t_n=\frac{n}{f_s}.
\]

We regard the whole segment as a vector \(x\in\mathbb C^N\), even though real audio normally has real-valued samples.

## 2. Complex exponentials as a basis

For each integer frequency index \(k=0,\ldots,N-1\), define

\[
\phi_k[n]=e^{2\pi i kn/N}.
\]

The vectors \(\phi_k\) are mutually orthogonal in \(\mathbb C^N\) under the inner product

\[
\langle x,y\rangle=\sum_{n=0}^{N-1}x_n\overline{y_n}.
\]

Indeed,

\[
\sum_{n=0}^{N-1}e^{2\pi i(k-\ell)n/N}=
\begin{cases}
N,&k=\ell,\\
0,&k\neq \ell.
\end{cases}
\]

This is the finite-dimensional linear-algebraic reason Fourier analysis works: the DFT computes coordinates of the signal in an orthogonal basis of oscillatory vectors.

## 3. The DFT

We use the convention

\[
X_k=\sum_{n=0}^{N-1}x_n e^{-2\pi i kn/N},\qquad k=0,\ldots,N-1.
\]

The inverse transform is

\[
x_n=\frac1N\sum_{k=0}^{N-1}X_k e^{2\pi i kn/N}.
\]

Thus the transform matrix \(F\) has entries

\[
F_{kn}=e^{-2\pi i kn/N},
\]

and satisfies

\[
F^*F=NI.
\]

Equivalently,

\[
F^{-1}=\frac1N F^*.
\]

This is a useful place to interpret the conjugate transpose as the adjoint operator with respect to the standard Hermitian inner product.

## 4. Frequency bins

The frequency corresponding to bin \(k\) is

\[
f_k=\frac{k}{N}f_s.
\]

The frequency resolution is therefore

\[
\Delta f=\frac{f_s}{N}.
\]

Longer observation windows improve frequency resolution.

For a real-valued signal, the DFT has conjugate symmetry:

\[
X_{N-k}=\overline{X_k}.
\]

Therefore only nonnegative frequencies are unique. `numpy.fft.rfft` exploits exactly this fact.

## 5. A pure sinusoid

Suppose

\[
x_n=A\cos\left(2\pi f_0\frac{n}{f_s}+\varphi\right).
\]

Using Euler's formula,

\[
\cos\theta=\frac{e^{i\theta}+e^{-i\theta}}2,
\]

so the real sinusoid contains one positive-frequency and one negative-frequency complex exponential.

If \(f_0\) lies exactly on a DFT bin,

\[
f_0=m\frac{f_s}{N},
\]

its energy is concentrated at the corresponding Fourier bins. If not, energy spreads across neighboring bins: spectral leakage.

## 6. Spectral leakage

The DFT does not observe an infinite signal. It observes a finite segment. Multiplying an ideal infinite-duration signal by a finite rectangular window in time corresponds to convolution in frequency.

This is why a sinusoid whose frequency is not exactly aligned with a Fourier bin produces a broadened spectrum.

Later, for STFT, we will replace the rectangular window with windows such as the Hann window to control sidelobes and leakage.

## 7. Nyquist frequency and aliasing

For sampling rate \(f_s\), the Nyquist frequency is

\[
f_N=\frac{f_s}{2}.
\]

A continuous-time sinusoid with frequency above \(f_s/2\) is indistinguishable after sampling from a lower-frequency sinusoid.

For example, with \(f_s=1000\) Hz, a 700 Hz sinusoid has samples identical up to phase/sign conventions to a 300 Hz alias because

\[
700=1000-300.
\]

Thus sampled data alone cannot recover frequencies above Nyquist without prior anti-alias filtering assumptions.

## 8. Parseval identity

With the DFT convention used above,

\[
\sum_{n=0}^{N-1}|x_n|^2
=
\frac1N\sum_{k=0}^{N-1}|X_k|^2.
\]

This is Parseval's identity. It says that total signal energy is preserved by the Fourier transform, up to the normalization factor determined by convention.

This is one of the most important bridges between time-domain and frequency-domain analysis.

## 9. Why FFT matters

The direct DFT evaluates \(N\) sums of length \(N\), giving computational complexity

\[
O(N^2).
\]

The fast Fourier transform (FFT) computes the same mathematical transform by exploiting algebraic structure, reducing complexity to approximately

\[
O(N\log N).
\]

The FFT is not a different transform. It is an efficient algorithm for evaluating the DFT.

## 10. Experiments in this repository

Run:

```bash
python -m sleep_audio.fourier_demo --out reports/fourier_demo
```

The demo generates:

- `bin_aligned_sine.png` — a sinusoid exactly on a DFT bin;
- `spectral_leakage.png` — a sinusoid between bins;
- `aliasing.png` — two continuous frequencies producing the same sampled frequency;
- `parseval.json` — numerical verification of Parseval's identity.

The accompanying implementation in `src/sleep_audio/fourier.py` contains both a direct \(O(N^2)\) DFT and NumPy FFT-based helpers so their outputs can be compared directly.
