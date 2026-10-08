# Understanding FFT in PyTorch

Read sections 1–6 first. They explain what the numbers mean. The remaining sections cover the `torch.fft` API and its use in numerical quantum mechanics.

## 1. What problem does a Fourier transform solve?

You have samples of a function:

```text
sample position:  0   1   2   3
function value:   1   0  -1   0
```

You can describe this by listing its values. Alternatively, you can describe it as a combination of waves.

Here the values are exactly one cosine cycle sampled at four points:

$$
x_j=\cos(2\pi j/4),\qquad j=0,1,2,3.
$$

A discrete Fourier transform (DFT) changes from **sample values** to **coefficients of waves**. Its inverse adds the waves back together.

FFT means *fast Fourier transform*: an efficient algorithm for computing the DFT. It computes the same mathematical transform as the direct sum, subject to floating-point rounding. Typical FFT algorithms take roughly $N\log N$ work instead of the direct sum's $N^2$.

## 2. The actual waves being used

For four samples, the Fourier basis consists of these four sampled complex waves:

| Wave index | Values at the four sample positions |
|---|---|
| 0 | `[1, 1, 1, 1]` |
| 1 | `[1, i, -1, -i]` |
| 2 | `[1, -1, 1, -1]` |
| 3 | `[1, -i, -1, i]` |

The first is constant. The others rotate in the complex plane.

Our signal is half of wave 1 plus half of wave 3:

```text
0.5 * [1, i, -1, -i] + 0.5 * [1, -i, -1, i]
    = [1, 0, -1, 0]
```

This is the identity

$$
\cos\theta=\frac{e^{i\theta}+e^{-i\theta}}{2}.
$$

**The Fourier coefficients tell us how much of each basis wave to combine.** Complex coefficients also encode phase: where each oscillation starts.

## 3. What `fft` and `ifft` compute

With the default normalization:

$$
X_m=\sum_{j=0}^{N-1}x_j e^{-i2\pi mj/N}
$$

$$
x_j=\frac{1}{N}\sum_{m=0}^{N-1}X_m e^{i2\pi mj/N}
$$

Here `j` indexes samples, `m` indexes waves, and `N` is the sample count. The forward sum measures the match between the data and each wave. Different basis waves cancel when summed against each other.

```python
import torch

x = torch.tensor([1.0, 0.0, -1.0, 0.0], dtype=torch.float64)
X = torch.fft.fft(x)
# tensor([0.+0.j, 2.+0.j, 0.+0.j, 2.+0.j], dtype=torch.complex128)

x_back = torch.fft.ifft(X)
# [1, 0, -1, 0], represented as complex numbers
```

The coefficient is `2` because the default inverse divides by `N=4`. Each wave therefore has weight `2/4 = 0.5` in the reconstructed signal.

The output has `N` complex coefficients. Even real inputs generally need complex coefficients. [PyTorch `fft` reference](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft.html)

## 4. What `fftfreq` adds

`fft(x)` knows the sample values but does not know whether samples are one second, one millisecond, or one meter apart.

`fftfreq(N, d)` supplies the physical frequency labels. It never examines `x`.

For our four samples spaced one second apart:

| Index | FFT coefficient | Frequency in Hz | Reconstruction weight |
|---|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 2 | +0.25 | 0.5 |
| 2 | 0 | −0.50 | 0 |
| 3 | 2 | −0.25 | 0.5 |

```python
f = torch.fft.fftfreq(4, d=1.0)
# tensor([0.00, 0.25, -0.50, -0.25])
```

The pair `(f[m], X[m])` means: **this coefficient belongs to this frequency**.

The frequency labels depend only on `N` and `d`. The function applies the ordering and scaling:

$$
f=\frac{[0,1,\ldots,\lfloor(N-1)/2\rfloor,
-\lfloor N/2\rfloor,\ldots,-1]}{Nd}.
$$

For `N=5`, the integer labels are `[0, 1, 2, -2, -1]`. For `N=8`, they are `[0, 1, 2, 3, -4, -3, -2, -1]`. For even `N`, the Nyquist bin is labeled negative. [PyTorch `fftfreq` reference](https://docs.pytorch.org/docs/main/generated/torch.fft.fftfreq.html)

## 5. Why the frequencies wrap

At integer sample indices `j`,

$$
e^{i2\pi(m+N)j/N}=e^{i2\pi mj/N}.
$$

Adding `N` to a wave index produces exactly the same sampled wave. Therefore index `N-1` and index `-1` are indistinguishable on this grid.

For the four-point example, wave 3 is also wave −1. The labels `[0, 1, -2, -1]` choose representatives around zero.

A negative frequency means opposite rotation in the complex exponential. A real cosine contains both rotations. It does not mean a negative amount of signal.

You could label the bins with `arange(N)/(N*d)`, but the upper bins would then use aliases of the negative frequencies. That is unsuitable if you interpret them as physical signed frequencies, for example when computing momentum or derivatives.

## 6. Sampling determines what you can distinguish

For `N` equally spaced samples with spacing `d`:

| Quantity | Formula | Meaning |
|---|---|---|
| Sampling rate | $1/d$ | Samples per unit time or length |
| Period represented by the DFT | $L=Nd$ | Length of the periodically repeated block |
| Frequency-bin spacing | $\Delta f=1/(Nd)$ | Separation between basis frequencies |
| Nyquist magnitude | $1/(2d)$ | Boundary of the principal frequency interval |

The distance from the first to last sample is `(N-1)*d`; the periodic block length is `N*d`.

Decreasing `d` allows higher frequencies to be distinguished. Increasing `N` at fixed `d` makes the block longer and the frequency bins closer.

Frequencies differing by an integer multiple of `1/d` give identical samples. This ambiguity is **aliasing**. Sampling cannot recover which alias the original function contained without extra information.

## 7. Reading magnitude and phase

```python
magnitude = X.abs()
phase = torch.angle(X)   # radians; meaningful where magnitude is nonzero
real_part = X.real
imaginary_part = X.imag
```

Magnitude measures the coefficient size; phase describes its complex angle. Keeping only magnitude loses information needed to reconstruct the signal.

Under default normalization, an exactly bin-aligned cosine of amplitude `A`, away from zero and Nyquist, has two peaks of magnitude `N*A/2`.

For a real signal's one-sided amplitude spectrum:

```python
R = torch.fft.rfft(x)
amplitude = R.abs() / x.numel()
if x.numel() % 2 == 0:
    amplitude[1:-1] *= 2  # leave zero and Nyquist unchanged
else:
    amplitude[1:] *= 2   # odd lengths have no Nyquist bin
```

This amplitude interpretation assumes the default FFT normalization and no window. Off-bin tones spread across bins. `abs(X)**2` is squared spectral magnitude, not automatically a calibrated power spectral density.

## 8. Normalization

Use the same `norm` in the forward and inverse transforms:

| `norm` | Forward factor | Inverse factor |
|---|---:|---:|
| `"backward"` (default) | 1 | $1/N$ |
| `"forward"` | $1/N$ | 1 |
| `"ortho"` | $1/\sqrt N$ | $1/\sqrt N$ |

For a full complex transform, `ortho` preserves the discrete squared norm: `sum(abs(x)**2) == sum(abs(X)**2)` up to rounding. With default normalization, the right side is `N` times the left.

For multidimensional transforms, replace `N` with the product of the transformed lengths. Physical integration weights such as `dx` are separate from this normalization. [Normalization reference](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fft.html)

## 9. All function families in `torch.fft`

| Forward | Inverse | Purpose |
|---|---|---|
| `fft` | `ifft` | Full transform along one axis |
| `fft2` | `ifft2` | Full transform along two axes |
| `fftn` | `ifftn` | Full transform along selected axes |
| `rfft` | `irfft` | Real input, one-sided spectrum, one axis |
| `rfft2` | `irfft2` | Real input, two axes |
| `rfftn` | `irfftn` | Real input, selected axes |
| `hfft` | `ihfft` | Hermitian input, real spectrum, one axis |
| `hfft2` | `ihfft2` | Hermitian input, two axes |
| `hfftn` | `ihfftn` | Hermitian input, selected axes |

| Helper | Purpose |
|---|---|
| `fftfreq` | Labels for the full spectrum |
| `rfftfreq` | Labels for the one-sided real-input spectrum |
| `fftshift` | Rearrange full-spectrum bins with zero in the center |
| `ifftshift` | Undo that rearrangement |

These are the function families listed in the [PyTorch FFT module](https://docs.pytorch.org/docs/2.14/fft.html).

## 10. Real inputs: `rfft` and `irfft`

For real input, negative-frequency coefficients are conjugates of positive-frequency coefficients:

$$
X_{(-m)\bmod N}=X_m^*.
$$

`rfft` stores only `N//2 + 1` coefficients. Its output is generally complex. `rfftfreq` labels them from zero upward; for even `N` it labels Nyquist positive.

```python
R = torch.fft.rfft(x)
f_real = torch.fft.rfftfreq(x.numel(), d=1.0)
restored = torch.fft.irfft(R, n=x.numel())
```

Always supply the original `n` to `irfft`. A stored length of 3 could come from an original length of 4 or 5. The default inverse assumes the even case. Zero-frequency and, for even `n`, Nyquist coefficients must be real; their imaginary components cannot contribute to real output. [PyTorch `irfft` reference](https://docs.pytorch.org/docs/2.14/generated/torch.fft.irfft.html)

A general quantum wavefunction is complex, so use the full `fft`/`ifft` pair.

## 11. Axes, batches, and transform lengths

```python
signals = torch.randn(16, 128)       # 16 separate signals
spectra = torch.fft.fft(signals, dim=-1)
# shape: [16, 128]; batch axis is untouched

images = torch.randn(16, 32, 48)
spectra_2d = torch.fft.fft2(images, dim=(-2, -1))
# shape: [16, 32, 48]

real_spectra_2d = torch.fft.rfft2(images, dim=(-2, -1))
# shape: [16, 32, 25]
images_back = torch.fft.irfft2(real_spectra_2d, s=(32, 48), dim=(-2, -1))
```

`fft` defaults to the last axis; `fft2` to the last two. `fftn` with neither `dim` nor `s` transforms all axes, including a batch axis. Specify `dim` when batching.

For multidimensional real transforms, only the last transformed axis is shortened. Other transformed axes retain full frequency ordering. A two-dimensional coefficient labels a pair of frequencies, one per axis. [Multidimensional reference](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftn.html)

`n` sets the transform length for 1D; `s` supplies lengths for multiple axes. Increasing a forward transform length appends zeros; decreasing it trims samples. Generate frequency labels using the actual transform length. Zero-padding gives more closely spaced spectral samples, but adds no measured information.

## 12. `fftshift` is a permutation

```python
f_plot = torch.fft.fftshift(f)
X_plot = torch.fft.fftshift(X)
# f_plot: [-0.50, -0.25, 0.00, 0.25]

restored = torch.fft.ifft(torch.fft.ifftshift(X_plot))
```

Shift both labels and coefficients together. This is useful for plots; it does not calculate a new transform. Undo the shift before an inverse expecting ordinary FFT ordering.

For batches, pass `dim` explicitly to avoid shifting batch entries. Use `ifftshift` as the inverse: for odd lengths, applying `fftshift` twice does not undo it. [Shift reference](https://docs.pytorch.org/docs/2.14/generated/torch.fft.fftshift.html)

## 13. What is `hfft`?

`rfft` exploits a real input to obtain a Hermitian spectrum. `hfft` handles the opposite situation: a Hermitian input whose transform is real.

Hermitian here means `a[-j] = conj(a[j])` under periodic indexing. `hfft` accepts the nonnegative-index part of that input. `ihfft` maps a real spectrum back to the compact Hermitian representation. Supply the full intended length to `hfft`, especially for odd lengths.

This is a specialized symmetry case, not the usual way to transform a real signal. [PyTorch `hfft` reference](https://docs.pytorch.org/docs/main/generated/torch.fft.hfft.html)

## 14. Quantum mechanics: why transform at all?

Plane waves make derivatives simple:

$$
\frac{d}{dx}e^{ikx}=ik e^{ikx}
$$

$$
\frac{d^2}{dx^2}e^{ikx}=-k^2 e^{ikx}
$$

Thus, on a periodic Fourier grid:

$$
\partial_x\psi=\operatorname{IFFT}(ik\operatorname{FFT}(\psi))
$$

$$
\partial_x^2\psi=\operatorname{IFFT}(-k^2\operatorname{FFT}(\psi))
$$

This follows directly by differentiating the inverse Fourier expansion term by term. It differentiates the represented Fourier interpolant; unresolved features in the original function remain unresolved.

```python
N = 128
L = 20.0
dx = L / N
position = torch.arange(N, dtype=torch.float64) * dx
k = 2 * torch.pi * torch.fft.fftfreq(N, d=dx, dtype=torch.float64)

psi = torch.exp(1j * (2 * torch.pi * 3 / L) * position)
psi_hat = torch.fft.fft(psi)
dpsi = torch.fft.ifft(1j * k * psi_hat)
d2psi = torch.fft.ifft(-k.square() * psi_hat)
```

`fftfreq` gives cycles per length. Multiplication by `2*pi` gives angular wavenumber. Momentum is `p = hbar*k`.

The kinetic-energy operator becomes

$$
T\psi=\operatorname{IFFT}\left(\frac{\hbar^2k^2}{2m}
\operatorname{FFT}(\psi)\right).
$$

For a free particle, each Fourier coefficient evolves independently:

```python
hbar = 1.0
mass = 1.0
dt = 0.01
energy = hbar**2 * k.square() / (2 * mass)
psi_next = torch.fft.ifft(torch.exp(-1j * energy * dt / hbar) * psi_hat)
```

The phase has unit magnitude, so this step preserves the discrete norm up to rounding. With a position-dependent potential, the full evolution is not generally just this kinetic step; operator splitting introduces a time-step approximation.

For a grid starting at `x0`, use the continuum convention

$$
\widetilde{\psi}(k)=\frac{1}{\sqrt{2\pi}}\int\psi(x)e^{-ikx}\,dx.
$$

At FFT wavenumbers, this is approximated by

$$
\widetilde\psi(k_m)\approx
\frac{dx}{\sqrt{2\pi}}e^{-ik_mx_0}\operatorname{FFT}(\psi)_m.
$$

This physical scaling is separate from `norm`. The origin phase cancels in the derivative and kinetic-operator round trips above, but matters when interpreting continuum Fourier amplitudes.

## 15. Periodicity, leakage, filtering, and convolution

The DFT represents a periodic block. For a periodic spatial grid, use `x0 + arange(N)*dx`; do not duplicate both endpoints of one period.

If repeating the sampled block makes a discontinuity, many Fourier modes are needed to represent that boundary. A tone between bins also spreads over multiple bins: **spectral leakage**. A window can reduce leakage at the cost of broader peaks and altered amplitudes; it changes the input.

Filtering means changing coefficients before inversion. For example, keeping only frequencies with `abs(f) <= cutoff` removes higher-frequency components. A sharp cutoff can cause ringing.

With default normalization, `ifft(fft(a)*fft(b))` computes circular convolution of equal-length sequences. To obtain full linear convolution of lengths `A` and `B`, zero-pad both transforms to at least `A+B-1` and retain that many output samples.

For quantum simulations, a packet reaching a periodic boundary reappears at the opposite side. The FFT alone does not implement a hard wall or absorbing boundary. Choose a domain and boundary treatment appropriate to the problem.

## 16. Dtypes, devices, and gradients

Float32 data normally produces complex64 spectra; float64 produces complex128. Keep frequency grids on the same device as the data:

```python
k = 2 * torch.pi * torch.fft.fftfreq(
    psi.shape[-1], d=dx, device=psi.device, dtype=psi.real.dtype
)
```

PyTorch FFT operations support autograd and accelerated CPU/CUDA implementations. A real scalar loss can backpropagate through an FFT:

```python
u = torch.randn(8, dtype=torch.float64, requires_grad=True)
U = torch.fft.fft(u, norm="ortho")
loss = U.abs().square().sum()
loss.backward()  # gradient is 2*u, up to rounding
```

This result follows from norm preservation: the loss equals `sum(u**2)`. [PyTorch FFT and autograd overview](https://pytorch.org/blog/the-torch-fft-module-accelerated-fast-fourier-transforms-with-autograd-in-pytorch/)

Half-precision FFT support has device and size restrictions. For learning and numerical checks, use float64/complex128; assess speed and accuracy before changing precision.

## 17. A short checklist

1. Identify the transformed axes and sample spacing.
2. Use `fft` for general complex data or `rfft` for real data.
3. Match the frequency labels to the transformed length and ordering.
4. Use `2*pi*fftfreq(...)` when formulas require angular wavenumber.
5. Keep forward and inverse normalization consistent.
6. Preserve phase if you need reconstruction.
7. Account for periodic boundaries and unresolved frequencies.

The central relationship is:

```text
samples --fft--> wave coefficients --ifft--> samples
                         |
                 fftfreq labels each coefficient
```

API details were checked against the linked PyTorch documentation. Examples are illustrative; local execution was unavailable because the existing project Python environment could not start through uv.
