# Split-operator Fourier propagation

## 1. Periodic grid and Fourier modes

Use $x_j=-L+j\,dx$, $dx=2L/N$, for `j=0,...,N-1`. Exclude `+L` because it
duplicates `-L` on a periodic grid. This differs from Chapter 3's interior-only
grid with zero Dirichlet walls.

The FFT uses angular wave numbers

```python
k = 2 * torch.pi * torch.fft.fftfreq(N, d=dx, dtype=torch.float64)
```

For even `N`, the order is `[0,1,...,N/2-1,-N/2,...,-1] * pi/L`.
`fftfreq` returns cycles per unit length; the factor `2*pi` converts to radians.
Keep this unshifted ordering during propagation. `fftshift` is for display only.
See the [PyTorch fftfreq documentation](https://docs.pytorch.org/docs/stable/generated/torch.fft.fftfreq.html).

With `norm="ortho"`, the discrete Fourier matrix `F` is unitary:

$$
\widehat\psi=F\psi,\qquad
dx\sum_j|\psi_j|^2=dx\sum_q|\widehat\psi_q|^2.
$$

Therefore `dx*abs(fft(psi, norm="ortho"))**2` gives probabilities per discrete
Fourier mode. It is not a continuous density per unit `k`; divide by `dk=pi/L`
for that interpretation. The phase from the grid origin does not change these
probabilities. See [PyTorch FFT normalization](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html).

## 2. Derive the update

For a fixed Hamiltonian, exact evolution is $U(dt)=e^{-i(T+V)dt/\hbar}$.
Since `T` and `V` generally do not commute, applying their full steps separately
introduces a leading error of order `dt^2` per step.

Let $A=-iV/\hbar$ and $B=-iT/\hbar$. Expanding the symmetric product gives

$$
e^{A dt/2}e^{B dt}e^{A dt/2}
=I+(A+B)dt+\tfrac12(A^2+AB+BA+B^2)dt^2+O(dt^3).
$$

This matches $e^{(A+B)dt}$ through second order. With
$\epsilon_q=\hbar^2k_q^2/(2m)$, one step is

$$
\psi^{n+1}=
e^{-iVdt/(2\hbar)}F^{-1}
\left[e^{-i\epsilon dt/\hbar}F\left(e^{-iVdt/(2\hbar)}\psi^n\right)\right].
$$

Precompute the two phase vectors once. For state columns `(N,B)`, use FFT
`dim=0` and phase vectors shaped `(N,1)`. The batch dimension is never transformed.
The symmetric split-operator construction is also derived in the
[Algorithm Archive implementation](https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html).

## 3. What is exact, and what is approximate?

Every factor is unitary for real potentials and real time steps, so norm is
preserved up to roundoff. Applying the same steps with negative `dt` reverses
the numerical evolution. Do not renormalize after steps: it hides norm errors.

At fixed finite grid, the local splitting error is `O(dt^3)` and the global
error at a fixed final time is `O(dt^2)`. Halving `dt` should reduce state error
by about four in the convergent regime. There is no explicit-method norm
instability restriction, but large time steps still give inaccurate dynamics.

The split update generally does not commute with the full spectral Hamiltonian.
Energy can drift even when norm and reversibility are excellent. Compute

$$
\langle H\rangle=dx\sum_q\epsilon_q|\widehat\psi_q|^2
+dx\sum_j V_j|\psi_j|^2
$$

and verify that its error decreases with `dt`. Free modes and constant potentials
are exact special cases: the operators commute.

Fourier differentiation accurately represents resolved periodic smooth states.
A discontinuous barrier, unresolved wavelengths, or an initial packet truncated
at the box edge can produce large errors. Check the probability near the highest
represented `|k|` and refine the grid; a small high-frequency tail alone is not
a complete accuracy proof.

## 4. References that isolate different errors

For a small grid, explicitly build
$H_{FFT}=F^{-1}\operatorname{diag}(\epsilon)F+\operatorname{diag}(V)$.
Compare splitting with `torch.linalg.matrix_exp(-1j*t*H/hbar)` on this same grid.
Only this reference isolates time-splitting error.

The NumPy implementation repeats the FFT calculation with identical parameters.
The SciPy comparison instead uses the three-point finite-difference kinetic
matrix, with periodic corner entries joining the first and last sites. It has
`3*N` stored entries in CSR form. Crank-Nicolson factors its left matrix once
using CSC storage for sparse LU, then reuses that factorization.

The finite-difference free dispersion is

$$
\epsilon_{FD}(k)=\frac{2\hbar^2}{m\,dx^2}\sin^2(k\,dx/2),
$$

whereas FFT uses $\epsilon_{FFT}=\hbar^2 k^2/(2m)$. They agree as `dx` tends to
zero for fixed resolved `k`. Do not demand identical results on a coarse grid or
compare a Dirichlet matrix exponential with a periodic FFT state as a time-only
test. The periodic corner links are the explicit change from earlier CN code.

## 5. Cost, experiments, and limitations

FFT propagation costs `O(B*N*log(N))` per step and `O(B*N)` working memory.
Saving `S` snapshots costs `O(S*B*N)` memory. The default stores every 25 steps,
plus the initial and final states. The dense spectral matrix exists only for
small validation problems, with `O(N^2)` storage.

Study one variable at a time:

1. Fix the grid and final time; reduce `dt` against a spectral exponential.
2. Fix the physical box and time; refine `N`, inspecting states and high-k mass.
3. Fix `dx`; enlarge `L`, comparing common-grid states and edge histories.
4. Inspect late-time region budgets to find a separation window before wraparound.
5. Vary packet width as a separate physical experiment, recording incident tails.

The Gaussian has no compact support. Initial barrier overlap and tails at the
periodic seam must be negligible at the desired accuracy. Saved edge samples
can miss motion between snapshots; combine them with domain comparisons.
No absorbing boundaries are used here.
