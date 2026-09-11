# FFT propagation on a rectangular grid

## 1. Grid and tensor ordering

Use $x_i=-L_x+i\,dx$, $y_j=-L_y+j\,dy$, with
$dx=2L_x/N_x$, $dy=2L_y/N_y$. Exclude the duplicate upper endpoint on both
periodic axes. Store a field as `(Ny,Nx)`, so `field[j,i]` means `(y_j,x_i)`.

Broadcast `x[None,:]` with `y[:,None]`. An explicit mesh is optional; if using
`meshgrid`, pass axes in y/x order with `indexing="ij"`.

For multiple initial packets use `(Ny,Nx,B)`, and keep the final axis for
the batch. The propagator transforms `dim=(0,1)` explicitly. The default
`fft2` axes would include the batch axis for this layout, so leaving out `dim`
would change the calculation. See the
[PyTorch FFT2 documentation](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft2.html).

## 2. Area normalization and spectral kinetic energy

Replace the spatial integral by

$$
\langle\psi|\phi\rangle_h=dx\,dy\sum_{j,i}\psi_{ji}^*\phi_{ji}.
$$

Normalize each batch member by its own area-weighted norm. With an orthonormal
FFT on both axes, Parseval's identity preserves this same inner product.
Do not normalize by `dx` alone, and do not sum the batch dimension.

Construct unshifted angular wave numbers separately for each axis:

```python
kx = 2 * torch.pi * torch.fft.fftfreq(Nx, d=dx, dtype=torch.float64)
ky = 2 * torch.pi * torch.fft.fftfreq(Ny, d=dy, dtype=torch.float64)
kinetic = hbar**2 * (ky[:, None] ** 2 + kx[None, :] ** 2) / (2 * mass)
```

For a Fourier mode $e^{i(k_xx+k_yy)}$, differentiating twice gives kinetic
energy $\hbar^2(k_x^2+k_y^2)/(2m)$. This is the reason for adding the squared
wave numbers, rather than multiplying them.

## 3. Symmetric splitting

The same derivation as the previous chapter applies:

$$
U(dt)\approx e^{-iVdt/(2\hbar)}F_2^{-1}
e^{-i\epsilon dt/\hbar}F_2e^{-iVdt/(2\hbar)}.
$$

Precompute phase fields `(Ny,Nx,1)` and apply them to all packet columns.
Use `fft2` and `ifft2` with `norm="ortho"` and `dim=(0,1)` during propagation.
For a saved single-packet history `(S,Ny,Nx)`, use `dim=(1,2)` to calculate
spectral energy for every time without transforming the time axis.

The FFT itself is separable into successive 1D transforms, even when the
potential and wavefunction are nonseparable. The sampled potential enters as
one complete 2D phase; do not drop or split away its xy coupling.

At a fixed finite grid, global time error is second order. Norm and
reversibility hold to roundoff for real potentials, but energy and trajectory
errors remain. No per-step renormalization is used.

## 4. Observables

Energy combines position and momentum representations:

$$
\langle H\rangle_h=dx\,dy
\left[\sum_{q,p}\epsilon_{qp}|\widehat\psi_{qp}|^2
+\sum_{j,i}V_{ji}|\psi_{ji}|^2\right].
$$

Measure the center $\boldsymbol\mu=\langle\mathbf r\rangle$ and covariance
$\Sigma_{ab}=\langle(r_a-\mu_a)(r_b-\mu_b)\rangle$. In particular, the xy
entry distinguishes a tilted coupled packet from a separable product.
The statistics function divides by the measured norm; norm is checked
independently so this does not hide loss in the evolution.

Define edge probability with the union of four strips. Adding four separate
strip integrals would double-count corners. Inspect the maximum over saved
times and compare with a larger box. Position moments on a periodic coordinate
are useful here only while the packet stays away from the seams.

The quadratic potential is not periodic on the infinite plane. FFT evolution
uses its periodic sampled extension, including seams where values or derivatives
can jump. The infinite-plane analytical solution is a reference only when the
state has negligible weight near those seams.

## 5. Validation references

- Free plane waves on unequal axes test phases, `hbar/m`, FFT ordering, and area.
- Free Gaussians test two center velocities, two spreading rates, and zero xy
  covariance without splitting error.
- Uncoupled product evolution must match two independent 1D evolutions.
- Coupled coherent states test a nonzero xy potential and covariance against
  exact rotated normal modes.
- A tiny spectral matrix exponential isolates time error for a generic
  periodic nonseparable potential.
- NumPy tests use the same grids, initial states, and time steps.

For the tiny matrix, reshape identity columns as `(Ny,Nx,Ny*Nx)`, apply the
spectral kinetic operator, then reshape back. Flattening is C order:
`flat_index = j*Nx+i`. This is a validation tool, not the production solver.

## 6. Scaling and convergence

Let `M=Ny*Nx`. Each step costs approximately `O(B*M*log(M))` and uses `O(B*M)`
working memory. `S` stored snapshots use `O(S*B*M)` memory. A dense Hamiltonian
would require `O(M^2)` storage. At the default 80-by-96 grid, one complex128 dense
matrix alone would use about 900 MiB; a single state uses about 120 KiB.
The default stores 31 snapshots, not every integration step.

Hold other parameters fixed when studying each error source. Refine both
spatial axes in a fixed box, then enlarge the box at fixed spacing. Reduce the
time step separately. Our default spatial grid is already well resolved;
most remaining error is time splitting. The report also includes a coarse grid
and a deliberately small box so these different failures are visible.
