# Two-dimensional wave-packet validation

Generated with PyTorch 2.13.0+cpu, CPU float64/complex128, m=hbar=1.
Default coupled potential: V=(x^2+0.7*x*y+1.69*y^2)/2.
Default coherent packet: center=(-2,1), momentum=(0.4,-0.6), covariance from the coupled ground state.
Lx and Ly denote half extents. Fields are indexed (y,x); batches add a final axis.

## Same-grid spectral time-step reference

Nx=12, Ny=10, Lx=4, Ly=3, T=1. V=0.7*cos(pi*x/4)*sin(pi*y/3).
Compare FFT2 splitting with the exact exponential of the same 120x120 spectral matrix.

| dt | Phase-aligned state error | Previous/current error |
| --- | --- | --- |
| 0.1 | 7.510e-04 | — |
| 0.05 | 1.876e-04 | 4.004 |
| 0.025 | 4.688e-05 | 4.001 |
| 0.0125 | 1.172e-05 | 4.000 |

## Coupled coherent-state time convergence

Default Nx=96, Ny=80, Lx=10, Ly=8, final time=6; snapshots at intervals no greater than 0.2.
State, center, covariance, and energy references are the infinite-plane normal-mode solution.
Errors in observables are maxima over saved times; state error is at the final time.

| dt | State error | Center error | Covariance error | Energy error | Norm error |
| --- | --- | --- | --- | --- | --- |
| 0.08 | 3.210e-03 | 3.144e-03 | 1.004e-03 | 2.944e-03 | 2.209e-14 |
| 0.04 | 8.024e-04 | 7.817e-04 | 2.522e-04 | 7.359e-04 | 5.052e-14 |
| 0.02 | 2.006e-04 | 1.954e-04 | 6.301e-05 | 1.840e-04 | 9.481e-14 |
| 0.01 | 5.014e-05 | 4.885e-05 | 1.575e-05 | 4.600e-05 | 2.011e-13 |

Conserved norm does not make the trajectory exact. Reducing dt improves the physical observables as well.

## Spatial-grid convergence

Keep Lx=10, Ly=8, dt=0.005, final time=6 fixed. Reference grid: Nx=160, Ny=128.
Compare the same final state on nested points without normalizing the sampled reference.

| Nx | Ny | dx | dy | State error to fine grid | State error to continuum | NumPy state difference |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 16 | 1 | 1 | 5.926e-01 | 5.926e-01 | 1.180e-13 |
| 40 | 32 | 0.5 | 0.5 | 1.752e-05 | 2.153e-05 | 3.771e-14 |
| 80 | 64 | 0.25 | 0.25 | 7.138e-14 | 1.254e-05 | 1.381e-13 |

Once grid error is negligible, the remaining continuum error is chiefly time splitting.
Refining the grid cannot remove a fixed-dt error.

## Domain convergence at fixed spacing

dx=dy=0.25, dt=0.02, final time=6, unchanged coupled packet and potential.
State reference: Lx=10, Ly=8 on the same spacing. Edge strips have width 1.5.
Sample every 0.1 time unit; corners are counted once in the edge union.

| Lx | Ly | State error to large box | Maximum center error | Maximum edge probability |
| --- | --- | --- | --- | --- |
| 4 | 3 | 1.202e-01 | 7.320e-02 | 4.874e-01 |
| 6 | 4 | 1.075e-03 | 2.081e-04 | 1.421e-02 |
| 8 | 6 | 4.130e-08 | 1.964e-04 | 3.247e-08 |

The periodic extension of a quadratic potential is not an infinite oscillator.
The comparison is valid only while packet tails at the seams are negligible.
Small-box errors include initial truncation and later boundary effects.

## Free anisotropic Gaussian

Nx=128, Ny=90, Lx=20, Ly=18, sigma=(1,1.4), center=(-3,1), momentum=(0.8,-0.4).
For V=0 the kinetic step is exact on the Fourier grid; there is no splitting error.
Remaining differences from the infinite plane come from sampling, periodic boundaries, and roundoff.

| Time | State error to continuum | Mean x | Mean y | Variance x | Variance y |
| --- | --- | --- | --- | --- | --- |
| 0 | 2.180e-16 | -3.000000000 | 1.000000000 | 1.000000000 | 1.960000000 |
| 1 | 1.918e-15 | -2.200000000 | 0.600000000 | 1.250000000 | 2.087551020 |
| 2 | 5.000e-15 | -1.400000000 | 0.200000000 | 2.000000000 | 2.470204082 |
| 3 | 1.199e-12 | -0.600000000 | -0.200000000 | 3.250000000 | 3.107959184 |
