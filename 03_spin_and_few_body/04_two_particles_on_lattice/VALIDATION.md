# Two-particle lattice validation

Generated with PyTorch 2.13.0+cpu and NumPy 2.5.2 on CPU in float64/complex128.
No interactions, spatial quadrature, or basis cutoff are used.

## Finite-chain spectra

Open chain, J=1, zero site energies. Reference: allowed sums of exact sine-mode energies.

| Sites | Statistics | Dimension | Ground energy | Max spectral error | NumPy matrix difference |
| --- | --- | --- | --- | --- | --- |
| 2 | distinguishable | 4 | -2.000000000000 | 8.882e-16 | 0.000e+00 |
| 2 | boson | 3 | -2.000000000000 | 8.899e-16 | 2.220e-16 |
| 2 | fermion | 1 | 0.000000000000 | 6.661e-16 | 0.000e+00 |
| 3 | distinguishable | 9 | -2.828427124746 | 9.533e-16 | 0.000e+00 |
| 3 | boson | 6 | -2.828427124746 | 1.332e-15 | 2.220e-16 |
| 3 | fermion | 3 | -1.414213562373 | 4.441e-16 | 2.220e-16 |
| 4 | distinguishable | 16 | -3.236067977500 | 3.109e-15 | 0.000e+00 |
| 4 | boson | 10 | -3.236067977500 | 8.882e-16 | 2.220e-16 |
| 4 | fermion | 6 | -2.236067977500 | 8.882e-16 | 2.220e-16 |
| 6 | distinguishable | 36 | -3.603875471610 | 7.550e-15 | 0.000e+00 |
| 6 | boson | 21 | -3.603875471610 | 3.109e-15 | 2.220e-16 |
| 6 | fermion | 15 | -3.048917339522 | 2.220e-15 | 2.220e-16 |
| 8 | distinguishable | 64 | -3.758770483144 | 5.329e-15 | 0.000e+00 |
| 8 | boson | 36 | -3.758770483144 | 4.441e-15 | 2.220e-16 |
| 8 | fermion | 28 | -3.411474127810 | 2.665e-15 | 2.220e-16 |

## Physical finite-size scaling

J=1 and lattice spacing are fixed. Increasing L lengthens the chain.
The full two-particle basis is used at every L. This is not a mesh-refinement error.
Both ground energies approach -4 as L grows with particle number fixed at two.

| L | Boson E0 | Fermion E0 | (L+1)^2*(boson E0+4) | (L+1)^2*(fermion E0+4) |
| --- | --- | --- | --- | --- |
| 2 | -2.000000000000 | 0.000000000000 | 18.00000000 | 36.00000000 |
| 4 | -3.236067977500 | -2.236067977500 | 19.09830056 | 44.09830056 |
| 8 | -3.758770483144 | -3.411474127810 | 19.53959087 | 47.67059565 |
| 12 | -3.883767269704 | -3.712795686159 | 19.64333142 | 48.53752904 |

Predicted scaled limits: bosons 2*pi^2=19.73920880; fermions 5*pi^2=49.34802201.
The finite-size shift is distinct from the small spectral errors in the preceding table.

## Two-site pair correlations

Initial state: one particle on each site. J=hbar=1; onsite energies are zero.
Analytical same-site probability: sin(2t)^2/2 for labeled particles, sin(2t)^2 for bosons, zero for fermions.

| t/pi | Statistics | Same-site probability | Analytical | Mean n0 | Mean n1 |
| --- | --- | --- | --- | --- | --- |
| 0 | distinguishable | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.125 | distinguishable | 0.250000000000 | 0.250000000000 | 1.000000000000 | 1.000000000000 |
| 0.25 | distinguishable | 0.500000000000 | 0.500000000000 | 1.000000000000 | 1.000000000000 |
| 0.5 | distinguishable | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0 | boson | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.125 | boson | 0.500000000000 | 0.500000000000 | 1.000000000000 | 1.000000000000 |
| 0.25 | boson | 1.000000000000 | 1.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.5 | boson | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0 | fermion | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.125 | fermion | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.25 | fermion | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |
| 0.5 | fermion | 0.000000000000 | 0.000000000000 | 1.000000000000 | 1.000000000000 |

Maximum probability error: 8.882e-16.
Mean site counts are identical, but pair correlations differ. No interaction was added.

## Complex hopping and batched evolution

Four sites. Seed 74 gives a complex Hermitian one-particle matrix, including long-range hopping.
Six normalized complex initial states use batch shape (2,3). Times are [0,0.17,0.91]; hbar=1.3.
PyTorch uses projection and matrix exponentials. NumPy uses occupation moves and spectral evolution.

| Statistics | Matrix difference | State difference | Norm error | Energy drift | Particle-count error | Reverse error | Exchange error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| distinguishable | 0.000e+00 | 8.968e-16 | 3.331e-16 | 1.166e-15 | 6.661e-16 | 1.259e-15 | not required |
| boson | 2.221e-16 | 7.789e-16 | 6.661e-16 | 8.882e-16 | 8.882e-16 | 7.574e-16 | 0.000e+00 |
| fermion | 2.220e-16 | 1.159e-15 | 8.882e-16 | 1.221e-15 | 1.776e-15 | 1.578e-15 | 0.000e+00 |

Each requested time is evaluated directly. These are floating-point checks, not dt convergence.
Tests also cover canonical basis order, double occupancy, uniform energy shifts, invalid inputs, and batch/single agreement.
