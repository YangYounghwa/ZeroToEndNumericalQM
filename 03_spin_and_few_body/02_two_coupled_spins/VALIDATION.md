# Two coupled spins: numerical validation

Generated with PyTorch 2.13.0+cpu, CPU float64/complex128.
H=J*sigma1.sigma2/4-gamma*hbar*(B1.sigma1+B2.sigma2)/2.
Basis order (++,+-,-+,--); J and detuning have energy units.

## Zero-field spectrum

The triplet has energy J/4; the singlet has energy -3J/4. hbar=1.
Residuals use the explicitly derived singlet/triplet states, not numerically selected eigenvectors.

| J | Triplet E | Singlet E | Maximum eigenpair residual | Sorted numerical energies |
| --- | --- | --- | --- | --- |
| -1 | -0.250000 | 0.750000 | 0.000e+00 | -0.250000, -0.250000, -0.250000, 0.750000 |
| 0 | 0.000000 | -0.000000 | 0.000e+00 | 0.000000, -0.000000, -0.000000, 0.000000 |
| 1 | 0.250000 | -0.750000 | 0.000e+00 | -0.750000, 0.250000, 0.250000, 0.250000 |
| 2 | 0.500000 | -1.500000 | 0.000e+00 | -1.500000, 0.500000, 0.500000, 0.500000 |

## Exchange dynamics and correlations

J=hbar=1, no fields, initial |+->. Exact swapped probability is sin(t/2)^2.
Maximum phase-aligned state error to the analytical solution: 1.127e-15.
Local z columns are Pauli expectations; multiply by hbar/2 for spin angular momentum.

| Time/pi | P(+-) | P(-+) | Local z1 | Local z2 | Connected zz | Product determinant |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.000000 | 0.000000 | 1.000000 | -1.000000 | 0.000000 | 0.000000 |
| 0.25 | 0.853553 | 0.146447 | 0.707107 | -0.707107 | -0.500000 | 0.353553 |
| 0.5 | 0.500000 | 0.500000 | 0.000000 | -0.000000 | -1.000000 | 0.500000 |
| 0.75 | 0.146447 | 0.853553 | -0.707107 | 0.707107 | -0.500000 | 0.353553 |
| 1 | 0.000000 | 1.000000 | -1.000000 | 1.000000 | 0.000000 | 0.000000 |
| 1.5 | 0.500000 | 0.500000 | 0.000000 | -0.000000 | -1.000000 | 0.500000 |
| 2 | 1.000000 | 0.000000 | 1.000000 | -1.000000 | 0.000000 | 0.000000 |

At t=pi/2, local Bloch vectors vanish but joint correlations do not.
The coefficient determinant is 1/2, so this pure state cannot be written as a product of two spinors.
At t=pi a complete swap has occurred and the state is a product again.

## Unequal longitudinal fields

J=gamma=hbar=1, B1z=delta/2 and B2z=-delta/2; initial |+->.
For each delta, sample the first transfer peak t=pi/sqrt(J^2+delta^2) exactly.
delta is an energy difference, not a numerical discretization parameter.

| delta | First peak time | Predicted maximum transfer | Numerical transfer | State error to block solution |
| --- | --- | --- | --- | --- |
| 0 | 3.141593 | 1.000000 | 1.000000 | 4.807e-16 |
| 0.5 | 2.809926 | 0.800000 | 0.800000 | 8.528e-16 |
| 1 | 2.221441 | 0.500000 | 0.500000 | 8.618e-16 |
| 2 | 1.404963 | 0.200000 | 0.200000 | 4.003e-16 |

## General-field NumPy comparison

Three independent parameter/state pairs; gamma=-0.8, hbar=0.7, times -1 through 9.
Cases: zero Hamiltonian, unequal noncollinear fields, and unequal longitudinal fields.
NumPy uses eigh; PyTorch uses native matrix_exp. Errors are maxima over time.

| Case | State error | Connected-correlation difference | Norm error | Energy drift |
| --- | --- | --- | --- | --- |
| 1 | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 2 | 5.593e-15 | 4.663e-15 | 9.104e-15 | 3.275e-15 |
| 3 | 4.961e-15 | 6.883e-15 | 6.883e-15 | 2.609e-15 |

## Which quantities are conserved?

J=hbar=gamma=1. Each entry is the maximum absolute matrix element of a commutator.
Zero commutator implies conservation for every initial state in this static model.
S_total,z is in hbar units and S_total^2 in hbar^2 units.

| Fields | [H,S_total,z] | [H,S_total^2] |
| --- | --- | --- |
| Zero field | 0.000e+00 | 0.000e+00 |
| Uniform z | 0.000e+00 | 0.000e+00 |
| Uniform tilted | 2.500e-01 | 0.000e+00 |
| Unequal z | 0.000e+00 | 1.400e+00 |
| Unequal transverse | 5.000e-01 | 5.000e-01 |

## Crank-Nicolson time error

J=hbar=1, zero fields, initial |+->, final time=10. Exact reference: exchange block solution.
State and swapped-probability errors are at the final time; invariant errors cover all steps.

| dt | State error | Previous/current error | Swap-probability error | Norm error | Energy drift |
| --- | --- | --- | --- | --- | --- |
| 0.4 | 2.879e-02 | — | 1.496e-02 | 2.442e-15 | 3.331e-16 |
| 0.2 | 7.268e-03 | 3.961 | 3.909e-03 | 1.366e-14 | 3.275e-15 |
| 0.1 | 1.821e-03 | 3.990 | 9.881e-04 | 5.329e-15 | 1.166e-15 |
| 0.05 | 4.556e-04 | 3.998 | 2.477e-04 | 1.099e-14 | 3.275e-15 |

Norm and energy conservation do not guarantee correct exchange timing.
The four-state spin model has no spatial or basis-cutoff error; the CN time error is studied separately from physical detuning.
