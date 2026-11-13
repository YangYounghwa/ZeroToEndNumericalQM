# Hubbard dimer validation

CPU: PyTorch 2.13.0+cpu, NumPy 2.5.2; float64/complex128.

## Symmetric dimer and exchange approximation

Hopping is 1. The full sector is retained. U changes the model, not the numerical resolution.
For positive U, J is the exact singlet-triplet gap. Compare 4/U with J.

| U | E0 | Energy error | Double occupancy | D error | Relative gap approximation error |
| --- | --- | --- | --- | --- | --- |
| -8 | -8.472135955000 | 3.553e-15 | 0.947213595500 | 1.110e-16 | not applicable |
| 0 | -2.000000000000 | 0.000e+00 | 0.500000000000 | 0.000e+00 | not applicable |
| 2 | -1.236067977500 | 2.220e-16 | 0.276393202250 | 1.110e-16 | 6.180340e-01 |
| 4 | -0.828427124746 | 2.220e-16 | 0.146446609407 | 5.551e-17 | 2.071068e-01 |
| 8 | -0.472135955000 | 6.106e-16 | 0.052786404500 | 0.000e+00 | 5.901699e-02 |
| 16 | -0.246211251235 | 5.551e-16 | 0.014928749927 | 5.204e-18 | 1.538820e-02 |
| 32 | -0.124515496597 | 1.020e-14 | 0.003861061643 | 4.337e-17 | 3.891109e-03 |
| 64 | -0.062439083763 | 3.594e-15 | 0.000973710759 | 1.084e-17 | 9.756107e-04 |

The leading relative gap error is proportional to 1/U^2 at fixed hopping.
This approximation error is distinct from eigensolver roundoff.

## Biased dimer dynamics

U=3.1, bias=0.8, hopping=1. Seed 52. Six states use batch shape (2,3).
Times are 0, 0.3, and 1.7. No integration time step is used.

- Independent matrix difference: 0.000e+00.
- Independent state difference: 3.820e-15.
- Eigenpair residual: 8.882e-16.
- Norm error: 5.329e-15.
- Energy drift: 1.332e-14.
- Particle-count error: 1.066e-14.
- Reverse-state error: 5.503e-15.

Tests also check the full fermion algebra, atomic and noninteracting limits,
exact singlet doublon oscillations, a stationary triplet, and invalid inputs.
CUDA is tested only when available. No bulk-limit claim follows from a dimer.
