# Transverse-field Ising validation

CPU: PyTorch 2.13.0+cpu, NumPy 2.5.2. Real matrices use float64.

## Critical open chains

J=g=1. The full finite spin space is retained. Compare with exact open-chain formulas.
L changes the physical model. Residuals and energy errors measure numerical accuracy.

| L | D | E0/L | L gap | Ground energy error | Gap error | Max sparse residual | Dense comparison error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 4 | -1.1180339887 | 2.4721359550 | 0.000e+00 | 2.220e-16 | 1.413e-15 | 2.220e-16 |
| 4 | 16 | -1.1896926208 | 2.7783708427 | 3.553e-15 | 3.442e-15 | 4.398e-15 | 1.776e-15 |
| 6 | 64 | -1.2160383018 | 2.8928803261 | 8.882e-16 | 3.997e-15 | 4.865e-12 | 1.421e-14 |
| 8 | 256 | -1.2297439309 | 2.9525875028 | 1.599e-14 | 8.715e-15 | 1.665e-14 | 5.329e-14 |
| 10 | 1024 | -1.2381490000 | 2.9892037435 | 1.066e-14 | 2.892e-14 | 4.918e-14 | not run |
| 12 | 4096 | -1.2438309258 | 3.0139449374 | 2.665e-14 | 7.494e-15 | 6.677e-14 | not run |

Bulk targets: E0/L -> -1.2732395447; L gap -> 3.1415926536.
These limits are not expected to equal finite-chain results.

## Sparse tolerance study

L=10 and J=g=1 are fixed. Report actual errors; roundoff need not improve monotonically.

| Requested tolerance | E0 error | Max residual |
| --- | --- | --- |
| 1e-04 | 5.329e-15 | 1.415e-06 |
| 1e-08 | 8.882e-15 | 5.815e-09 |
| 1e-12 | 1.066e-14 | 4.918e-14 |

## Fixed-size field study

L=6, J=1. Mx and Mz are mean Pauli magnetizations. C is <Z0 Z5>.
Finite parity eigenstates have zero Mz, even when C is large.

| g | E0 | Gap | Mx | Mz | C | Ground parity |
| --- | --- | --- | --- | --- | --- | --- |
| 0.25 | -5.1258509115 | 4.577638e-04 | 0.1692362321 | 4.932e-12 | 0.9369673394 | 1.0000000000 |
| 0.5 | -5.5220295708 | 2.346189e-02 | 0.3679092324 | -2.121e-14 | 0.7290866873 | 1.0000000000 |
| 1 | -7.2962298106 | 4.821467e-01 | 0.7711447056 | -5.551e-17 | 0.1561482947 | 1.0000000000 |
| 2 | -12.6309642765 | 2.293278e+00 | 0.9464119674 | -8.327e-17 | 0.0070390752 | 1.0000000000 |

## Independent matrix and symmetry checks

L=5, J=0.8, g=0.6.

- Parity -1 invariant-subspace residual: 4.441e-16.
- Parity +1 invariant-subspace residual: 4.441e-16.
- Union-of-sector spectral error: 4.441e-15.
- Independent matrix difference: 2.220e-16.

Tests also cover exact limits, complex batches, local bit order, and invalid inputs.
CUDA checks run only when available. Sparse residuals do not certify missed degeneracy multiplicities.
No phase transition or critical exponent is inferred from these small chains.
