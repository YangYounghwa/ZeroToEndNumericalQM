# Bell states validation

Generated on CPU with PyTorch 2.13.0+cpu using float64/complex128.
Regenerate from the repository root:

```powershell
uv run python 03_spin_and_few_body/03_bell_states/solutions/bell_states_convergence.py
```

## Joint and local states

| State | Joint purity | Joint S (bits) | Local purity A | S(A) | S(B) | xx | yy | zz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Product 00 | 1.000000 | -0.000000 | 1.000000 | -0.000000 | -0.000000 | 0.000000 | 0.000000 | 1.000000 |
| Phi+ | 1.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 | 1.000000 | -1.000000 | 1.000000 |
| Phi- | 1.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 | -1.000000 | 1.000000 | 1.000000 |
| Psi+ | 1.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | -1.000000 |
| Psi- | 1.000000 | 0.000000 | 0.500000 | 1.000000 | 1.000000 | -1.000000 | -1.000000 | -1.000000 |
| Classical 00/11 mixture | 0.500000 | 1.000000 | 0.500000 | 1.000000 | 1.000000 | 0.000000 | 0.000000 | 1.000000 |

Bell states and the classical mixture have the same local states I/2.
The mixture is explicitly separable; its local entropy of one bit is not an entanglement measure.
Phi+ and the mixture share computational-basis probabilities and zz=1, but differ in xx and yy.

## Schmidt-angle parameter study

State: cos(theta)|00> + exp(0.73i) sin(theta)|11>.
Eigenvalues of either reduced state are cos(theta)^2 and sin(theta)^2.
Changing theta changes the physical state; it is not numerical refinement.

| theta/pi | S(A) | Analytical binary entropy | Absolute error |
| --- | --- | --- | --- |
| 0 | -0.000000000000 | -0.000000000000 | 0.000e+00 |
| 0.0625 | 0.233326628651 | 0.233326628651 | 0.000e+00 |
| 0.125 | 0.600876036693 | 0.600876036693 | 0.000e+00 |
| 0.1875 | 0.891618601858 | 0.891618601858 | 0.000e+00 |
| 0.25 | 1.000000000000 | 1.000000000000 | 0.000e+00 |
| 0.375 | 0.600876036693 | 0.600876036693 | 1.110e-16 |
| 0.5 | 0.000000000000 | -0.000000000000 | 4.039e-31 |

## Connection to exchange dynamics

J=hbar=1, zero fields, initial |01>. Exact entropy is h2(sin(t/2)^2).
Maximum density-matrix entry error against a PyTorch matrix exponential: 2.220e-15.
Global phases cancel in density matrices. These are exact-time samples, not integration steps.

| t/pi | P(10) | S(A) | Analytical entropy |
| --- | --- | --- | --- |
| 0 | 0.000000 | -0.000000 | -0.000000 |
| 0.25 | 0.146447 | 0.600876 | 0.600876 |
| 0.5 | 0.500000 | 1.000000 | 1.000000 |
| 0.75 | 0.853553 | 0.600876 | 0.600876 |
| 1 | 1.000000 | 0.000000 | -0.000000 |
| 1.5 | 0.500000 | 1.000000 | 1.000000 |
| 2 | 0.000000 | 0.000000 | 0.000000 |

The state is maximally entangled halfway to a complete swap (t=pi/2).
At the complete swap t=pi, it is a product state again.

## Small eigenvalues and floating-point error

Density matrix diag(1-p,p). Do not discard small positive eigenvalues using the positivity tolerance.
Zero eigenvalues contribute exactly zero; only negative roundoff within the validation tolerance is clipped.

| p | PyTorch entropy | Scalar reference | Absolute error |
| --- | --- | --- | --- |
| 0 | -0.000000000000e+00 | -0.000000000000e+00 | 0.000e+00 |
| 1e-14 | 4.794853525514e-13 | 4.794853525514e-13 | 0.000e+00 |
| 1e-10 | 3.466197610906e-09 | 3.466197610906e-09 | 0.000e+00 |
| 1e-06 | 2.137426288891e-05 | 2.137426288891e-05 | 0.000e+00 |
| 0.01 | 8.079313589591e-02 | 8.079313589591e-02 | 0.000e+00 |
| 0.5 | 1.000000000000e+00 | 1.000000000000e+00 | 0.000e+00 |

A rotated rank-one projector can acquire eigenvalues of either sign at roundoff scale.
The positivity tolerance (1e-12) permits small numerical defects, but does not repair a physical state.
Entropy near zero should be interpreted with an absolute error tolerance.
No spatial grid, truncated basis, or time integrator is used in this chapter, so there is no dt/dx convergence rate to report.

## Batched NumPy comparison

Seed 2026; batch shape (2,3), five random normalized pure states per mixture.
NumPy partial traces use index loops, independently checking the PyTorch contractions.
Maximum trace error: 4.441e-16; minimum joint eigenvalue: 1.084e-02.

| Quantity | Maximum absolute library difference |
| --- | --- |
| Density matrix | 5.581e-17 |
| Reduced matrix 0 | 1.113e-16 |
| Reduced entropy 0 | 2.220e-16 |
| Reduced matrix 1 | 1.112e-16 |
| Reduced entropy 1 | 2.220e-16 |
| Joint purity | 1.110e-16 |
| Pauli correlations | 1.110e-16 |
