# Transverse-field Ising chain: plan

1. Define an open chain with H = -J sum Z_i Z_(i+1) - g sum X_i.
2. Fix the bit order and distinguish Pauli matrices from spin operators.
3. Construct a dense PyTorch Hamiltonian with bit flips.
4. Construct an independent NumPy/SciPy reference with Kronecker products.
5. Introduce global spin-flip parity and exact sector projection.
6. Measure magnetization, pair correlations, and the low-energy gap.
7. Check zero-coupling limits, the two-site spectrum, and critical-chain formulas.
8. Separate finite-size physics, dense roundoff, and sparse solver residuals.
9. Add lessons, three starters, tests, and a reproducible validation report.
10. Update project indexes and run the full quality checks.

Scope: real uniform J and g, open boundaries, spin one-half, and no
longitudinal field. Use dense PyTorch only for small chains. Use SciPy sparse
eigensolvers for a larger comparison. No time evolution or phase-diagram fit
is required. Do not treat finite-system symmetry as spontaneous symmetry
breaking. Keep the chapter uncommitted until the user approves a commit.

## Completion evidence

Added all planned lesson, solution, starter, and test files. Both examples
and the validation generator ran through uv. The chapter passed 23 tests
with one CUDA skip. The full suite passed 314 tests with 11 CUDA skips.
Strict mypy passed 157 source files. Ruff lint and formatting passed. The
CPU smoke test passed. See VALIDATION.md for measured results and limits.
