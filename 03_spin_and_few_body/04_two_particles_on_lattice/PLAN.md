# Two particles on a lattice: plan

1. Define a finite open chain with real site energies and nearest-neighbor hopping.
2. Derive labeled, symmetric boson, and antisymmetric spin-polarized fermion bases.
3. Derive occupation numbers, boson hopping factors, and fermion signs.
4. Build the PyTorch Hamiltonian by projection from the labeled product basis.
5. Build an independent NumPy reference with occupation-number hopping rules.
6. Evolve normalized states and batches with a matrix exponential or eigendecomposition.
7. Check pair-sum spectra, exchange symmetry, particle number, and two-site dynamics.
8. Report finite-size scaling and roundoff checks. Increasing lattice size changes
   the physical model; output-time sampling is not time-step convergence.
9. Add lessons, matching starters, tests, and a generated validation report.
10. Run chapter examples and all project checks. Update chapter indexes.

Scope: exactly two particles, one orbital per site, no interactions, no periodic
boundary, and no spin dynamics. Fermions share one fixed spin state. The generic
Hamiltonian builders also accept complex Hermitian one-particle matrices for
tests. Small dense matrices keep the projection visible. Add interactions in
the next chapter, after the basis and statistics are validated.

Leave the chapter uncommitted for review.

## Completion evidence

Added theory, numerical methods, coding hints, three starters, independent
PyTorch and NumPy solutions, tests, and a reproducible validation report.
Both examples and the report generator ran successfully through `uv`.

- Chapter: 34 tests passed; 3 CUDA checks skipped on the CPU-only build.
- Full project: 274 tests passed; 9 CUDA checks skipped.
- Strict mypy: 141 source files passed.
- Ruff lint and formatting passed. Formatting also corrected existing Python
  examples in the tracked FFT note; their behavior is unchanged.
- CPU smoke test passed for tensors, float64, eigensolvers, and autograd.

The largest spectral error in the reported finite-chain study is below `8e-15`.
The two-site probability error is below `9e-16`. Complex hopping and batched
state comparisons agree within `2e-15`. The size study approaches the predicted
band-edge limits while retaining the exact finite-size spectra.
