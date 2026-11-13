# Project status — 2026-11-14

## Reviewed state

The starting commit was `778ac1e`, which added the Hubbard dimer and clarified
exchange symmetry. It was pushed to origin/main. The working tree was clean.
The project had 17 chapters. The previous suite passed 291 tests with 10
CUDA checks skipped.

## Completed next step

Added [Phase 4 Chapter 1: Transverse-field Ising chain](04_exact_diagonalization/01_transverse_field_ising/README.md).
The project now has 18 chapters: seven stationary, five time-evolution, five
spin/few-body, and one many-body exact-diagonalization chapter.

- Defined open boundaries, Pauli normalization, and a fixed bit order.
- Added dense PyTorch construction with bit flips.
- Added independent NumPy/SciPy Kronecker-product construction.
- Added sparse low-energy eigenpairs with measured residuals.
- Added exact global spin-flip parity projection.
- Added batched magnetization and pair-correlation measurements.
- Added theory, numerical methods, coding hints, and three starters.
- Added exact-limit, critical-chain, symmetry, and input-validation tests.
- Generated a [validation report](04_exact_diagonalization/01_transverse_field_ising/VALIDATION.md).
- Updated chapter indexes and added Phase 4 to default pytest discovery.

The critical-chain study covers 2 through 12 sites. Ground-energy errors
are below 3e-14. Dense/sparse low-energy differences are below 6e-14 in the
reported comparisons. The maximum reported critical-chain sparse residual
is below 5e-12. The union of both parity spectra matches the full spectrum
within 5e-15 in the reported five-site check.

Finite-size changes are physical effects. They are reported separately from
numerical errors. The tolerance study shows why a small ground-energy error
alone does not establish the accuracy of every returned eigenvector.

## Verification

- Chapter tests: 23 passed, one CUDA check skipped.
- Full suite: 314 passed, 11 CUDA checks skipped.
- Strict mypy: 157 source files passed.
- Ruff lint and formatting checks passed.
- Both solution examples and the validation report generator ran successfully.
- CPU smoke test passed for tensors, float64, eigensolvers, and autograd.
- Git whitespace checks passed.

CUDA is unavailable in the installed CPU build. Dense construction is limited
to 10 sites. The sparse interface is limited to 14 sites; the report checks
through 12 sites. No dependency changes were needed. Commands used uv with
the repository-local cache. This chapter remains uncommitted.

## Next step

Add the Heisenberg spin chain. Derive total Z-magnetization conservation and
construct fixed-magnetization sectors directly. Check the two-site
singlet/triplet spectrum before studying longer chains and correlations.
