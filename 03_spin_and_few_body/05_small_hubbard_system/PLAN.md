# Small Hubbard system: plan

1. Introduce four spin orbitals on two open-chain sites.
2. Derive fermion signs and the sector with one up and one down electron.
3. Construct full Fock operators in PyTorch, then project the Hamiltonian.
4. Construct an independent explicit four-state NumPy matrix.
5. Derive the symmetric dimer spectrum, double occupancy, and exchange gap.
6. Check biased dimers, exact evolution, batches, and physical invariants.
7. Measure the large-U approximation error separately from roundoff error.
8. Add lessons, starters, tests, a validation report, and chapter links.
9. Run the full test suite, Ruff, mypy, and CPU smoke test.

Scope: a spinful Fermi-Hubbard dimer with real hopping and site bias.
Use the complete N_up=N_down=1 sector. This sector includes the singlet and
the zero-magnetization triplet. Other triplet components lie outside it.
Use hbar=1. Do not claim a spatial-grid or occupation-cutoff error.
Larger chains and Bose-Hubbard occupation cutoffs belong to Phase 4.
Leave changes uncommitted for review.

## Completion evidence

All planned files and chapter links are present. Both examples and the
validation report generator ran through uv. The chapter passed 17 tests;
one CUDA test was skipped. The full project passed 291 tests with 10 CUDA
skips. Ruff lint and formatting passed. Strict mypy passed 149 source files.
The CPU smoke test passed. See VALIDATION.md for measured numerical errors.
