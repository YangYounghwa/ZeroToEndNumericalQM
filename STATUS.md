# Project status — 2026-11-14

## Reviewed state

The latest commit was `bb57f84` (`feat: add two-particle lattice chapter`).
The repository had 16 chapters: seven stationary, five time-evolution, and
four spin/few-body chapters. The baseline passed 274 tests, with nine CUDA
checks skipped. The next planned subject was a small Hubbard system.

Four theory files already had uncommitted changes. This update did not edit
them: infinite square well, harmonic oscillator, two coupled spins, and two
particles on a lattice.

## Completed next step

Added [Phase 3 Chapter 5](03_spin_and_few_body/05_small_hubbard_system/README.md).
The project now has 17 chapters. This chapter completes the listed Phase 3
systems with a spinful Fermi-Hubbard dimer.

- Derived spin-orbital signs and the complete one-up, one-down sector.
- Added PyTorch Fock-operator projection and an independent NumPy matrix.
- Added exact spectral checks, double occupancy, and batched dynamics.
- Added theory, numerical methods, coding hints, and three starters.
- Added 17 passing tests and one optional CUDA test.
- Generated a reproducible [validation report](03_spin_and_few_body/05_small_hubbard_system/VALIDATION.md).
- Updated the repository and phase indexes and chapter guide.

The measured independent state difference is below 4e-15. The large-U study
separates exchange-approximation error from eigensolver roundoff. The dimer
uses a complete finite sector. It has no continuum-grid or bosonic-cutoff error.

## Verification

- Full suite: 291 passed, 10 skipped because CUDA is unavailable.
- Strict mypy: 149 source files passed.
- Ruff lint and formatting checks passed.
- Both solution examples and the report generator ran successfully.
- CPU smoke test passed for tensors, float64, eigensolvers, and autograd.

The default uv cache was not writable in this session. Setting
`UV_CACHE_DIR` to the ignored repository `.uv-cache` directory resolved it.
No dependency changes were needed. No commit, pull request, or merge was made.

## Next step

Start Phase 4 with the transverse-field Ising chain. Introduce a bit-string
basis, local spin operators, small-chain spectra, and finite-size studies.
Use exact limits before interpreting interacting results. Larger Hubbard
chains, sparse solvers, and bosonic cutoff studies remain later work.
