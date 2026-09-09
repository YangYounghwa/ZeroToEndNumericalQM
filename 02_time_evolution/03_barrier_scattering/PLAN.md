# Barrier-scattering chapter plan

Build one chapter after the free and harmonic packet chapters.

1. Derive scattering at a finite rectangular barrier, probability current, and
   the distinction between plane-wave and Gaussian-packet transmission.
2. Implement the PyTorch finite-difference Hamiltonian, reused Crank–Nicolson
   factorization, batched incident packets, and probability diagnostics.
3. Provide a NumPy/SciPy sparse comparison with the same parameters and output
   conventions. Keep both implementations self-contained for learner exercises.
4. Add matching starters and tests for norm/energy conservation, local current,
   reversibility, batching, analytical limits, and actual scattering.
5. Study time-step, grid, measurement-time, and domain convergence separately.
   Align the rectangular barrier edges in grid-refinement experiments.
6. Run project checks and publish the observed results in this chapter's
   validation document. Update the chapter indexes.

Use zero Dirichlet boundaries and observe the scattering before wall returns.
Do not introduce FFT propagation or absorbers in this chapter. Do not interpret
the initial left-side probability as reflection, or all transmitted probability
as sub-barrier tunnelling. Commit only after a new user confirmation.

## Completion evidence

Implemented the chapter, matching starters, and numerical validation report.
The full project check passed with 165 tests passing and one CUDA test skipped
because the installed PyTorch build is CPU-only. Strict mypy checked 93 source
files without errors. Ruff formatting/lint and the CPU smoke test passed.
Both example commands and the Markdown convergence-report command ran.
