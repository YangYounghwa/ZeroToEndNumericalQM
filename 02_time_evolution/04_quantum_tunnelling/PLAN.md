# Quantum-tunnelling chapter plan

1. Use a smooth Gaussian barrier and an incident Gaussian packet. Derive the
   momentum spread and bound the above-barrier part before interpreting transmission.
   Explain why a narrower momentum distribution needs a wider packet and larger box.
2. Introduce a periodic, endpoint-excluded spatial grid, FFT ordering, Parseval
   normalization, and symmetric potential/kinetic/potential splitting in PyTorch.
   Support complex128 state columns and explicit devices without a dense Hamiltonian.
3. Provide a NumPy FFT comparison and a small SciPy sparse Crank-Nicolson reference
   with periodic corner couplings, so both methods have the same boundary condition.
4. Check exact free Fourier-mode evolution, a small spectral matrix exponential,
   norm, reversibility, batching, energy error, and NumPy agreement. Measure left,
   near, right, boundary-strip, and high-frequency probabilities.
5. Run separate time-step, grid, domain, observation-time, and packet-width studies.
   Compare converged FFT transmission with the finite-difference method. Document
   that unitarity and reversibility do not prove physical accuracy.
6. Supply theory, numerical method, coding hints, matching starters, tests, and a
   reproducible Markdown validation report; update the chapter indexes.

The smooth barrier makes Fourier convergence easier to isolate than a rectangular
step. It is a different physical potential from Chapter 3, so its transmission
must not be compared with the rectangular-barrier formula. No new dependencies,
absorbers, notebooks, or separate foundations folder are needed.

## Completion evidence

Implemented the documented solvers, exercises, tests, and reproducible validation
report. Full checks: 178 tests passed, two CUDA tests skipped on the CPU build;
strict mypy passed for 101 source files; Ruff lint/format and the CPU smoke test
passed. Both example scripts and the report generator ran successfully.

The user requested this chapter's commit before proceeding to Chapter 5.
