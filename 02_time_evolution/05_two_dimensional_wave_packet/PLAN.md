# Two-dimensional wave-packet chapter plan

1. Derive 2D Schrödinger evolution and area-weighted probability. Use rectangular
   grids with explicit `(Ny,Nx,batch)` ordering and FFTs on the two spatial axes.
2. Implement PyTorch split-operator propagation for arbitrary sampled real
   potentials, with a NumPy FFT comparison and no new dependencies.
3. Validate free anisotropic Gaussian spreading and a coupled quadratic potential
   containing an xy term. Derive its rotated normal modes, coherent packet,
   exact center motion, covariance, energy, and state up to a global phase.
4. Test unequal grid sizes/spacings, two-dimensional Fourier modes, separable
   product evolution, batching, reversibility, and a small spectral exponential.
5. Separate time, grid, and domain convergence. Track norm, spectral energy,
   boundary strips, and spatial covariance. Include a deliberately small box.
6. Add theory, numerical method, coding hints, matching starters, and a generated
   validation report. Update indexes and run the full project checks.

The coupled oscillator is nonseparable in x/y but solvable after rotation. Use
that exact solution as a check, while keeping the propagator independent of the
potential. Preserve periodic FFT boundaries; absorbers are outside this chapter.
Leave the new chapter uncommitted for review after the requested Chapter 4 commit.

## Completion evidence

Implemented theory, numerical method, coding hints, matching starters, PyTorch
and NumPy solutions, tests, and the generated validation report. Both examples
and the report generator ran successfully. Full project checks: 188 tests
passed, three CUDA tests skipped on the CPU build, strict mypy passed for 109
source files, Ruff lint/format passed, and local documentation links passed.

The default coupled-state error decreases by about four when dt is halved.
Separate studies demonstrate spatial under-resolution, small-box errors, and
the time-error floor after the grid is resolved. The free anisotropic Gaussian
agrees with the continuum state to about 1.2e-12 at time 3 in its tested box.
