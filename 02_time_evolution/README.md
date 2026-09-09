# Time Evolution

This phase studies the time-dependent Schrödinger equation with complex
wavefunctions and unitary numerical propagation.

PyTorch is the main implementation. Learn complex128 tensors, conjugate inner
products, LU factorization, and batched state columns inside these chapters.
Use the NumPy/SciPy implementation for comparison on identical grids and times.

## Chapters

1. [Free Gaussian wave packet](01_free_gaussian_wave_packet/README.md)
2. [Wave packet in a harmonic potential](02_harmonic_wave_packet/README.md)
3. [Scattering from a potential barrier](03_barrier_scattering/README.md)

## Current methods

- Small dense matrix exponential as a reference
- Sparse Crank-Nicolson propagation
- Reused LU factorizations
- Batched PyTorch propagation
- Norm, energy, reversibility, and time-step checks
- Native PyTorch matrix-exponential references and phase-aligned state errors
- SciPy sparse exponential action, without forming a dense propagator
- Separate PyTorch time-step, spatial-grid, and fixed-spacing domain studies
- Probability-current and left/near/right budgets during barrier scattering
- Packet-averaged continuum transmission and post-collision measurement windows

The present solvers use finite boxes with zero Dirichlet boundaries. These
boundaries reflect packets and preserve norm; they do not absorb outgoing
probability. FFT periodic boundaries and absorbers belong to later methods.

## Next chapter

Tunnelling with controlled incident energy spread and split-operator Fourier
propagation. The FFT method will require checking periodic boundaries and
comparing its results with the existing finite-difference method.
