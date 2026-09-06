# Zero to End Numerical Quantum Mechanics

This project is a structured path for learning:

- quantum mechanics;
- numerical simulation;
- NumPy;
- PyTorch.

The project uses Python 3.14 through `uv`.

The project starts with small problems that have analytical solutions. It then
progresses toward many-body systems, tensor networks, quantum Monte Carlo, and
neural quantum states.

Only one chapter should be developed at a time. A chapter is complete only when
its numerical results are tested against known results or physical invariants.

## Documents

- [Learning roadmap](ROADMAP.md)
- [Chapter structure and workflow](CHAPTER_GUIDE.md)
- [Chapter 1: Infinite square well](01_stationary_one_particle/01_infinite_square_well/README.md)
- [Chapter 2: Harmonic oscillator](01_stationary_one_particle/02_harmonic_oscillator/README.md)
- [Chapter 3: Finite square well](01_stationary_one_particle/03_finite_square_well/README.md)
- [Chapter 4: Double-well potential](01_stationary_one_particle/04_double_well_potential/README.md)
- [Chapter 5: General one-dimensional potential](01_stationary_one_particle/05_general_one_dimensional_potential/README.md)
- [Chapter 6: Radial hydrogen equation](01_stationary_one_particle/06_radial_hydrogen/README.md)
- [Chapter 7: Two-dimensional stationary problems](01_stationary_one_particle/07_two_dimensional_stationary/README.md)
- [Phase 2: Time evolution](02_time_evolution/README.md)
- [Phase 2, Chapter 1: Free Gaussian wave packet](02_time_evolution/01_free_gaussian_wave_packet/README.md)
- [Phase 2, Chapter 2: Harmonic wave packet](02_time_evolution/02_harmonic_wave_packet/README.md)
- [PyTorch troubleshooting](PYTORCH_TROUBLESHOOTING.md)
- [Code-quality tools](CODE_QUALITY.md)

## Main principles

1. Read the required physics and equation derivation in the chapter's
   `theory.md` before writing the implementation.
2. Build a clear NumPy reference implementation first.
3. Build the PyTorch version after the NumPy result is verified.
4. Use PyTorch for a reason: autograd, optimization, GPU execution, or tensor
   operations.
5. Check convergence and physical invariants. A plausible plot is not enough.
6. Use Python files and Markdown rather than notebooks.
7. Use `uv` for Python environments, dependencies, and commands.

## Current milestone

Phase 1 now ends with a general sparse 2D Cartesian stationary solver,
following the radial hydrogen symmetry reduction. Phase 2 Chapters 1 and 2
introduce Crank-Nicolson time evolution through free and harmonically confined
Gaussian packets. Both are checked against analytical dynamics and a small
matrix-exponential reference.

The next chapter studies scattering from a potential barrier.
