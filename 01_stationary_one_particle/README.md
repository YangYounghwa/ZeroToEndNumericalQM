# Stationary One-Particle Problems

This subject studies the time-independent Schrödinger equation for one particle.

Learn each calculation in PyTorch first, then use NumPy for comparison. The
SciPy sparse reference begins in Chapter 5 and is reused in Chapters 6 and 7.
Foundations are introduced inside the chapter documents.

| Chapter | Foundations and PyTorch skills introduced |
| --- | --- |
| Infinite well | Shapes, matrix multiplication, eigenvectors, weighted sums, float64 |
| Oscillator | Physical units, finite domains, autograd and basis-expansion extension |
| Finite well | Masks, discontinuities, bound-state selection |
| Double well | Parity, superpositions, energy splitting, domain convergence |
| General potential | Callables, broadcasting, batched eigh, residuals, sparse comparison |
| Radial hydrogen | Reduced radial states, angular momentum, batched sectors |
| 2D | Meshes, Kronecker sums, reshaping, degenerate-subspace comparisons |

## Chapters

1. [Infinite square well](01_infinite_square_well/README.md)
2. [Quantum harmonic oscillator](02_harmonic_oscillator/README.md)
3. [Finite square well](03_finite_square_well/README.md)
4. [Double-well potential](04_double_well_potential/README.md)
5. [General one-dimensional potential](05_general_one_dimensional_potential/README.md)
6. [Radial hydrogen equation](06_radial_hydrogen/README.md)
7. [Two-dimensional stationary problems](07_two_dimensional_stationary/README.md)

## Phase completion

The seven chapters cover finite-difference grids, boundary conditions, dense
and sparse Hamiltonians, normalization, analytical validation, Kronecker sums,
multidimensional indexing, scaling, and convergence. Chapters 1–5 form the 1D
core. Radial hydrogen and 2D are extensions; time evolution can begin after
the core is verified. Complete each selected chapter's tests and experiments.
