# Chapter 7: Two-Dimensional Stationary Problems

This chapter provides a general stationary Schrödinger solver for arbitrary
real potentials sampled as `V(x, y)` on a rectangular 2D grid. The main example
contains an `x^2 y^2` interaction and cannot be separated into independent
one-dimensional equations.

## Learning goals

- Store a wavefunction with shape `(Ny, Nx)`.
- Fix a consistent flattening convention between arrays and operators.
- Construct `Iy kron Tx + Ty kron Ix`.
- Normalize with the area element `dx * dy`.
- Solve general real `V(x, y)` potentials with a sparse eigensolver.
- Solve a nonseparable coupled-quartic potential numerically.
- Use a separable oscillator only to validate against analytical energies.
- Compare sparse NumPy with dense and batched PyTorch calculations.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/stationary_2d_numpy_starter.py`
5. `boilerplates/stationary_2d_torch_starter.py`
6. `boilerplates/stationary_2d_convergence_starter.py`
7. `solutions/stationary_2d_numpy_solution.py`
8. `solutions/stationary_2d_torch_solution.py`
9. `solutions/stationary_2d_convergence.py`

Editable copies are in `workbench/`, which is ignored by Git, Ruff, and mypy.

## Commands

```powershell
uv run python 01_stationary_one_particle/07_two_dimensional_stationary/solutions/stationary_2d_numpy_solution.py
uv run python 01_stationary_one_particle/07_two_dimensional_stationary/solutions/stationary_2d_torch_solution.py
uv run python 01_stationary_one_particle/07_two_dimensional_stationary/solutions/stationary_2d_convergence.py
uv run pytest 01_stationary_one_particle/07_two_dimensional_stationary
```

## Completion criteria

- The sparse Hamiltonian is Hermitian.
- Reshaped eigenstates are orthonormal under `dx * dy` integration.
- The nonseparable problem converges toward a refined numerical reference.
- Separable oscillator energies approach their analytical sums.
- The first isotropic excited pair is degenerate.
- Residual norms are small and NumPy agrees with PyTorch.
- Grid refinement reduces the energy error.

## Next step

Extend the same indexing and Kronecker ideas to three dimensions, where sparse
storage becomes essential rather than optional.
