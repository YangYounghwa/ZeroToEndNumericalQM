# Chapter 8: Three-Dimensional Stationary Problems

This chapter extends the Cartesian stationary solver to shape `(Nz, Ny, Nx)`.
The main lesson is scaling: sparse Kronecker construction is essential because
the flattened Hilbert space grows as `Nx * Ny * Nz`.

## Learning goals

- Store and normalize a genuine 3D Cartesian wavefunction.
- Construct a three-term Kronecker-sum kinetic operator.
- Keep `x` as the fastest-changing C-order index.
- Solve low states without allocating a dense production Hamiltonian.
- Verify anisotropic oscillator energies and isotropic degeneracy.
- Estimate dense memory before selecting a grid.
- Use PyTorch only as a deliberately small dense and batched reference.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/stationary_3d_numpy_starter.py`
5. `boilerplates/stationary_3d_torch_starter.py`
6. `boilerplates/stationary_3d_convergence_starter.py`
7. `solutions/stationary_3d_numpy_solution.py`
8. `solutions/stationary_3d_torch_solution.py`
9. `solutions/stationary_3d_convergence.py`

Editable copies are in `workbench/`, which is ignored by Git, Ruff, and mypy.

## Commands

```powershell
uv run python 01_stationary_one_particle/08_three_dimensional_stationary/solutions/stationary_3d_numpy_solution.py
uv run python 01_stationary_one_particle/08_three_dimensional_stationary/solutions/stationary_3d_torch_solution.py
uv run python 01_stationary_one_particle/08_three_dimensional_stationary/solutions/stationary_3d_convergence.py
uv run pytest 01_stationary_one_particle/08_three_dimensional_stationary
```

## Completion criteria

- Array, flattening, and Kronecker conventions agree.
- The sparse Hamiltonian is Hermitian.
- States are orthonormal with volume element `dx * dy * dz`.
- Oscillator energies converge toward analytical sums.
- The isotropic first-excited triplet is degenerate.
- Dense memory scaling is calculated before large-grid work.
- NumPy and PyTorch agree on the same small grid.

## Next step

Continue to Phase 2 time evolution. A later 2D propagation chapter will reuse
these multidimensional indexing rules with FFT-based methods.
