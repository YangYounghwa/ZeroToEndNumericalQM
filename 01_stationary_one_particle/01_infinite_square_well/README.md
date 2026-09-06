# Chapter 1: Infinite Square Well

This chapter solves the one-dimensional infinite square well with a
finite-difference Hamiltonian.

## Learning goals

- Understand the stationary Schrödinger equation and its analytical solution.
- Turn a second derivative into a matrix.
- Solve a Hermitian eigenvalue problem.
- Normalize wavefunctions on a discrete grid.
- Verify numerical convergence.
- Implement the same calculation with NumPy and PyTorch.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/infinite_square_well_torch_starter.py`
5. `boilerplates/infinite_square_well_numpy_starter.py`
6. `boilerplates/infinite_square_well_convergence_starter.py`
7. `solutions/infinite_square_well_torch_solution.py`
8. `solutions/infinite_square_well_numpy_solution.py`
9. `solutions/infinite_square_well_convergence.py`

Copy starter files into a local `workbench/` folder before editing them. The
`workbench/` folder is ignored by Git, Ruff, and mypy so personal work does not
change the reusable learner materials.

## Commands

From the repository root:

```powershell
uv sync
uv run python torch_smoke_test.py
uv run pytest
uv run python 01_stationary_one_particle/01_infinite_square_well/solutions/infinite_square_well_torch_solution.py
uv run python 01_stationary_one_particle/01_infinite_square_well/solutions/infinite_square_well_numpy_solution.py
uv run python 01_stationary_one_particle/01_infinite_square_well/solutions/infinite_square_well_convergence.py
```

See [PyTorch troubleshooting](../../PYTORCH_TROUBLESHOOTING.md) if the smoke
test fails. A skipped CUDA check is acceptable because this chapter supports
CPU execution.

## Default units

Examples use dimensionless values

$$
\hbar = m = L = 1.
$$

The implementation keeps these values as parameters, so other consistent units
can also be used.
