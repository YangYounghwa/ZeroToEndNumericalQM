# Chapter 4: Double-Well Potential

This chapter studies a symmetric quartic double well. It introduces tunneling
splitting: the two lowest stationary states are delocalized parity states with
nearby, but unequal, energies.

## Learning goals

- Interpret a smooth potential with two classical minima.
- Connect symmetry with even and odd stationary states.
- Measure the ground-pair energy splitting.
- Construct approximate left- and right-localized superpositions.
- Study how the splitting changes with barrier height.
- Compare NumPy and PyTorch implementations.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/double_well_torch_starter.py`
5. `boilerplates/double_well_numpy_starter.py`
6. `boilerplates/double_well_convergence_starter.py`
7. `solutions/double_well_torch_solution.py`
8. `solutions/double_well_numpy_solution.py`
9. `solutions/double_well_convergence.py`

Copy starter files into `workbench/` before editing. That folder is ignored by
Git, Ruff, and mypy.

## Commands

```powershell
uv run python 01_stationary_one_particle/04_double_well_potential/solutions/double_well_torch_solution.py
uv run python 01_stationary_one_particle/04_double_well_potential/solutions/double_well_numpy_solution.py
uv run python 01_stationary_one_particle/04_double_well_potential/solutions/double_well_convergence.py
uv run pytest
```

Examples use $\hbar=m=1$, minima at $x=\pm1.5$, barrier height $8$, and outer
boundaries at $x=\pm6$.
