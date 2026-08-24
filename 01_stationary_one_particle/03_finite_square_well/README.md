# Chapter 3: Finite Square Well

This chapter replaces the infinite walls with a finite attractive well. Bound
states leak into the classically forbidden region, and the finite computational
box also produces positive-energy states that approximate a continuum.

## Learning goals

- Distinguish physical well edges from computational boundaries.
- Identify bound states using the outside-potential threshold.
- Observe exponential tails and quantum penetration.
- Check numerical energies against transcendental matching equations.
- Understand why a discontinuous potential complicates grid convergence.
- Compare NumPy and PyTorch implementations.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/finite_square_well_numpy_starter.py`
5. `boilerplates/finite_square_well_torch_starter.py`
6. `boilerplates/finite_square_well_convergence_starter.py`
7. `solutions/finite_square_well_numpy_solution.py`
8. `solutions/finite_square_well_torch_solution.py`
9. `solutions/finite_square_well_convergence.py`

Copy starter files into `workbench/` before editing. That folder is ignored by
Git, Ruff, and mypy.

## Commands

```powershell
uv run python 01_stationary_one_particle/03_finite_square_well/solutions/finite_square_well_numpy_solution.py
uv run python 01_stationary_one_particle/03_finite_square_well/solutions/finite_square_well_torch_solution.py
uv run python 01_stationary_one_particle/03_finite_square_well/solutions/finite_square_well_convergence.py
uv run pytest
```

Examples use $\hbar=m=a=1$, depth $V_0=20$, and outer boundaries at
$x=\pm8$.
