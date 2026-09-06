# Chapter 5: General One-Dimensional Potential

This chapter extracts the repeated finite-difference work from Chapters 1–4
into a reusable stationary Schrödinger solver. The physical potential is now a
function supplied by the caller rather than a formula built into the solver.

## Learning goals

- Separate a numerical algorithm from the potential being studied.
- Solve on a general finite interval rather than only a symmetric domain.
- Validate potential shape, dtype, and finite values at the solver boundary.
- Calculate eigenstate residuals and expectation values.
- Verify the generic solver with a shifted harmonic oscillator.
- Use PyTorch batched eigendecomposition for several potentials on one grid.
- Understand when dense diagonalization is no longer an appropriate method.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/general_potential_torch_starter.py`
5. `boilerplates/general_potential_numpy_starter.py`
6. `boilerplates/general_potential_convergence_starter.py`
7. `solutions/general_potential_torch_solution.py`
8. `solutions/general_potential_numpy_solution.py`
9. `solutions/general_potential_convergence.py`

Copy starter files into `workbench/` before editing. That folder is ignored by
Git, Ruff, and mypy.

## Commands

Run these from the project root:

```powershell
uv run python 01_stationary_one_particle/05_general_one_dimensional_potential/solutions/general_potential_torch_solution.py
uv run python 01_stationary_one_particle/05_general_one_dimensional_potential/solutions/general_potential_numpy_solution.py
uv run python 01_stationary_one_particle/05_general_one_dimensional_potential/solutions/general_potential_convergence.py
uv run pytest 01_stationary_one_particle/05_general_one_dimensional_potential
```

The reference problem uses

$$
V(x)=V_{\mathrm{off}}+\frac12m\omega^2(x-x_c)^2
$$

with $m=\hbar=1$, $\omega=1.25$, $x_c=0.75$, and
$V_{\mathrm{off}}=0.4$.

## Completion criteria

- The Hamiltonian is Hermitian.
- Eigenvectors are discretely orthonormal.
- Eigenpair residuals are small.
- Shifted harmonic-oscillator energies and $\langle x\rangle$ match their
  analytical values.
- Adding a constant to the potential shifts energies without changing states.
- NumPy and PyTorch agree within a defined tolerance.
- Grid refinement reduces the smooth-potential energy error.
- The sparse SciPy comparison agrees with dense PyTorch on the same grid.
- Domain size is varied separately at fixed spacing. Grid and boundary errors
  can partially cancel, so absolute energy error need not decrease monotonically.

## Next step

Use the reusable one-dimensional ideas in the radial hydrogen equation, where
the radial coordinate, origin boundary, centrifugal term, and Coulomb
singularity require additional care.
