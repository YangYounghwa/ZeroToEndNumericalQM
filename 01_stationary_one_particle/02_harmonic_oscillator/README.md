# Chapter 2: Quantum Harmonic Oscillator

This chapter solves the one-dimensional quantum harmonic oscillator with the
same finite-difference structure used for the infinite square well. The new
part is a position-dependent potential and a finite approximation to an
unbounded spatial domain.

## Learning goals

- Add a potential-energy diagonal to a kinetic-energy matrix.
- Approximate an infinite domain with a finite computational box.
- Separate domain-truncation error from grid-discretization error.
- Verify energies, parity, normalization, orthogonality, and expectation values.
- Compare NumPy and PyTorch implementations.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/harmonic_oscillator_numpy_starter.py`
5. `boilerplates/harmonic_oscillator_torch_starter.py`
6. `boilerplates/harmonic_oscillator_convergence_starter.py`
7. `solutions/harmonic_oscillator_numpy_solution.py`
8. `solutions/harmonic_oscillator_torch_solution.py`
9. `solutions/harmonic_oscillator_convergence.py`

Copy starter files into `workbench/` before editing them. This folder is
ignored by Git, Ruff, and mypy.

## Commands

From the repository root:

```powershell
uv run python 01_stationary_one_particle/02_harmonic_oscillator/solutions/harmonic_oscillator_numpy_solution.py
uv run python 01_stationary_one_particle/02_harmonic_oscillator/solutions/harmonic_oscillator_torch_solution.py
uv run python 01_stationary_one_particle/02_harmonic_oscillator/solutions/harmonic_oscillator_convergence.py
uv run pytest
```

## Default units

Examples use

$$
\hbar=m=\omega=1.
$$

The corresponding oscillator length is $a=1$. The default computational
domain is $[-8a,8a]$, where low-energy wavefunctions are already very small.
