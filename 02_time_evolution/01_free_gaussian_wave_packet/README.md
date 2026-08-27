# Chapter 1: Free Gaussian Wave Packet

This chapter begins time-dependent quantum mechanics. A localized complex
wave packet moves freely, spreads, and is propagated numerically with the
Crank-Nicolson method.

## Learning goals

- Distinguish a time-dependent state from a stationary eigenstate.
- Construct a normalized complex Gaussian packet.
- Derive the exact free-packet center and width.
- Derive the Crank-Nicolson update from the Schrödinger equation.
- Reuse one sparse LU factorization over many time steps.
- Compare against a dense matrix-exponential reference on a small grid.
- Verify norm, energy, time reversibility, and second-order time convergence.
- Use PyTorch to propagate a batch of initial momenta.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/free_gaussian_numpy_starter.py`
5. `boilerplates/free_gaussian_torch_starter.py`
6. `boilerplates/free_gaussian_convergence_starter.py`
7. `solutions/free_gaussian_numpy_solution.py`
8. `solutions/free_gaussian_torch_solution.py`
9. `solutions/free_gaussian_convergence.py`

Editable starter copies are in `workbench/`, which is ignored by Git, Ruff,
and mypy.

## Commands

```powershell
uv run python 02_time_evolution/01_free_gaussian_wave_packet/solutions/free_gaussian_numpy_solution.py
uv run python 02_time_evolution/01_free_gaussian_wave_packet/solutions/free_gaussian_torch_solution.py
uv run python 02_time_evolution/01_free_gaussian_wave_packet/solutions/free_gaussian_convergence.py
uv run pytest 02_time_evolution/01_free_gaussian_wave_packet
```

## Completion criteria

- The Hamiltonian is Hermitian.
- The initial packet has the requested center and width.
- Norm and energy remain constant within numerical precision.
- The packet center and spreading match the analytical free solution.
- Forward evolution followed by reverse evolution recovers the initial state.
- Crank-Nicolson approaches the matrix exponential with second-order time error.
- NumPy and PyTorch results agree on the same grid.

## Next step

Add a harmonic potential. A coherent Gaussian should oscillate without
spreading, providing a stronger test of potential energy and periodic motion.
