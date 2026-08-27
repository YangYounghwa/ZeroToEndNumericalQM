# Chapter 2: Wave Packet in a Harmonic Potential

This chapter adds a position-dependent potential to time evolution. A coherent
Gaussian packet follows classical harmonic motion while retaining the width of
the oscillator ground state.

## Learning goals

- Add harmonic potential energy to the finite-difference Hamiltonian.
- Construct a displaced, momentum-boosted coherent state.
- Compare `<x>(t)` with the exact classical oscillator trajectory.
- Verify that a coherent packet does not spread.
- Check norm, energy, periodic return, and time reversibility.
- Reuse Crank-Nicolson and the matrix-exponential reference.
- Batch several initial displacements with PyTorch.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/harmonic_packet_numpy_starter.py`
5. `boilerplates/harmonic_packet_torch_starter.py`
6. `boilerplates/harmonic_packet_convergence_starter.py`
7. `solutions/harmonic_packet_numpy_solution.py`
8. `solutions/harmonic_packet_torch_solution.py`
9. `solutions/harmonic_packet_convergence.py`

Editable starter copies are in `workbench/`, which is ignored by Git, Ruff,
and mypy.

## Commands

```powershell
uv run python 02_time_evolution/02_harmonic_wave_packet/solutions/harmonic_packet_numpy_solution.py
uv run python 02_time_evolution/02_harmonic_wave_packet/solutions/harmonic_packet_torch_solution.py
uv run python 02_time_evolution/02_harmonic_wave_packet/solutions/harmonic_packet_convergence.py
uv run pytest 02_time_evolution/02_harmonic_wave_packet
```

## Completion criteria

- The kinetic-plus-potential Hamiltonian is Hermitian.
- Norm and energy are conserved.
- The packet center follows the exact sinusoidal trajectory.
- The coherent-state width remains nearly constant.
- The state returns after one oscillator period, up to global phase.
- Forward then reverse evolution recovers the initial state.
- Time-step error is second order and NumPy agrees with PyTorch.

## Next step

Chapter 3 introduces scattering from a potential barrier. Reflected and
transmitted probabilities replace simple periodic motion as the main physical
observables.
