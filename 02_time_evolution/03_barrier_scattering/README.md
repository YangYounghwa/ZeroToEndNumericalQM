# Chapter 3: Scattering from a Potential Barrier

Launch a Gaussian packet toward a finite rectangular barrier. Follow its
collision, then estimate reflection and transmission after the outgoing packets
separate. Use PyTorch first and NumPy/SciPy for comparison.

## Learning goals

- Build a discontinuous potential with Boolean masks and `torch.where`.
- Reuse Crank–Nicolson and propagate several incident packets as tensor columns.
- Measure left, near-barrier, and right probabilities with weighted reductions.
- Derive probability current and check a discrete continuity equation.
- Distinguish a plane wave's transmission from a packet's spectrum average.
- Find a measurement window before artificial boundary reflections matter.
- Separate time-step, spatial-grid, domain, and finite-time errors.

## Reading and implementation order

1. [Quantum theory and derivations](theory.md)
2. [Numerical method and error sources](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. [PyTorch starter](boilerplates/barrier_scattering_torch_starter.py)
5. [NumPy/SciPy comparison starter](boilerplates/barrier_scattering_numpy_starter.py)
6. [Convergence starter](boilerplates/barrier_scattering_convergence_starter.py)
7. [PyTorch solution](solutions/barrier_scattering_torch_solution.py)
8. [NumPy/SciPy comparison](solutions/barrier_scattering_numpy_solution.py)
9. [Convergence experiments](solutions/barrier_scattering_convergence.py)
10. [Measured validation results](VALIDATION.md)

Copy starters into an ignored `workbench/` directory before editing them.
The earlier free and harmonic packet chapters provide the required propagation
background; no separate foundations folder is needed.

## Commands

Run from the repository root:

```powershell
uv run python 02_time_evolution/03_barrier_scattering/solutions/barrier_scattering_torch_solution.py
uv run python 02_time_evolution/03_barrier_scattering/solutions/barrier_scattering_numpy_solution.py
uv run pytest 02_time_evolution/03_barrier_scattering
uv run python 02_time_evolution/03_barrier_scattering/solutions/barrier_scattering_convergence.py
```

The convergence command writes `VALIDATION.md` in this chapter. To keep a
separate experiment report, append `--output workbench/barrier-validation.md`.
If the global uv cache is inaccessible, use `uv --cache-dir .uv-cache run ...`.

## Default experiment

| Parameter | Value |
| --- | --- |
| Units | m = hbar = 1 |
| Zero-value walls | x = -40 and +40 |
| Interior grid points | 439; dx = 2/11 |
| Barrier | Height 2.5, width 2, centered at zero |
| Incident packet | x0 = -12, position standard deviation 2, k0 = 2 |
| Propagation | dt = 0.02, 700 steps, final time 14 |
| Stored states | Initial, every tenth step, and final |
| Near-barrier measurement region | [-2, 2] |
| Boundary diagnostics | Probability within 3 units of either wall |

This default grid makes the tensor calculation small enough for CPU learning.
Its final right probability is about 0.09619; the continuum packet reference is
about 0.10645. It is a working example with measurable grid error. Refinement
to 1639 points in the sparse comparison reduces the absolute transmission error
below 0.001. Read the validation report before interpreting the result.

The central momentum corresponds to energy 2, while the packet's mean free
energy is 2.03125. Both are below the barrier, but its momentum distribution
also includes above-barrier components. Total transmitted probability is not
automatically a measurement of sub-barrier tunnelling alone.

## Completion criteria

- PyTorch and NumPy/SciPy agree at identical parameters through the collision.
- Norm and total energy stay constant within numerical precision.
- The three region probabilities sum to the current norm at every saved time.
- Midpoint link currents satisfy the local CN continuity equation.
- The fixed-grid time error approaches second order.
- Aligned grid refinement approaches the packet-averaged continuum result.
- Near-barrier probability is small and right probability has reached a plateau
  in the chosen measurement window.
- Increasing the domain leaves the common-grid state approximately unchanged;
  inspect the boundary history as well as final transmission.
- The analytical transmission formula passes free, threshold, resonance, and
  independent wavefunction-matching checks.

## Next chapter

Study tunnelling with controlled incident energy spread and introduce the
split-operator Fourier method. Its periodic boundaries require a different
boundary analysis from the reflecting walls used here.
