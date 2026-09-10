# Quantum tunnelling and FFT propagation

Learn `torch.fft`, complex phase multiplication, broadcasting, and batched
evolution through tunnelling across a smooth Gaussian barrier.

## Reading order

1. [Theory and incident energy spread](theory.md)
2. [Numerical method and periodic boundaries](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. [PyTorch starter](boilerplates/tunnelling_torch_starter.py) and
   [solution](solutions/tunnelling_torch_solution.py)
5. [NumPy starter](boilerplates/tunnelling_numpy_starter.py) and
   [comparison](solutions/tunnelling_numpy_solution.py)
6. [Convergence starter](boilerplates/tunnelling_convergence_starter.py),
   [experiments](solutions/tunnelling_convergence.py), and [results](VALIDATION.md)

## Run from the repository root

```powershell
uv run python 02_time_evolution/04_quantum_tunnelling/solutions/tunnelling_torch_solution.py
uv run python 02_time_evolution/04_quantum_tunnelling/solutions/tunnelling_numpy_solution.py
uv run python 02_time_evolution/04_quantum_tunnelling/solutions/tunnelling_convergence.py
uv run pytest 02_time_evolution/04_quantum_tunnelling
```

The convergence command writes `VALIDATION.md`; use `--output workbench/tunnelling.md`
to keep your own report separately. No additional dependencies are needed.

## Default experiment

Use `m=hbar=1`, periodic box `[-64,64)`, 1024 points, `dt=0.02`, and final time 30.
The packet starts at -20 with position width 3 and wave number 1.5.
The barrier is `2.5*exp(-x**2/(2*0.8**2))`. Its Gaussian width is not the
rectangular full width used in the previous chapter.

At the final time, right probability is about 0.0160309 and near-barrier
probability is about 0.0000661. The incoming above-barrier energy weight is only
0.00000502, so the transmitted probability is predominantly tunnelling.

This is a finite-resolution result. Halving `dt` changes right probability by
about 0.00000192. Norm is conserved near roundoff, while maximum spectral energy
drift is about 0.00000872. The box study finds a state difference of about
0.00000603 between half extents 64 and 80. See the full validation tables before
choosing a tighter accuracy target.

## Completion criteria

- Exact evolution of free Fourier modes, including negative momenta.
- Second-order state convergence against the same-grid spectral exponential.
- NumPy agreement, batch/single agreement, norm, and reversibility checks.
- Energy drift decreases with the time step.
- Grid refinement and independently refined finite-difference transmission.
- Separate incident spectrum, measurement-time, and periodic-domain checks.

## Next chapter

Two-dimensional wave-packet propagation: `torch.fft.fft2`, rectangular grids,
two spatial axes, area-weighted normalization, and nonseparable potentials.
