# Two-dimensional wave-packet propagation

Extend the 1D FFT method to two spatial axes. Learn `torch.fft.fft2`, rectangular
grids, broadcasting, area normalization, covariance, and batched 2D states.

The main example is a coupled harmonic potential with an xy term. Its rotated
normal modes provide exact reference dynamics. The same propagator accepts
other sampled real potentials; an independent free Gaussian is also included.

## Reading order

1. [Theory and normal-mode derivation](theory.md)
2. [Numerical method and array ordering](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. [PyTorch starter](boilerplates/wave_packet_2d_torch_starter.py) and
   [solution](solutions/wave_packet_2d_torch_solution.py)
5. [NumPy starter](boilerplates/wave_packet_2d_numpy_starter.py) and
   [comparison](solutions/wave_packet_2d_numpy_solution.py)
6. [Convergence starter](boilerplates/wave_packet_2d_convergence_starter.py),
   [experiments](solutions/wave_packet_2d_convergence.py), and [results](VALIDATION.md)

## Run from the repository root

```powershell
uv run python 02_time_evolution/05_two_dimensional_wave_packet/solutions/wave_packet_2d_torch_solution.py
uv run python 02_time_evolution/05_two_dimensional_wave_packet/solutions/wave_packet_2d_numpy_solution.py
uv run python 02_time_evolution/05_two_dimensional_wave_packet/solutions/wave_packet_2d_convergence.py
uv run pytest 02_time_evolution/05_two_dimensional_wave_packet
```

The convergence command writes `VALIDATION.md`. Use `--output workbench/wave2d.md`
for a separate report. This chapter adds no dependencies; NumPy uses its own FFT,
and no new SciPy solver is required.

## Default setup and accuracy

Use `m=hbar=1`, `V=(x**2+0.7*x*y+1.69*y**2)/2`, center `(-2,1)`, and momentum
`(0.4,-0.6)`. The initial covariance comes from the coupled ground state.
The periodic box is `[-10,10) x [-8,8)`, with `(Ny,Nx)=(80,96)`, `dt=0.02`,
and final time 6. History stores every ten steps plus the initial state.

The observed maximum center error is about `1.95e-4`, covariance error `6.30e-5`,
and energy error `1.84e-4`. Halving `dt` reduces each by about four. Norm error
is below `1e-12`; that alone would not reveal the trajectory error.

The quadratic potential's periodic extension differs from the infinite
oscillator at the seams. The default packet remains far from them. The
validation report shows failures in smaller boxes and separates time and grid
errors using both analytical and same-grid references.

## Completion criteria

- Unequal-axis Fourier modes and free anisotropic Gaussian dynamics.
- Coupled center motion, covariance, energy, and full-state comparison.
- Second-order time convergence against a small spectral exponential.
- Batch/single agreement, reversibility, and NumPy agreement.
- Separate grid and fixed-spacing domain studies.

## Next phase

Phase 3 begins with spin-1/2 in a magnetic field. Move from spatial grids to a
two-component state, introducing Pauli matrices and spin precession before
tensor products and coupled spins.
