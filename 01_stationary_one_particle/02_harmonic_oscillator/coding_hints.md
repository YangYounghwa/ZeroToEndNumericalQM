# Coding Hints

Try the starter files before reading the solutions.

## Data representation

- Use `float64`; these stationary eigenfunctions can be chosen real.
- Store the interior grid and potential with shape `(num_points,)`.
- Store the Hamiltonian with shape `(num_points, num_points)`.
- Store eigenvectors as columns with shape `(num_points, num_states)`.

## Suggested functions

```text
make_grid(num_points, x_max) -> grid, spacing
harmonic_potential(grid, mass, omega) -> potential
build_hamiltonian(...) -> grid, spacing, potential, H
normalize_wavefunctions(wavefunctions, spacing) -> wavefunctions
solve_harmonic_oscillator(...) -> result
analytical_energies(num_states, omega, hbar) -> energies
expectation_x_power(result, power) -> values
```

## Hamiltonian construction

1. Rebuild the same second-derivative matrix used in Chapter 1.
2. Multiply it by $-\hbar^2/(2m)$.
3. Compute `0.5 * mass * omega**2 * grid**2`.
4. Add that array to the main diagonal with `np.diag` or `torch.diag`.

The potential should not be added to every matrix element.

## Checks while solving

- Use a Hermitian eigensolver.
- Select exactly `num_states` eigenvalues and eigenvector columns.
- Normalize with the grid spacing.
- Compare state index `0` with $E_0=\hbar\omega/2$; unlike the square well,
  the harmonic-oscillator quantum number starts at zero.
- Check parity using the reversed grid vector, allowing for an arbitrary global
  eigenvector sign.

## Convergence experiment

First use a fixed `x_max=8` with point counts such as 50, 100, 200, and 400.
Then test several domain sizes while keeping spacing close to a fixed target.
Report the boundary probability density as a diagnostic for truncation error.

## PyTorch version

- Set `dtype=torch.float64` on every constructed tensor.
- Put the grid, diagonals, Hamiltonian, and results on the selected device.
- Do not convert to NumPy inside the solver.
- Use `.mT` for the transpose of the real wavefunction matrix.

## Common mistakes

- Using `num_points` instead of `num_points + 1` in the spacing denominator.
- Building a grid from `-x_max` to `x_max` that accidentally includes the
  boundary points as unknowns.
- Starting the analytical quantum number at one.
- Assuming a larger domain is always more accurate without also controlling
  the grid spacing.
