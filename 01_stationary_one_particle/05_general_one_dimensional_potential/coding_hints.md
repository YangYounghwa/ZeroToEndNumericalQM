# Coding Hints

Try the starter files before reading the solutions.

## Suggested NumPy functions

```text
make_grid(...) -> grid, spacing
kinetic_energy_matrix(...) -> kinetic
evaluate_potential(grid, potential_function) -> potential
build_hamiltonian(potential_function, ...) -> grid, spacing, potential, H
normalize_wavefunctions(...) -> wavefunctions
solve_stationary(potential_function, ...) -> result
expectation_position(result) -> expectations
residual_norms(result) -> residuals
```

Represent a potential as a function receiving the whole grid:

```python
def quartic_potential(grid: FloatArray) -> FloatArray:
    return 0.25 * grid**4
```

Do not loop over individual grid points. Check that the returned array has
shape `(num_points,)`, contains real values, and contains no `NaN` or infinity.

## Array shapes

```text
grid:              (N,)
potential:         (N,)
hamiltonian:       (N, N)
energies:          (S,)
wavefunctions:     (N, S)
batched potentials (PyTorch):     (B, N)
batched Hamiltonians (PyTorch):   (B, N, N)
batched wavefunctions (PyTorch):  (B, N, S)
```

Store eigenstates as columns. In the PyTorch batch, the final axis still
selects the eigenstate.

## Pseudocode

```text
validate scalar inputs
construct the interior grid
construct the kinetic matrix
evaluate and validate V(grid)
H = kinetic + diagonal(V)
energies, vectors = Hermitian eigensolver(H)
select the lowest requested columns
normalize columns using the grid spacing
return grid, potential, eigenpairs, and H
```

## Important details

- Use `float64` in both implementations.
- A general interval does not need to be symmetric about zero.
- The potential function describes physics; the outer zero boundaries describe
  the finite numerical approximation.
- A small matrix residual checks the eigensolver result, not continuous-space
  convergence.
- Eigenvector signs are arbitrary. Compare absolute values or align signs when
  comparing NumPy and PyTorch wavefunctions.
- Do not call `.numpy()` inside the PyTorch solver. Convert only in tests or at
  a clear reporting boundary.

## Tests

Check input validation, Hermiticity, `float64`, orthonormality, residuals, the
shifted harmonic spectrum, translated position expectation values, constant
energy shifts, NumPy/PyTorch agreement, and batched/individual agreement.
