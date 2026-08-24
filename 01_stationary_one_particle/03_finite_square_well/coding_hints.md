# Coding Hints

Try the starter files before reading the solutions.

## Suggested functions

```text
make_grid(...) -> grid, spacing
finite_square_well_potential(...) -> potential
build_hamiltonian(...) -> grid, spacing, potential, H
normalize_wavefunctions(...) -> wavefunctions
solve_finite_square_well(...) -> result
bound_state_mask(result) -> mask
probability_inside_well(result, half_width) -> probabilities
analytical_bound_energies(...) -> energies
```

Use `float64`. Store eigenvectors as columns with shape
`(num_points, num_states)`.

## Important details

- Use `abs(grid) < half_width` consistently in NumPy and PyTorch.
- Add the potential only to the Hamiltonian diagonal.
- Use `energy < 0`, not localization probability, as the bound-state rule for
  this energy convention.
- Do not impose zero values at the finite-well edges.
- Use separate pole-free intervals for even and odd transcendental roots.
- Do not compare positive box-state energies with bound-state formulas.

## Tests

Check Hermiticity, grid-weighted orthonormality, alternating parity, agreement
with matching roots, bound-state localization, invalid inputs, float64, and
NumPy/PyTorch agreement.
