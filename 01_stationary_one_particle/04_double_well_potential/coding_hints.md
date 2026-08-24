# Coding Hints

Try the starter files before reading the solutions.

## Suggested functions

```text
make_grid(...) -> grid, spacing
double_well_potential(...) -> potential
build_hamiltonian(...) -> grid, spacing, potential, H
normalize_wavefunctions(...) -> wavefunctions
solve_double_well(...) -> result
tunneling_splitting(result) -> scalar
probability_left(result) -> probabilities
localized_pair(result) -> left_state, right_state
```

Use `float64`; stationary eigenvectors can be chosen real. Store eigenvectors
as columns.

## Important details

- `separation` means the distance from the origin to either minimum; the
  distance between minima is `2 * separation`.
- Evaluate the quartic potential with array operations.
- Do not impose boundary conditions at the minima or barrier.
- Eigenvector signs are arbitrary. Decide which localized combination is left
  by measuring its left-side probability, not by assuming a sign.
- A high barrier can make the splitting smaller than numerical error. Always
  perform convergence checks before interpreting it.

## Tests

Check the potential landmarks, Hermiticity, orthonormality, parity, balanced
stationary probabilities, localized combinations, splitting behavior,
invalid inputs, float64, and NumPy/PyTorch agreement.
