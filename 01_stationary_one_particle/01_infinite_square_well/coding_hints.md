# Coding Hints

Try the implementation before reading the solution files.

## Data representation

- Use `float64` for this real-valued problem.
- Store the interior grid as an array with shape `(num_points,)`.
- Store the Hamiltonian as `(num_points, num_points)`.
- Store eigenvectors as columns with shape `(num_points, num_states)`.

## Suggested functions

```text
make_grid(num_points, length) -> grid, spacing
build_hamiltonian(num_points, length, mass, hbar) -> grid, spacing, H
normalize_wavefunctions(wavefunctions, spacing) -> normalized wavefunctions
solve_infinite_well(...) -> result
analytical_energies(num_states, length, mass, hbar) -> energies
```

Use a small result container so the grid, spacing, energies, and wavefunctions
remain together.

## Hamiltonian construction

1. Create the main diagonal with value $-2$.
2. Create the two adjacent diagonals with value $1$.
3. Divide by $h^2$ to form the second derivative.
4. Multiply by $-\hbar^2/(2m)$.

Do not add a finite value to represent the infinite walls. Excluding the
boundary values already applies the correct Dirichlet conditions.

## Solving and normalization

Use a symmetric/Hermitian eigensolver. Keep only the first requested states.
For each eigenvector, compute

```text
norm = sqrt(spacing * sum(abs(psi)**2))
psi = psi / norm
```

## PyTorch version

- Set `dtype=torch.float64` explicitly.
- Allow `device="cpu"` or a GPU device.
- Construct every tensor on the selected device.
- Do not convert to NumPy inside the solver.
- Convert only in display or comparison code when necessary.

## Tests to write

- Reject invalid point counts and physical parameters.
- Check `H == H.T` within floating-point tolerance.
- Check the first three energies against the analytical values.
- Check discrete normalization and orthogonality.
- Check that finer grids reduce the ground-state energy error.
- Check NumPy and PyTorch energy agreement.

## Foundations and PyTorch practice

Learn torch.arange, torch.diag, @, torch.linalg.eigh, and reductions on float64 tensors. States have shape (N, S); sum over dim=0 to integrate each state. Use residual_norms to distinguish an accurate matrix eigenpair from an accurate continuum approximation.

Implement the PyTorch starter first, then compare with the NumPy reference.
