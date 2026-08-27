# Coding Hints

## Required convention

Use arrays shaped `(Ny, Nx)` and flatten with C order. Then construct

```text
kron(Iy, Tx) + kron(Ty, Ix)
```

Reversing either convention silently connects the wrong neighboring points.

## Recommended functions

1. `make_axis`
2. `kinetic_1d`
3. a callable `potential(x_mesh, y_mesh)`
4. `build_hamiltonian`
5. `solve_stationary_2d`
6. observables and residuals

Use `meshgrid(x_grid, y_grid, indexing="xy")` in NumPy. In PyTorch, requesting
`meshgrid(y_grid, x_grid, indexing="ij")` naturally returns arrays shaped
`(Ny, Nx)`.

## Shapes

```text
potential:     (Ny, Nx)
flat vector:   (Ny*Nx,)
wavefunctions: (Ny, Nx, num_states)
```

## Tests

- Check the exact shapes before checking physics.
- Check sparse Hermiticity and orthonormality with `dx*dy`.
- Compare oscillator energies with analytical sums.
- Verify degeneracy on equal grids for the isotropic oscillator.
- Compare NumPy and PyTorch on the identical small grid.

## Common mistakes

- Writing `kron(Tx, Iy)` for a C-flattened `(Ny,Nx)` field.
- Supplying a potential shaped `(Nx,Ny)`.
- Assuming `eigsh` returns sorted eigenvalues.
- Comparing individual eigenvectors inside a degenerate subspace; the solver
  may return any orthonormal rotation of that subspace.
