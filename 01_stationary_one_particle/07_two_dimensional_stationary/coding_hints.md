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

Do not construct separate `Vx` and `Vy` arrays inside the solver. The callable
must return the complete `(Ny, Nx)` potential, including any coordinate-coupling
terms. A separable potential is one possible input, not part of the interface.

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
- Verify a nonseparable potential against a refined numerical reference.
- Verify degeneracy on equal grids for the isotropic oscillator.
- Compare NumPy and PyTorch on the identical small grid.

## Common mistakes

- Writing `kron(Tx, Iy)` for a C-flattened `(Ny,Nx)` field.
- Supplying a potential shaped `(Nx,Ny)`.
- Accidentally replacing a general `V(x,y)` with `Vx(x) + Vy(y)`.
- Assuming `eigsh` returns sorted eigenvalues.
- Comparing individual eigenvectors inside a degenerate subspace; the solver
  may return any orthonormal rotation of that subspace.

## Comparing degenerate states in PyTorch

Flatten a complete degenerate multiplet to (Ny*Nx, number_of_states), with
columns orthonormal under dx*dy. Compute singular values of
dx*dy * first.mH @ second using subspace_overlaps. Values near one mean the
subspaces agree even if the solvers rotate or phase-shift individual columns.
Keep the same grid and integration weight; do not truncate through a multiplet.
This is also a small introduction to torch.linalg.svdvals.

## Foundations and PyTorch practice

Learn torch.meshgrid(indexing='ij'), torch.kron, reshape, and reductions over two spatial axes. Keep (Ny, Nx) ordering explicit. Use small dense PyTorch grids, then compare with the sparse SciPy implementation for refined calculations; dense storage grows as (Nx*Ny)^2.

Implement the PyTorch starter first, then compare with the NumPy reference.
