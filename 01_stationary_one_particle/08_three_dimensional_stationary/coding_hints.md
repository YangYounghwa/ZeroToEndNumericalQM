# Coding Hints

## Fixed convention

Use shape `(Nz, Ny, Nx)`, C-order flattening, and

```text
kron(kron(Iz, Iy), Tx)
+ kron(kron(Iz, Ty), Ix)
+ kron(kron(Tz, Iy), Ix)
```

Write this convention next to the construction. Three dimensions make an index
ordering error difficult to diagnose from energies alone.

## Mesh construction

Use

```python
z_mesh, y_mesh, x_mesh = meshgrid(z_grid, y_grid, x_grid, indexing="ij")
```

and call the potential as `potential(x_mesh, y_mesh, z_mesh)`. Validate the
returned shape before flattening.

## Memory check

For `M = Nx * Ny * Nz`, a float64 dense Hamiltonian needs

```text
8 * M**2 bytes
```

This excludes temporary arrays and eigensolver workspace. Do not attempt a
large dense calculation merely because the wavefunction vector fits in memory.

## Tests

- Check `(Nz,Ny,Nx)` shapes and Hermiticity.
- Normalize using all three spacing factors.
- Compare low anisotropic-oscillator energies with exact sums.
- Verify the isotropic first-excited triplet.
- Check residuals in flattened space.
- Compare NumPy and PyTorch only on a deliberately small grid.

## Common mistakes

- Confusing physical 3D space with three-particle configuration space.
- Constructing the correct terms in the wrong Kronecker order.
- Allocating a dense matrix before estimating memory.
- Assuming a GPU makes `N^6` dense storage scaling acceptable.
- Comparing individual eigenvectors within a degenerate triplet.
