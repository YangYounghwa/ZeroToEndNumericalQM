# Numerical Method

## 1. Shape and flattening

Store fields as `(Nz, Ny, Nx)` and use C-order flattening:

```text
flat_index = (iz * Ny + iy) * Nx + ix
```

Thus `x` changes fastest, then `y`, then `z`.

## 2. Three-term Kronecker sum

With one-dimensional kinetic matrices `Tx`, `Ty`, and `Tz`, the flattened 3D
kinetic operator is

$$
T_{3D}=I_z\otimes I_y\otimes T_x
+I_z\otimes T_y\otimes I_x
+T_z\otimes I_y\otimes I_x.
$$

Add `diag(V.ravel(order="C"))`. Each term acts along one coordinate while
leaving the other two unchanged.

## 3. Scaling

Let `N` points be used on each axis. The vector dimension is `M=N^3`. A dense
Hamiltonian contains `M^2=N^6` values. At `N=40`, `M=64,000`, and one float64
dense matrix requires about 30.5 GiB before eigensolver workspace.

The seven-point finite-difference stencil has only `O(M)` nonzero entries.
Sparse storage and a low-eigenpair iterative solver are therefore mandatory for
useful 3D grids.

The PyTorch implementation intentionally uses a tiny dense grid to demonstrate
tensor construction and batching. It is a reference, not a scalable 3D solver.

## 4. Normalization and residual

Normalize each state by

$$
\Delta x\Delta y\Delta z
\sum_{i,j,k}|\psi_{k,j,i}|^2=1.
$$

Flatten the state with the same C-order convention before applying `H` in a
residual calculation.

## 5. Convergence

Every axis has second-order finite-difference error. A balanced convergence
study refines all axes. Anisotropic physical length scales may justify unequal
spacings, but the smallest resolved wavefunction structure must have multiple
points in every direction.

Before increasing grid size, calculate both vector dimension and estimated
dense memory. Sparse matrices reduce storage, but eigensolver runtime and
eigenvector storage still grow with `N^3`.
