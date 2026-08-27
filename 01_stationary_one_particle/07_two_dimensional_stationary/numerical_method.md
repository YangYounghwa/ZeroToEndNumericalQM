# Numerical Method

## 1. Grid and flattening

Use `Nx` interior `x` points and `Ny` interior `y` points. Store sampled fields
with shape `(Ny, Nx)`. C-order flattening makes `x` the fastest-changing index:

```text
flat_index = iy * Nx + ix
```

Every operator construction and reshape must use this same convention.

## 2. Kronecker-sum kinetic operator

Let `Tx` and `Ty` be the usual one-dimensional finite-difference kinetic
matrices. The 2D kinetic matrix is

$$
T_{2D}=I_y\otimes T_x+T_y\otimes I_x.
$$

The first term differentiates along `x` without mixing rows. The second term
differentiates along `y` without mixing columns. Add the flattened potential:

$$
H=T_{2D}+\operatorname{diag}(V_{\mathrm{flat}}).
$$

## 3. Sparse scaling

The Hilbert-space dimension is `M = Nx * Ny`. A dense Hamiltonian stores
`O(M^2) = O(Nx^2 Ny^2)` values. The five-point finite-difference stencil has
only `O(M)` nonzero entries. The NumPy implementation therefore uses sparse
Kronecker products and `eigsh` to request only low states.

PyTorch uses a small dense matrix as a learning and batching reference. It is
not the recommended large-grid representation.

## 4. Normalization and residuals

For a sampled state,

$$
\Delta x\Delta y\sum_{i,j}|\psi_{j,i}|^2=1.
$$

Flattening does not change the order of the sum. Residuals are calculated in
flat operator space and weighted by the area element.

## 5. Accuracy and failure modes

Each second derivative has `O(dx^2)` or `O(dy^2)` error. Both axes must be
refined; improving only one eventually leaves the other as the dominant error.

Common failures are swapping `(x,y)` and `(y,x)` shapes, using a reshape order
inconsistent with the Kronecker sum, dense allocation at an impractical grid
size, and normalizing with only one spacing factor.
