# Numerical Method

## 1. Interior radial grid

Truncate the half-line to `[0, r_max]` and choose `N` unknown interior points:

$$
h=\frac{r_{\max}}{N+1},\qquad r_i=ih,
\quad i=1,\ldots,N.
$$

The omitted endpoints encode `u(0) = u(r_max) = 0`. The first sampled point is
`h`, so the singular origin is never evaluated.

## 2. Finite-difference Hamiltonian

At an interior point,

$$
u''(r_i)=\frac{u_{i+1}-2u_i+u_{i-1}}{h^2}+O(h^2).
$$

Define

$$
t=\frac{\hbar^2}{2mh^2}.
$$

The matrix elements are

$$
H_{ii}=2t+V_{\mathrm{eff}}(r_i),
\qquad H_{i,i\pm1}=-t.
$$

Only the main diagonal and two neighboring diagonals are nonzero. The matrix
is real symmetric and tridiagonal.

## 3. Sparse NumPy/SciPy solve

Storing a dense `N x N` matrix costs `O(N^2)` memory. A tridiagonal sparse
matrix stores only `O(N)` values. `scipy.sparse.linalg.eigsh` uses an iterative
Lanczos-type method for a real symmetric matrix and returns only the requested
low-energy eigenpairs.

Use `which="SA"` for the algebraically smallest eigenvalues. Iterative solvers
do not guarantee sorted output, so sort the returned energies and apply the
same ordering to eigenvectors.

The approximate cost depends on convergence and sparse matrix-vector products;
it is much lower than full `O(N^3)` dense diagonalization when `N` is large and
only a few states are requested.

## 4. PyTorch reference and batching

The PyTorch implementation builds the same operator densely and uses
`torch.linalg.eigh`. This is a transparent cross-check but costs `O(N^2)`
memory and approximately `O(N^3)` work. It is therefore given a smaller default
grid.

PyTorch also constructs several angular-momentum sectors with shapes

```text
effective_potentials: (batch, N)
hamiltonians:         (batch, N, N)
wavefunctions:        (batch, N, num_states)
```

and diagonalizes the batch in one call. This demonstrates tensor batching; it
does not make dense diagonalization preferable for large radial grids.

## 5. Discrete normalization and observables

An eigensolver returns vectors with Euclidean norm one. Rescale every column so

$$
h\sum_i|u_i|^2=1.
$$

Then calculate

$$
\langle r\rangle\approx h\sum_i r_i|u_i|^2.
$$

The eigenpair residual is

$$
\rho_n=
\sqrt{h\sum_i|(Hu_n)_i-E_nu_{n,i}|^2}.
$$

A small residual checks the algebraic solve. It does not measure the error
caused by finite spacing or finite `r_max`.

## 6. Two independent convergence errors

Grid error decreases as `O(h^2)` for a sufficiently resolved state and a fixed,
adequate domain. Domain error comes from forcing the tail to zero at a finite
radius. These effects must be studied separately:

1. Hold `r_max` fixed and increase `N` to test spacing convergence.
2. Hold `h` nearly fixed and increase `r_max` to test domain convergence.

Excited states need larger domains because their tails and mean radii extend
farther outward.

## 7. Failure modes

- Sampling `r = 0` causes division by zero.
- Treating `R(r)` as the discretized unknown gives the wrong kinetic operator
  and normalization measure.
- Requesting too many sparse eigenpairs removes the advantage of a sparse
  method.
- A small residual with a coarse grid can still give an inaccurate energy.
- A small `r_max` can make an excited state look converged under grid refinement
  while it is actually distorted by the outer wall.
- Dense PyTorch grids can exhaust memory long before the sparse NumPy version.
