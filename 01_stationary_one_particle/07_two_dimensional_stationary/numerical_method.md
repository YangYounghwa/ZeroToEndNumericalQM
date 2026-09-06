# Numerical Method: Solving a General 2D Stationary Problem

## 1. Numerical objective

The equation to discretize is

$$
-\frac{\hbar^2}{2m}
\left(
\frac{\partial^2\psi}{\partial x^2}
+\frac{\partial^2\psi}{\partial y^2}
\right)
+V(x,y)\psi
=E\psi.
$$

The numerical method must not assume that

$$
V(x,y)=V_x(x)+V_y(y).
$$

Instead, it samples the complete function $V(x,y)$ on a 2D grid and solves one
matrix eigenvalue problem of dimension $N_xN_y$.

This chapter uses a uniform Cartesian grid and zero Dirichlet boundary
conditions. These choices are simple enough to derive explicitly while still
supporting nonseparable potentials.

## 2. Interior grid and boundary conditions

Let the rectangular domain be

$$
[x_{\min},x_{\max}]\times[y_{\min},y_{\max}].
$$

The code stores $N_x$ interior points along $x$ and $N_y$ interior points
along $y$. The spacings are

$$
\Delta x=\frac{x_{\max}-x_{\min}}{N_x+1},
\qquad
\Delta y=\frac{y_{\max}-y_{\min}}{N_y+1}.
$$

The stored coordinates are

$$
x_i=x_{\min}+(i+1)\Delta x,
\qquad i=0,\ldots,N_x-1,
$$

and

$$
y_j=y_{\min}+(j+1)\Delta y,
\qquad j=0,\ldots,N_y-1.
$$

The four boundary lines are not stored. Their values are known:

$$
\psi=0
\quad\text{at}\quad
x=x_{\min},\ x=x_{\max},\ y=y_{\min},\ y=y_{\max}.
$$

Omitting the boundary points does not mean that the boundary conditions are
ignored. For example, the stencil at the first interior $x$ point contains a
left-neighbor term multiplied by the known boundary value zero. Therefore no
matrix column is needed for that neighbor.

## 3. The five-point finite-difference equation

Write

$$
\psi_{j,i}=\psi(x_i,y_j),
\qquad
V_{j,i}=V(x_i,y_j).
$$

At an interior mesh point, use the centered second derivatives

$$
\frac{\partial^2\psi}{\partial x^2}(x_i,y_j)
\approx
\frac{\psi_{j,i+1}-2\psi_{j,i}+\psi_{j,i-1}}{\Delta x^2},
$$

$$
\frac{\partial^2\psi}{\partial y^2}(x_i,y_j)
\approx
\frac{\psi_{j+1,i}-2\psi_{j,i}+\psi_{j-1,i}}{\Delta y^2}.
$$

Substitution gives the discrete equation

$$
\begin{aligned}
&-t_x\psi_{j,i-1}
-t_x\psi_{j,i+1}
-t_y\psi_{j-1,i}
-t_y\psi_{j+1,i}\\
&\quad+
\left(2t_x+2t_y+V_{j,i}\right)\psi_{j,i}
=E\psi_{j,i},
\end{aligned}
$$

where

$$
t_x=\frac{\hbar^2}{2m\Delta x^2},
\qquad
t_y=\frac{\hbar^2}{2m\Delta y^2}.
$$

Thus each matrix row connects one point to at most four neighbors:

- left and right with coefficient $-t_x$;
- below and above with coefficient $-t_y$;
- itself with coefficient $2t_x+2t_y+V_{j,i}$.

This is the five-point stencil. Points next to a boundary have fewer stored
neighbors because the missing neighbor is a known zero boundary value. The
construction is not periodic: the last point of one row must never connect to
the first point of the next row.

## 4. Array shape and flattening

The sampled wavefunction and potential are stored with shape

```text
(Ny, Nx)
```

so that

```text
psi[j, i] = psi(y_j, x_i)
```

The code uses C-order flattening. The flat index of point $(j,i)$ is

$$
\boxed{p=jN_x+i}.
$$

The inverse mapping is

$$
j=\left\lfloor\frac{p}{N_x}\right\rfloor,
\qquad
i=p\bmod N_x.
$$

For $N_x=4$ and $N_y=3$, the mapping is

```text
          x index i
          0  1  2  3
y j = 0   0  1  2  3
  j = 1   4  5  6  7
  j = 2   8  9 10 11
```

Consequences:

- an $x$ neighbor has flat index $p\pm1$, provided it remains in the same row;
- a $y$ neighbor has flat index $p\pm N_x$;
- `potential.reshape(-1)` must use the same order;
- an eigenvector of length $N_xN_y$ reshapes to `(Ny, Nx)`.

An indexing mismatch can still produce a real symmetric matrix and plausible
energies. Shape tests alone therefore do not prove that the geometry is
correct.

## 5. One-dimensional kinetic matrices

Define the tridiagonal matrices

$$
T_x=t_x
\begin{pmatrix}
2 & -1 & 0 & \cdots & 0\\
-1 & 2 & -1 & \ddots & \vdots\\
0 & -1 & 2 & \ddots & 0\\
\vdots & \ddots & \ddots & \ddots & -1\\
0 & \cdots & 0 & -1 & 2
\end{pmatrix},
$$

and an analogous $T_y$ using $t_y$.

$T_x$ acts along one row while $T_y$ acts along one column. Both already
contain the zero Dirichlet boundary treatment described above.

## 6. Why the Kronecker sum gives the 2D kinetic operator

With the flat ordering $p=jN_x+i$, the kinetic matrix is

$$
\boxed{
T_{2D}=I_y\otimes T_x+T_y\otimes I_x
}.
$$

The two terms have different roles:

- $I_y\otimes T_x$ places one copy of $T_x$ in every $y$ block. It connects
  left and right neighbors without mixing different rows.
- $T_y\otimes I_x$ connects equal $x$ indices in adjacent $y$ blocks. In the
  flat vector, these entries are separated by $N_x$ positions.

The order of the Kronecker products follows from the flattening convention. In
general,

$$
I_x\otimes T_y+T_x\otimes I_y
$$

is not the correct matrix for a C-flattened `(Ny, Nx)` array.

The Kronecker sum is possible because the Cartesian Laplacian is a sum of an
$x$ derivative and a $y$ derivative. It makes no assumption about the
potential.

## 7. Adding a general potential

A local scalar potential acts by multiplication:

$$
(\hat V\psi)_{j,i}=V_{j,i}\psi_{j,i}.
$$

It is therefore diagonal in the position-grid basis:

$$
V_{\mathrm{matrix}}
=\operatorname{diag}\left(V_{\mathrm{flat}}\right).
$$

The full discrete Hamiltonian is

$$
\boxed{
H=I_y\otimes T_x+T_y\otimes I_x
+\operatorname{diag}\left(V_{\mathrm{flat}}\right)
}.
$$

Diagonal does not mean separable. For the coupled potential

$$
V(x,y)=\frac12\left(x^2+1.4^2y^2\right)+0.08x^2y^2,
$$

every diagonal value is calculated from both coordinates. No pair of 1D
potential matrices can reproduce the coupling term.

The solver interface reflects this fact:

```text
potential_function(x_mesh, y_mesh) -> potential with shape (Ny, Nx)
```

The function may represent a separable oscillator, a coupled polynomial, a
rotated well, a disordered landscape, or sampled physical data. The matrix
assembly is unchanged.

## 8. Mesh construction in NumPy and PyTorch

The coordinate meshes must match the `(Ny, Nx)` storage convention.

NumPy uses

```python
x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="xy")
```

PyTorch uses

```python
y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
```

In both cases,

```text
x_mesh.shape == y_mesh.shape == (Ny, Nx)
```

The different assignment order is an API detail. The physical convention is
the same: axis 0 is $y$ and axis 1 is $x$.

The potential must also be checked for the correct shape, real values, and
finite values. A silent broadcast can otherwise create a potential that is
mathematically different from the intended one.

## 9. Sparse structure and memory scaling

Let

$$
M=N_xN_y
$$

be the matrix dimension. A dense real `float64` Hamiltonian requires roughly

$$
8M^2\ \text{bytes}.
$$

Doubling both grid dimensions multiplies $M$ by four and dense matrix memory
by sixteen.

The five-point Hamiltonian has only the following structural nonzero
locations:

$$
M+2N_y(N_x-1)+2N_x(N_y-1)
=5M-2N_x-2N_y.
$$

This grows as $O(M)$ rather than $O(M^2)$. The SciPy solution therefore stores
the Hamiltonian in sparse CSR form and uses sparse Kronecker products.

The PyTorch solution intentionally uses a small dense Hamiltonian. It
demonstrates tensor operations, device selection, and batched potentials, but
it is not the scalable method for a large 2D grid.

## 10. Solving for low-energy eigenstates

The NumPy/SciPy algorithm is:

1. Build the two interior coordinate axes.
2. Create coordinate meshes of shape `(Ny, Nx)`.
3. Evaluate and validate the complete potential.
4. Build sparse $T_x$ and $T_y$.
5. Assemble the sparse Kronecker sum.
6. Add the flattened potential diagonal.
7. Request the lowest `num_states` eigenpairs with `eigsh(..., which="SA")`.
8. Sort the returned energies and reorder the eigenvectors.
9. Apply the 2D integral normalization.
10. Reshape the eigenvectors to `(Ny, Nx, num_states)`.

`which="SA"` means smallest algebraic eigenvalues. This is appropriate for the
lowest states of the real symmetric Hamiltonians used here. With `which="SA"`
and eigenvectors returned, SciPy documents algebraic sorting. The code also
sorts explicitly to keep its interface independent of changes to solver options.

`eigsh` is an iterative Lanczos-type solver. It avoids diagonalizing the full
dense matrix, but its cost still depends on grid size, the number of requested
states, spectral gaps, and the convergence tolerance. Closely spaced or
degenerate states may require more iterations.

The PyTorch version calls `torch.linalg.eigh`, which computes the full dense
spectrum and then selects the first states. Dense diagonalization has roughly
$O(M^3)$ arithmetic cost and $O(M^2)$ memory cost, so its grid must remain
small.

### 10.1 Why this solve is not perturbation theory

For the coupled example, the discrete Hamiltonian can be written as

$$
H_h(\lambda)=H_{0,h}+\lambda W_h,
$$

where $h$ represents the chosen grid and

$$
W_h=\operatorname{diag}(x_i^2y_j^2).
$$

The solver does not first calculate the eigenstates of $H_{0,h}$ and then add
only a first-order energy correction. It passes the complete matrix
$H_h(\lambda)$ to `eigsh` or `eigh`:

$$
H_h(\lambda)c_n=E_{n,h}(\lambda)c_n.
$$

The returned vector $c_n$ is allowed to change completely as $\lambda$
changes. In an eigenbasis of $H_{0,h}$, the coupling has off-diagonal matrix
elements that mix many unperturbed states. Direct diagonalization solves all
of that finite-matrix mixing simultaneously.

The word “diagonal” in `diag(V_flat)` can cause confusion. The potential is
diagonal only in the position-grid basis because it multiplies $\psi(x,y)$ at
each point. It is generally not diagonal in the eigenbasis of the uncoupled
oscillator, which is why it changes both energies and eigenfunctions.

| Question | First-order perturbation | Direct grid diagonalization |
| --- | --- | --- |
| Coupling treatment | Keeps terms through $O(\lambda)$ | Uses the complete finite-grid matrix |
| Requires weak coupling? | Yes | No perturbative smallness assumption |
| Uses uncoupled eigenstates? | Yes, as the expansion basis | No analytical states are required |
| Main limitation | Truncation in powers of $\lambda$ | Grid, domain, and eigensolver errors |
| Result | Approximation to the continuum energy | Eigenpair of the discretized Hamiltonian |

“Nonperturbative” here has a precise but limited meaning: the numerical method
does not truncate a power series in $\lambda$. It does not mean that the
numerical result is free of approximation.

A large $\lambda$ can make the potential and wavefunction vary more sharply.
The method remains valid, but the existing grid may not: convergence must be
checked again as the coupling changes.

### 10.2 Comparing the two methods numerically

Let $E_n^{\mathrm{num}}(\lambda)$ be a converged grid result and define

$$
\Delta E_n^{\mathrm{num}}(\lambda)
=E_n^{\mathrm{num}}(\lambda)-E_n^{\mathrm{num}}(0).
$$

First-order theory predicts

$$
\Delta E_n^{(1)}(\lambda)
=\lambda\langle n^{(0)}|x^2y^2|n^{(0)}\rangle.
$$

A useful comparison procedure is:

1. Converge the domain and grid at $\lambda=0$.
2. Solve the full matrix for several small positive values of $\lambda$.
3. Compare $\Delta E_n^{\mathrm{num}}(\lambda)$ with the first-order line.
4. Reduce $\lambda$ and check that the difference decreases approximately as
   $O(\lambda^2)$.
5. Increase $\lambda$ to observe where the first-order approximation becomes
   inaccurate while the direct solver remains well defined.

This comparison tests perturbation theory. It is not part of the algorithm
used to obtain the numerical eigenstates.

## 11. Normalization, overlaps, and residuals

Matrix eigensolvers normalize a vector using the Euclidean sum

$$
\sum_p|v_p|^2=1.
$$

The wavefunction requires the 2D quadrature normalization

$$
\Delta x\Delta y
\sum_{j=0}^{N_y-1}\sum_{i=0}^{N_x-1}
|\psi_{j,i}|^2=1.
$$

Therefore each returned vector is divided by

$$
\sqrt{
\Delta x\Delta y
\sum_{j,i}|v_{j,i}|^2
}.
$$

The discrete overlap of two states is

$$
\langle\psi_a|\psi_b\rangle_h
=\Delta x\Delta y\sum_{j,i}\psi_{a,j,i}^*\psi_{b,j,i}.
$$

The eigenpair residual is

$$
r_n=H\psi_n-E_n\psi_n,
$$

with the weighted norm

$$
\|r_n\|_h
=\sqrt{\Delta x\Delta y\sum_{j,i}|r_{n,j,i}|^2}.
$$

A small residual shows that the eigensolver accurately solved the discrete
matrix problem. It does not show that the grid and finite domain accurately
represent the continuous problem.

## 12. Discrete observables

For any coordinate function $A(x,y)$, the grid approximation is

$$
\langle A\rangle_n
\approx
\Delta x\Delta y\sum_{j,i}
|\psi_{n,j,i}|^2A(x_i,y_j).
$$

Important checks for the coupled example are

$$
\langle x\rangle,
\quad
\langle y\rangle,
\quad
\langle x^2\rangle,
\quad
\langle y^2\rangle,
\quad
\langle x^2y^2\rangle.
$$

For parity eigenstates, $\langle x\rangle$ and $\langle y\rangle$ should be
near zero. The numerical coupling energy is

$$
\langle V_{\mathrm{coupling}}\rangle
=\lambda\langle x^2y^2\rangle.
$$

## 13. Validation in two layers

### 13.1 Separable analytical calibration

First set the coupling to zero and solve the anisotropic oscillator. Compare
the computed energies with

$$
E_{n_xn_y}
=\hbar\omega_x\left(n_x+\frac12\right)
+\hbar\omega_y\left(n_y+\frac12\right).
$$

Agreement checks the grid spacing, Kronecker ordering, potential flattening,
energy scale, and eigensolver. This example is a test of the general solver,
not an assumption built into it.

### 13.2 Nonseparable numerical validation

For the coupled problem, no exact full spectrum is available. Use independent
checks:

1. Verify $H=H^\dagger$ numerically.
2. Verify weighted normalization and orthogonality.
3. Verify small eigenpair residuals.
4. Check the exact $x$ and $y$ reflection parities.
5. Compare small-grid NumPy and PyTorch results using identical parameters.
6. Refine both spacings and enlarge the domain.
7. Compare weak-coupling energy shifts with perturbation theory.
8. Check that energies rise when a positive $x^2y^2$ coupling is increased.

A fine-grid calculation is a numerical reference, not an exact answer. It is
useful only if its spacings are smaller and its boundaries are sufficiently
farther from the important probability density.

## 14. Separate the main error sources

### Spatial discretization error

For a smooth wavefunction, each centered second derivative is second-order
accurate. The energy error is generally controlled by

$$
O(\Delta x^2)+O(\Delta y^2).
$$

Refining only $x$ eventually leaves the $y$ error unchanged, and vice versa.
An anisotropic problem can require different resolutions along the two axes.

### Domain truncation error

The physical problem may extend over the whole plane, but the computation
forces the wavefunction to zero on a finite rectangle. The domain is large
enough only when the low-state probability density is already negligible near
all four edges.

Grid refinement at fixed boundaries does not remove this error. Domain size
and grid spacing must be tested separately.

### Algebraic eigensolver error

An iterative solver stops at a finite tolerance. Residual norms measure this
error. A tiny residual together with unstable energies under grid refinement
means that the matrix was solved accurately but the continuum problem was not
yet resolved.

### Reference error

When a refined numerical result replaces an exact value, it also has spatial,
domain, and eigensolver errors. Report it as a reference calculation, not as
the exact spectrum.

## 15. Degeneracy and eigenvector comparisons

If several states have the same energy, the eigensolver may return any
orthonormal combination inside that degenerate subspace. Two correct solvers
can therefore produce different-looking eigenvectors.

For a degenerate group, compare:

- the group of energies;
- orthonormality and residuals;
- the projector onto the whole subspace;
- symmetry-adapted combinations, if needed.

Do not require element-by-element equality of individual degenerate
eigenvectors. Even a nondegenerate real eigenvector is defined only up to an
overall sign.

## 16. Common implementation failures

- Storing `(Nx, Ny)` while assembling operators for `(Ny, Nx)`.
- Using the wrong Kronecker-product order.
- Letting $p+1$ connect the last $x$ point of one row to the first point of the
  next row.
- Flattening the potential in a different order from the wavefunction.
- Omitting either $\Delta x$ or $\Delta y$ from normalization.
- Treating a small residual as proof of continuum convergence.
- Refining the grid while leaving the domain too small.
- Using a dense Hamiltonian after $N_xN_y$ has become large.
- Comparing individual vectors inside a degenerate subspace.
- Assuming that a diagonal potential matrix implies a separable potential.

## 17. What this method does and does not cover

The method handles a real local scalar potential on a uniform rectangular
grid with zero Dirichlet boundaries. A potential need not be separable or have
an analytical solution.

It does not directly cover periodic boundaries, irregular domains, adaptive
meshes, magnetic vector potentials, nonlocal potentials, or complex absorbing
potentials. Those cases require changes to the operator or boundary
construction, not merely a different potential function.

For an unconfined or scattering potential, the finite box always produces a
discrete spectrum. Those box states should not automatically be interpreted as
physical bound states.
