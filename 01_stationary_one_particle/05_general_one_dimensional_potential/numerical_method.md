# Numerical Method: General One-Dimensional Potentials

## Uniform interior grid

Use $N$ unknown interior values between $x_{\min}$ and $x_{\max}$. The spacing
and grid points are

$$
h=\frac{x_{\max}-x_{\min}}{N+1},
\qquad
x_i=x_{\min}+ih,
\qquad i=1,\ldots,N.
$$

The omitted endpoints have known value zero. This keeps the matrix dimension
equal to the number of unknown wavefunction values.

## Discrete Hamiltonian

At an interior point,

$$
\psi''(x_i)
=\frac{\psi_{i-1}-2\psi_i+\psi_{i+1}}{h^2}+O(h^2).
$$

The second-derivative matrix is

$$
D_2=\frac{1}{h^2}
\begin{pmatrix}
-2 & 1 & 0 & \cdots & 0\\
1 & -2 & 1 & \ddots & \vdots\\
0 & 1 & -2 & \ddots & 0\\
\vdots & \ddots & \ddots & \ddots & 1\\
0 & \cdots & 0 & 1 & -2
\end{pmatrix}.
$$

Evaluate the caller's potential function once on the grid,

$$
V_i=V(x_i),
$$

then construct

$$
H=-\frac{\hbar^2}{2m}D_2+\operatorname{diag}(V_1,\ldots,V_N).
$$

For real finite $V_i$, this matrix is real symmetric and therefore Hermitian.

## Reusable solver interface

The numerical algorithm should know only that the supplied potential function
maps an array of shape `(N,)` to one real finite value per grid point. It should
reject:

- a scalar or wrong-shaped output;
- complex potential values;
- `NaN` or infinite values;
- invalid domain, mass, or state counts.

This validation keeps failures close to the interface. Otherwise, a malformed
potential may produce a broadcasting error or meaningless eigenvalues much
later in the calculation.

## Diagonalization and normalization

Use a Hermitian eigensolver,

$$
H U=U\operatorname{diag}(E_0,E_1,\ldots),
$$

which returns ordered eigenvalues and eigenvectors as columns. Retain the
lowest requested states and normalize each column with

$$
h\sum_{i=1}^{N}|\psi_{i,n}|^2=1.
$$

Discrete orthonormality then requires

$$
h\Psi^\dagger\Psi=I.
$$

## Residual and observable checks

For each numerical eigenpair, form

$$
\mathbf r_n=H\boldsymbol\psi_n-E_n\boldsymbol\psi_n
$$

and measure

$$
\|\mathbf r_n\|_h
=\sqrt{h\sum_i|r_{i,n}|^2}.
$$

The position expectation is approximated by

$$
\langle x\rangle_n
\approx h\sum_i x_i|\psi_{i,n}|^2.
$$

The shifted harmonic reference should reproduce its analytical energies and
$\langle x\rangle_n=x_c$ as the grid and domain are refined.

## PyTorch batching

If several potentials are sampled on the same grid, stack them into an array
with shape `(B, N)`. Adding each diagonal potential to the shared kinetic
matrix produces Hamiltonians with shape `(B, N, N)`. `torch.linalg.eigh`
accepts this batch directly and returns energies of shape `(B, N)` and
eigenvectors of shape `(B, N, N)`.

Batching reduces Python-level loops and maps naturally to accelerators. It does
not reduce the asymptotic cost of dense diagonalization.

## Accuracy, cost, and failure modes

For a smooth, resolved wavefunction, the centered derivative has $O(h^2)$
discretization error. Total error also includes:

- finite-domain error when the wavefunction reaches the boundaries;
- inadequate resolution of sharp or discontinuous potentials;
- floating-point and eigensolver error;
- modeling error from inappropriate boundary conditions.

Dense Hermitian diagonalization uses $O(N^3)$ time and the stored Hamiltonian
uses $O(N^2)$ memory. A batch of size $B$ uses approximately $O(BN^2)$ memory.
For large grids when only a few low-energy states are needed, a sparse matrix
and iterative eigensolver are more appropriate. That optimization is not added
here because the project does not yet depend on a sparse linear-algebra
library.

The required convergence procedure is:

1. refine $h$ at fixed domain;
2. enlarge the domain at sufficiently fine $h$;
3. check energy stability, orthonormality, and residuals;
4. compare with an analytical result or a trusted refined reference.
