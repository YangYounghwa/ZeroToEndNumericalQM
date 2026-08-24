# Numerical Method: Finite Differences

## Interior grid

Divide $[0,L]$ into $N+1$ equal intervals. There are $N$ unknown interior
values:

$$
x_i=ih, \qquad i=1,\ldots,N,
$$

with grid spacing

$$
h=\frac{L}{N+1}.
$$

The complete discrete grid contains the two boundary points $x_0=0$ and
$x_{N+1}=L$. The infinite walls impose the Dirichlet boundary conditions

$$
\psi_0=0, \qquad \psi_{N+1}=0.
$$

Only the interior values are unknown, so the vector solved by the eigensolver is

$$
\boldsymbol\psi=
\begin{bmatrix}
\psi_1 & \psi_2 & \cdots & \psi_N
\end{bmatrix}^{\mathsf T}.
$$

The boundary values are not stored in this vector. This choice is part of the
numerical boundary-condition treatment, not merely a memory optimization.

## Second derivative

At an interior point, the centered finite difference is

$$
\frac{d^2\psi}{dx^2}\bigg|_{x_i}
\approx
\frac{\psi_{i-1}-2\psi_i+\psi_{i+1}}{h^2}.
$$

Its truncation error is $O(h^2)$.

The discrete second-derivative matrix is

$$
D_2=\frac{1}{h^2}
\begin{bmatrix}
-2 & 1 & 0 & \cdots & 0\\
1 & -2 & 1 & \ddots & \vdots\\
0 & 1 & -2 & \ddots & 0\\
\vdots & \ddots & \ddots & \ddots & 1\\
0 & \cdots & 0 & 1 & -2
\end{bmatrix}.
$$

## How the boundary condition enters the matrix

For the first interior point, $i=1$, the finite-difference equation is

$$
\frac{d^2\psi}{dx^2}\bigg|_{x_1}
\approx
\frac{\psi_0-2\psi_1+\psi_2}{h^2}.
$$

Applying the left boundary condition $\psi_0=0$ gives

$$
\frac{d^2\psi}{dx^2}\bigg|_{x_1}
\approx
\frac{-2\psi_1+\psi_2}{h^2}.
$$

This becomes the first row of $D_2$:

$$
\frac{1}{h^2}
\begin{bmatrix}
-2 & 1 & 0 & \cdots & 0
\end{bmatrix}.
$$

There is no column for $\psi_0$ because it is a known boundary value equal to
zero.

Similarly, the last interior point, $i=N$, initially contains

$$
\frac{\psi_{N-1}-2\psi_N+\psi_{N+1}}{h^2}.
$$

Applying $\psi_{N+1}=0$ gives

$$
\frac{\psi_{N-1}-2\psi_N}{h^2},
$$

which produces the last row

$$
\frac{1}{h^2}
\begin{bmatrix}
0 & \cdots & 0 & 1 & -2
\end{bmatrix}.
$$

Therefore, the zero boundary conditions are encoded jointly by:

1. excluding $\psi_0$ and $\psi_{N+1}$ from the unknown vector; and
2. constructing the first and last matrix rows after substituting their known
   value, zero.

The differential equation alone does not define a unique problem. It must be
combined with boundary conditions. Different boundary conditions, such as
Neumann or periodic conditions, would change the first and last rows of the
matrix and therefore change the discrete Hamiltonian.

## Discrete Hamiltonian

Inside the well, the potential is zero. Therefore,

$$
H=-\frac{\hbar^2}{2m}D_2.
$$

The matrix is real and symmetric, so it is Hermitian. Use a Hermitian
eigensolver such as `numpy.linalg.eigh` or `torch.linalg.eigh`.

The eigensolver returns

$$
H\boldsymbol\psi_n=E_n\boldsymbol\psi_n.
$$

The eigenvalues are ordered from the ground state upward.

## Discrete normalization

Linear algebra eigensolvers normally return vectors satisfying

$$
\sum_i |v_i|^2=1.
$$

The physical grid normalization is instead

$$
h\sum_i |\psi_i|^2\approx 1.
$$

Therefore, normalize each eigenvector using the discrete integral. For a
uniform grid this is equivalent to dividing the eigensolver vector by
$\sqrt h$.

Eigenvector signs are arbitrary. Both $\psi$ and $-\psi$ represent the same
state, so tests should not require a specific sign.

## Accuracy

The centered second derivative has second-order accuracy. For a fixed low-energy
state, the energy error should behave approximately as

$$
|E_n(h)-E_n|\propto h^2.
$$

Doubling the resolution should reduce the error by roughly a factor of four
once the calculation is in the asymptotic convergence region.

## Computational cost

This chapter uses a dense $N\times N$ Hamiltonian for clarity:

- Hamiltonian memory: $O(N^2)$
- Full dense diagonalization time: $O(N^3)$

Later chapters will introduce sparse matrices and iterative eigensolvers for
larger systems.

## Failure modes

- Including boundary points as unknowns without enforcing their values changes
  the physical problem.
- Reusing this matrix for Neumann or periodic boundaries is incorrect because
  those conditions require different first and last rows.
- Using `eig` instead of a Hermitian eigensolver can introduce avoidable
  numerical noise.
- Forgetting grid-weighted normalization gives incorrectly scaled
  wavefunctions.
- Comparing eigenvector signs directly can report false failures.
- A grid that is too coarse gives inaccurate high-energy states.
