# Numerical Method: Finite Square Well

## Computational domain and boundary conditions

Replace the physical infinite line by $[-x_{\max},x_{\max}]$ and impose

$$
\psi(-x_{\max})=\psi(x_{\max})=0.
$$

These are artificial outer boundary conditions approximating decay at
infinity. There is no boundary condition $\psi(\pm a)=0$ at the finite-well
edges. The finite-difference equation automatically connects grid points
across those edges.

Keep only $N$ interior unknowns. Their spacing is

$$
h=\frac{2x_{\max}}{N+1},
$$

and their positions are $x_i=-x_{\max}+ih$ for $i=1,\ldots,N$.

The absent endpoint columns encode the outer zero Dirichlet conditions. In the
first row, the missing left neighbor is the known value zero; the last row
works the same way on the right.

## Discrete Hamiltonian

Use the centered second derivative

$$
\psi''(x_i)\approx
\frac{\psi_{i-1}-2\psi_i+\psi_{i+1}}{h^2}.
$$

The kinetic matrix is tridiagonal:

$$
T=-\frac{\hbar^2}{2mh^2}
\begin{bmatrix}
-2&1&&\\
1&-2&1&\\
&\ddots&\ddots&\ddots\\
&&1&-2
\end{bmatrix}.
$$

Evaluate the piecewise potential on the interior grid and construct

$$
H=T+\operatorname{diag}(V).
$$

Solve the real symmetric eigenproblem with a Hermitian eigensolver. Normalize
each eigenvector with $h\sum_i|\psi_i|^2=1$.

## State classification

The outside potential is zero, so classify a computed state as bound when
$E<0$. Request enough low-energy states to include the first positive state;
otherwise a count of all returned negative values does not prove that every
bound state was found.

The probability inside the well is

$$
P_{\mathrm{inside}}\approx
h\sum_{|x_i|<a}|\psi_i|^2.
$$

It is large but not exactly one for bound states because their tails penetrate
outside the well.

## Error sources

- Grid discretization: the centered derivative has local error $O(h^2)$.
- Domain truncation: an outer wall too close distorts exponential tails.
- Discontinuity placement: moving $h$ changes how the step at $|x|=a$ falls
  relative to grid points. This can make errors non-monotone even though the
  derivative stencil is second order in smooth regions.
- Continuum discretization: positive energies move when $x_{\max}$ changes.

For a clean convergence study, vary both resolution and domain size. Compare
negative energies with the matching-equation roots, check parity and
orthonormality, and monitor edge probability density.

Dense diagonalization costs $O(N^3)$ time and $O(N^2)$ memory. A later general
solver should use sparse tridiagonal methods when only a few states are needed.
