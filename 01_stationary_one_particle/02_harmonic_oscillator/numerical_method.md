# Numerical Method: Harmonic Oscillator

## Truncating the infinite domain

The physical domain is infinite, but a finite array cannot represent it.
Choose a half-width $x_{\max}$ and solve on

$$
-x_{\max}<x<x_{\max}.
$$

At the artificial endpoints, impose

$$
\psi(-x_{\max})=\psi(x_{\max})=0.
$$

These are not exact physical walls. They approximate decay at infinity. The
approximation is good only when the states being studied are already negligible
near both endpoints.

The classical turning point for state $n$ is

$$
x_{\mathrm{turn},n}
=\sqrt{2n+1}\,a.
$$

The box must extend several decay lengths beyond the largest relevant turning
point.

## Interior grid and boundary conditions

Use $N$ unknown interior values and spacing

$$
h=\frac{2x_{\max}}{N+1}.
$$

The points are

$$
x_i=-x_{\max}+ih,
\qquad i=1,\ldots,N.
$$

The solved vector contains only $\psi_1,\ldots,\psi_N$. The known endpoint
values are excluded. As in Chapter 1, the first and last matrix rows result
from substituting the zero boundary values into the finite-difference formula.
Thus the boundary condition lives jointly in the choice of unknown vector and
the Hamiltonian's endpoint rows.

## Kinetic energy

Use the centered second derivative

$$
D_2=\frac{1}{h^2}
\begin{bmatrix}
-2 & 1 & & 0\\
1 & -2 & \ddots & \\
& \ddots & \ddots & 1\\
0 & & 1 & -2
\end{bmatrix}.
$$

The kinetic-energy matrix is

$$
T=-\frac{\hbar^2}{2m}D_2.
$$

## Potential energy and Hamiltonian

The potential acts point by point:

$$
(V\psi)_i=V(x_i)\psi_i.
$$

It is therefore a diagonal matrix:

$$
V_{ij}=\frac{1}{2}m\omega^2x_i^2\delta_{ij}.
$$

The complete discrete Hamiltonian is

$$
H=T+V.
$$

It is real and symmetric. Use `numpy.linalg.eigh` or `torch.linalg.eigh`, not a
general nonsymmetric eigensolver.

## Normalization and observables

Normalize each eigenvector with

$$
h\sum_i|\psi_{n,i}|^2=1.
$$

For a grid function $A(x)$, compute

$$
\langle A\rangle_n
\approx h\sum_i\psi_{n,i}^*A(x_i)\psi_{n,i}.
$$

For the real solutions in this chapter, the complex conjugate changes nothing,
but keeping it in the formula is important for later chapters.

## Two independent errors

This calculation has two numerical controls:

1. Grid error: at fixed $x_{\max}$, decreasing $h$ reduces the centered
   finite-difference error as $O(h^2)$.
2. Domain error: at fixed resolution, increasing $x_{\max}$ moves the
   artificial boundaries farther into the decaying tails.

Changing $N$ and $x_{\max}$ together can hide which error dominates. Perform
two separate studies:

- hold $x_{\max}$ fixed and increase $N$;
- hold $h$ approximately fixed and increase $x_{\max}$.

When the domain error is already negligible, doubling the grid resolution
should reduce the low-state energy error by about four.

## Computational cost

With a dense $N\times N$ matrix:

- memory is $O(N^2)$;
- full diagonalization is $O(N^3)$.

The matrix is actually tridiagonal. Dense storage is retained here for direct
comparison with Chapter 1; a later general solver should use sparse or
tridiagonal routines for large grids.

## Failure modes

- A box that is too narrow behaves like an unintended finite box and raises the
  energies.
- A box that is very wide at fixed $N$ makes the grid too coarse.
- High-energy states require both a wider box and finer spacing.
- Including endpoints as unknowns without enforcing their values changes the
  boundary-value problem.
- Normalizing with an unweighted Euclidean norm gives the wrong spatial scale.
- Directly comparing eigenvector signs fails because each sign is arbitrary.
- Judging convergence from one grid refinement cannot establish second-order
  behavior.

## Optional extension: variational states and basis truncation

After the grid solver, use the variational starter and solution to optimize
the Gaussian from theory.md. Start with lambda = 0, then use lambda = 0.1.
Compare autograd with dE/dlog(alpha) = (alpha - 1/alpha)/4 -
3 lambda/(2 alpha^2). The optimizer's convergence and the trial family's
approximation error are different questions.

For a second representation, expand in the first K harmonic-oscillator states.
The position operator has X[n,n+1] = X[n+1,n] = sqrt((n+1)/2). Build X with
four extra levels before forming X^4, then retain the leading K by K block:

H_K = diag(n + 1/2) + lambda * (X^4)[:K, :K].

Taking the fourth power after truncating X loses intermediate ladder paths
at the cutoff and produces the wrong projected continuum operator. Check the
diagonal against <n|q^4|n> = 3(2n^2 + 2n + 1)/4.

Increase K at fixed lambda. Rayleigh–Ritz energies decrease toward the continuum
answer for nested bases with the correct matrix elements. Compare with the
grid solver while separately refining dx and checking its domain. Finite-
difference grid energies need not be upper bounds to continuum energies.

Run through uv:

```powershell
uv run python 01_stationary_one_particle/02_harmonic_oscillator/solutions/harmonic_oscillator_variational.py
```

The NumPy basis calculation is a comparison; PyTorch autograd and optimization
are the learning target.
