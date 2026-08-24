# Numerical Method: Double-Well Potential

## Domain and boundary conditions

Approximate the infinite line with $[-x_{\max},x_{\max}]$ and impose artificial
zero Dirichlet conditions:

$$
\psi(-x_{\max})=\psi(x_{\max})=0.
$$

The well minima $x=\pm d$ and barrier center $x=0$ are ordinary interior
points. No boundary conditions are imposed there.

With $N$ interior unknowns,

$$
h=\frac{2x_{\max}}{N+1},
\qquad
x_i=-x_{\max}+ih.
$$

The missing endpoint columns in the first and last Hamiltonian rows encode the
known zero values outside the unknown vector.

## Hamiltonian and solution

Use the centered difference

$$
\psi''(x_i)\approx
\frac{\psi_{i-1}-2\psi_i+\psi_{i+1}}{h^2}.
$$

Construct

$$
H=-\frac{\hbar^2}{2m}D_2+\operatorname{diag}(V),
$$

then solve with a Hermitian eigensolver and normalize using

$$
h\sum_i|\psi_{i,n}|^2=1.
$$

The potential and symmetric grid preserve parity up to floating-point error.
Check each eigenvector against its reversed copy, allowing for the expected
even or odd sign.

## Observables

The ground-pair splitting is

$$
\Delta E=E_1-E_0.
$$

For each stationary parity state, calculate

$$
P_L\approx h\sum_{x_i<0}|\psi_i|^2.
$$

It should be approximately $1/2$. For a localized combination, it should be
close to one on one side only when the lowest pair is sufficiently isolated
and nearly degenerate.

## Convergence and sensitivity

There is no elementary analytical spectrum for this quartic problem. Use a
high-resolution calculation as a numerical reference, but do not mistake it
for an exact result. Verify:

- Hermiticity and grid-weighted orthonormality;
- alternating parity and $P_L\approx1/2$;
- positive $E_1-E_0$;
- stable energies and splitting under grid and domain refinement;
- decreasing splitting when the barrier is raised while other parameters are
  fixed.

Resolving a very small splitting is harder than resolving either energy. If
the discretization error or roundoff scale is comparable with $\Delta E$, the
computed tunneling time is unreliable.

The centered derivative is second-order accurate for smooth resolved states.
Dense diagonalization costs $O(N^3)$ time and $O(N^2)$ memory.
