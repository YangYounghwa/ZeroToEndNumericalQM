# Quantum Theory: General One-Dimensional Potentials

## Physical system

Consider one nonrelativistic particle of mass $m$ moving in a real,
time-independent potential $V(x)$. Its stationary states satisfy

$$
H\psi_n(x)=E_n\psi_n(x),
$$

with

$$
H=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x).
$$

The kinetic operator is common to every problem in this phase. The potential
distinguishes the infinite well, harmonic oscillator, finite well, double well,
and any new one-dimensional system. A reusable solver should therefore accept
$V(x)$ as input instead of embedding one particular formula.

## Assumptions and scope

This chapter assumes that:

- $V(x)$ is real and time independent;
- the particle has a constant positive mass;
- the requested states can be represented accurately on a finite interval;
- zero Dirichlet boundary conditions are an acceptable approximation;
- a uniform grid resolves the important variation in both $V$ and $\psi$.

A real potential makes the Hamiltonian Hermitian under suitable boundary
conditions. Consequently, its eigenvalues are real, eigenstates belonging to
different eigenvalues are orthogonal, and stationary eigenfunctions can be
chosen real.

Complex absorbing potentials, position-dependent masses, periodic boundaries,
and singular potentials need modified operators or boundary conditions and are
outside this chapter's solver interface.

## Boundary conditions and the finite box

The physical problem may live on the whole real line. Numerically, choose a
finite interval $[x_{\min},x_{\max}]$ and impose

$$
\psi(x_{\min})=\psi(x_{\max})=0.
$$

These are artificial boundaries unless the physical system actually contains
hard walls. For a localized bound state, the approximation is accurate when
the wavefunction is already negligible at both ends. The result must be tested
by moving the boundaries outward.

For a nonconfining potential, the finite box can turn continuum states into a
discrete sequence. Such box-dependent levels should not automatically be
interpreted as physical bound states.

## Energy expectation and variational meaning

For a normalized state,

$$
\int_{x_{\min}}^{x_{\max}}|\psi(x)|^2\,dx=1,
$$

the energy expectation is

$$
\langle H\rangle
=\int \psi^*(x)
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x)\right]
\psi(x)\,dx.
$$

An exact stationary state has zero eigenpair residual,

$$
r_n(x)=H\psi_n(x)-E_n\psi_n(x)=0.
$$

After discretization, the residual norm is a direct algebraic check that the
reported vector and eigenvalue satisfy the matrix eigenproblem. A small
residual does not prove that the spatial grid or finite domain represents the
continuous problem accurately; convergence tests are still required.

## General observables

Once the wavefunction is normalized, any position-dependent observable
$A(x)$ has expectation value

$$
\langle A\rangle_n
=\int \psi_n^*(x)A(x)\psi_n(x)\,dx.
$$

For example,

$$
\langle x\rangle_n
=\int x|\psi_n(x)|^2\,dx.
$$

Unlike the symmetric examples in earlier chapters, a general potential need
not have parity symmetry, so $\langle x\rangle$ need not vanish.

## Analytical reference: shifted harmonic oscillator

Use

$$
V(x)=V_{\mathrm{off}}+\frac12m\omega^2(x-x_c)^2.
$$

Define the shifted coordinate

$$
y=x-x_c.
$$

Because $d/dx=d/dy$, the Schrödinger equation becomes

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dy^2}
+\frac12m\omega^2y^2\right]\psi_n(y)
=(E_n-V_{\mathrm{off}})\psi_n(y).
$$

This is the ordinary harmonic oscillator plus a constant energy shift.
Therefore,

$$
E_n=V_{\mathrm{off}}+\hbar\omega\left(n+\frac12\right),
\qquad n=0,1,2,\ldots
$$

and every stationary state is centered at

$$
\langle x\rangle_n=x_c.
$$

The reference tests three independent behaviors:

1. changing the spatial center translates the wavefunctions;
2. adding $V_{\mathrm{off}}$ shifts every energy by the same amount;
3. the level spacing remains $\hbar\omega$.

The energy-zero statement is general. If

$$
\widetilde V(x)=V(x)+C,
$$

then

$$
\widetilde H=H+C I.
$$

Thus the eigenfunctions are unchanged and

$$
\widetilde E_n=E_n+C.
$$

This is a useful test for every stationary solver.
