# Quantum Theory: Harmonic Oscillator

## Physical system

A particle of mass $m$ moves in the potential

$$
V(x)=\frac{1}{2}m\omega^2x^2,
$$

where $\omega>0$ is the angular frequency. Unlike the infinite square well,
the physical domain is the entire real line:

$$
-\infty<x<\infty.
$$

A bound-state wavefunction must be square-integrable and must decay at both
ends:

$$
\lim_{x\to\pm\infty}\psi(x)=0.
$$

## Schrödinger equation

The time-independent equation is

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}
+\frac{1}{2}m\omega^2x^2\right]\psi(x)=E\psi(x).
$$

Define the oscillator length

$$
a=\sqrt{\frac{\hbar}{m\omega}}
$$

and the dimensionless coordinate $\xi=x/a$. Since
$d^2/dx^2=(1/a^2)d^2/d\xi^2$, substitution gives

$$
\left[-\frac{1}{2}\frac{d^2}{d\xi^2}
+\frac{1}{2}\xi^2\right]\psi(\xi)
=\frac{E}{\hbar\omega}\psi(\xi).
$$

This shows that $a$ sets the length scale and $\hbar\omega$ sets the energy
scale.

## Why only certain energies are allowed

For large $|\xi|$, the $\xi^2$ term dominates. A normalizable solution must
therefore contain Gaussian decay. Write

$$
\psi(\xi)=e^{-\xi^2/2}f(\xi).
$$

Substitution into the dimensionless Schrödinger equation gives

$$
f''-2\xi f'+(2\epsilon-1)f=0,
\qquad
\epsilon=\frac{E}{\hbar\omega}.
$$

A power-series solution is normalizable only when the series terminates. The
termination condition is

$$
2\epsilon-1=2n,
\qquad n=0,1,2,\ldots
$$

and therefore

$$
E_n=\hbar\omega\left(n+\frac{1}{2}\right).
$$

The terminating polynomials are the Hermite polynomials $H_n$.

## Analytical wavefunctions

The normalized stationary states are

$$
\psi_n(x)=
\frac{1}{\sqrt{2^n n!\sqrt{\pi}\,a}}
H_n\!\left(\frac{x}{a}\right)
e^{-x^2/(2a^2)}.
$$

The first two are

$$
\psi_0(x)=\frac{1}{\pi^{1/4}\sqrt a}e^{-x^2/(2a^2)},
$$

$$
\psi_1(x)=\frac{\sqrt 2\,x}{\pi^{1/4}a^{3/2}}
e^{-x^2/(2a^2)}.
$$

The Hamiltonian is real, so these stationary eigenfunctions can be chosen
real. General superpositions and time-dependent states are usually complex.

## Physical checks

The potential is even, so each eigenstate has definite parity:

$$
\psi_n(-x)=(-1)^n\psi_n(x).
$$

Useful exact expectation values are

$$
\langle x\rangle_n=0,
$$

$$
\langle x^2\rangle_n=
\left(n+\frac{1}{2}\right)\frac{\hbar}{m\omega},
$$

and the virial theorem gives

$$
\langle T\rangle_n=\langle V\rangle_n=\frac{E_n}{2}.
$$

The numerical solution should reproduce these values for states that are well
resolved and negligible near the computational boundaries.
