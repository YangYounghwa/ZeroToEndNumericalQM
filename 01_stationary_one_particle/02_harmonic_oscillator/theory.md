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

## Spatial parity and the Hamiltonian

Spatial parity reflects position about the origin. Use $\Pi$ for this operator:

$$
(\Pi\psi)(x)=\psi(-x),\qquad \Pi^2=I.
$$

Reflection preserves the inner product, so $\Pi$ is unitary. Since
$\Pi^{-1}=\Pi$, it is also Hermitian. If $\Pi\psi=p\psi$, applying parity
twice gives $p^2=1$. Its eigenvalues are therefore $+1$ and $-1$:

$$
\psi(-x)=+\psi(x)\quad\text{(even)},\qquad
\psi(-x)=-\psi(x)\quad\text{(odd)}.
$$

For $H=-\hbar^2\partial_x^2/(2m)+V(x)$, compare the two orders of operation:

$$
(H\Pi\psi)(x)=-\frac{\hbar^2}{2m}\psi''(-x)+V(x)\psi(-x),
$$

$$
(\Pi H\psi)(x)=-\frac{\hbar^2}{2m}\psi''(-x)+V(-x)\psi(-x).
$$

The second derivative is unchanged by reflection. For an even potential and
reflection-invariant boundary conditions, the expressions agree:

$$
[H,\Pi]\equiv H\Pi-\Pi H=0.
$$

This does not mean parity is an extra energy term. It means reflecting a
state before applying H gives the same result as reflecting afterward.
If $H\psi=E\psi$, then $H(\Pi\psi)=E(\Pi\psi)$. The oscillator levels are
nondegenerate, so $\Pi\psi$ must be proportional to $\psi$. Each energy
eigenstate therefore has definite parity. With degeneracy, one can choose a
joint eigenbasis, but an arbitrary eigenvector need not have definite parity.

The oscillator eigenstates obey

$$
\psi_n(-x)=(-1)^n\psi_n(x).
$$

The ground state is even; the first excited state is odd. A superposition
such as $(\psi_0+\psi_1)/\sqrt2$ has no definite parity. The symmetric
Hamiltonian allows both parities and their superpositions.

This differs from the [center reflection of the infinite well](../01_infinite_square_well/theory.md),
which sends $x$ to $L-x$ in that chapter's coordinates. It also differs from
particle exchange: parity reflects space; exchange swaps two particle labels.
An even one-particle wavefunction is not a boson, and an odd one is not a
fermion. The [two-particle lattice lesson](../../03_spin_and_few_body/04_two_particles_on_lattice/theory.md)
explains the exchange rule for identical particles.

## Physical checks

For a parity eigenstate, $|\psi(x)|^2$ is even. Thus
$x|\psi(x)|^2$ is odd, and its integral over the symmetric domain vanishes.
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

## Oscillator units and an early variational exercise

Choose a = sqrt(hbar/(m omega)), q = x/a, and epsilon = E/(hbar omega).
The dimensionless harmonic Hamiltonian is

$$H/(\hbar\omega)=\frac12\left(-\frac{d^2}{dq^2}+q^2\right).$$

Normalize phi(q) = sqrt(a) psi(aq). The familiar m = hbar = omega = 1 examples
are calculations in these oscillator units, not a claim that physical
constants have no units.

Use the normalized trial state
phi_alpha(q) = (alpha/pi)^(1/4) exp(-alpha q^2/2), with alpha > 0.
Its moments are <q^2> = 1/(2 alpha), <p^2> = alpha/2, and
<q^4> = 3/(4 alpha^2), where p is the dimensionless momentum operator.
For H = (p^2 + q^2)/2 + lambda q^4, its continuum energy is

$$E(\alpha)=\frac{\alpha+\alpha^{-1}}4+\frac{3\lambda}{4\alpha^2}.$$

For lambda = 0, the derivative vanishes at alpha = 1 and the minimum is 1/2.
The variational principle says a normalized trial state's energy is at least
the true ground energy. A restricted trial family can still miss the exact
ground state when lambda is nonzero. Learn autograd by differentiating this
energy and optimizing log(alpha); positivity then holds automatically.
