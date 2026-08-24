# Quantum Theory: Symmetric Double Well

## Potential

Use the smooth quartic potential

$$
V(x)=V_b\left[\left(\frac{x}{d}\right)^2-1\right]^2,
$$

where $V_b>0$ and $d>0$. It has

$$
V(\pm d)=0,
\qquad
V(0)=V_b.
$$

Thus $\pm d$ are the two classical minima and $V_b$ is the central barrier
height. The potential grows as $x^4$, so its physical bound-state
wavefunctions decay at infinity.

## Stationary equation and parity

The equation is

$$
\left[-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x)\right]\psi_n(x)
=E_n\psi_n(x).
$$

Because $V(-x)=V(x)$, the Hamiltonian commutes with parity. Its nondegenerate
one-dimensional eigenstates can be chosen with definite parity. Ordered from
the ground state, they alternate even and odd:

$$
\psi_n(-x)=(-1)^n\psi_n(x).
$$

Consequently, each stationary state's probability density is symmetric and

$$
P(x<0)=P(x>0)=\frac12.
$$

The particle is not permanently located in one well in an energy eigenstate.

## Tunneling splitting

If the wells were completely independent, corresponding left and right states
would have the same energy. Finite barrier penetration couples them. The
lowest combinations are approximately

$$
|0\rangle\approx\frac{|L\rangle+|R\rangle}{\sqrt2},
\qquad
|1\rangle\approx\frac{|L\rangle-|R\rangle}{\sqrt2}.
$$

The even state has slightly lower energy than the odd state. Define the
tunneling splitting

$$
\Delta E=E_1-E_0>0.
$$

A higher or wider barrier reduces overlap between the wells and generally
reduces $\Delta E$.

Conversely, approximate localized states can be constructed from the numerical
parity eigenstates:

$$
|L\rangle\approx\frac{|0\rangle+|1\rangle}{\sqrt2},
\qquad
|R\rangle\approx\frac{|0\rangle-|1\rangle}{\sqrt2}.
$$

An eigenvector has an arbitrary global sign, so code may need to swap these
labels after checking which combination has more probability at $x<0$.

## Tunneling dynamics

If the system begins in the approximate left-localized state, the relative
phase between the two energy eigenstates changes in time. The state transfers
between wells with angular frequency $\Delta E/\hbar$. The probability has
period

$$
T=\frac{2\pi\hbar}{\Delta E}.
$$

This chapter computes the stationary ingredients. Explicit time propagation
belongs in the later time-evolution phase.

The real Hamiltonian permits real stationary eigenvectors, although the
time-dependent superposition acquires complex phases.
