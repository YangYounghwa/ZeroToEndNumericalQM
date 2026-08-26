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

The stationary points follow from

$$
\frac{dV}{dx}
=\frac{4V_bx}{d^2}\left[\left(\frac{x}{d}\right)^2-1\right]=0.
$$

The second derivative is

$$
\frac{d^2V}{dx^2}
=\frac{4V_b}{d^2}\left[3\left(\frac{x}{d}\right)^2-1\right].
$$

It is negative at $x=0$, confirming that the origin is the top of the
barrier, and positive at $x=\pm d$, confirming that these points are minima.

Near either minimum, write $x=\pm d+\xi$ with $|\xi|\ll d$. A Taylor
expansion gives

$$
V(\pm d+\xi)\approx \frac12 V''(\pm d)\xi^2
=\frac{4V_b}{d^2}\xi^2.
$$

Each isolated well therefore resembles a harmonic oscillator with

$$
\omega_{\mathrm{well}}=\sqrt{\frac{8V_b}{md^2}}.
$$

When a local oscillator energy lies well below $V_b$, it gives a rough energy
scale for a nearly degenerate pair:

$$
E_{2r}\approx E_{2r+1}\approx
\hbar\omega_{\mathrm{well}}\left(r+\frac12\right).
$$

This is only a local approximation. The quartic shape, tunneling, and the
other well shift the actual eigenvalues.

## Dimensionless form

Set

$$
y=\frac{x}{d},
\qquad
\epsilon=\frac{E}{V_b}.
$$

The stationary Schrödinger equation becomes

$$
\left[-\eta\frac{d^2}{dy^2}+(y^2-1)^2\right]\psi(y)
=\epsilon\psi(y),
$$

where

$$
\eta=\frac{\hbar^2}{2md^2V_b}.
$$

Apart from the overall energy scale $V_b$ and length scale $d$, the spectrum
is controlled by $\eta$. A smaller $\eta$ means that kinetic energy is weak
relative to the barrier. This occurs for a heavier particle, a larger well
separation, or a higher barrier, and it generally produces more strongly
localized single-well states and smaller tunneling splittings.

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

The parity ordering can also be understood using the node theorem: the
$n$th one-dimensional bound state has $n$ nodes. The ground state has no node
and is even. The first excited state has one node, which symmetry places at
$x=0$, so it is odd. The pattern continues for higher states.

For an energy $E<V_b$, the central region around $x=0$ is classically
forbidden. The wavefunction is exponentially suppressed there but is not
zero. This nonzero barrier amplitude allows the two wells to communicate.
States with $E\gtrsim V_b$ are still bound because the quartic potential rises
at large $|x|$, but they are no longer naturally interpreted as two weakly
coupled, below-barrier states.

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

This result can be made explicit with a two-state model. Let $|L\rangle$ and
$|R\rangle$ be orthonormal states concentrated in the left and right wells.
For a symmetric system, their diagonal energies are equal. Choosing their
phases so the coupling is $-J$, with $J>0$, gives

$$
H_{\mathrm{two\ state}}=
\begin{pmatrix}
E_{\mathrm w} & -J\\
-J & E_{\mathrm w}
\end{pmatrix}.
$$

Its eigenvectors and eigenvalues are

$$
|0\rangle=\frac{|L\rangle+|R\rangle}{\sqrt2},
\qquad E_0=E_{\mathrm w}-J,
$$

and

$$
|1\rangle=\frac{|L\rangle-|R\rangle}{\sqrt2},
\qquad E_1=E_{\mathrm w}+J.
$$

Therefore,

$$
\Delta E=E_1-E_0=2J.
$$

The even state is lower because it has no node at the barrier center and can
vary more smoothly than the odd state. Less curvature generally means less
kinetic energy.

In a semiclassical description, the small splitting has the approximate
exponential dependence

$$
\Delta E\propto
\exp\left[-\frac{1}{\hbar}
\int_{x_-}^{x_+}\sqrt{2m\,[V(x)-E_{\mathrm{loc}}]}\,dx\right].
$$

Here $x_-$ and $x_+$ are the turning points on the two sides of the central
forbidden region, and $E_{\mathrm{loc}}$ is the relevant single-well energy.
The omitted prefactor depends on the shape of the wells. The formula explains
why the splitting can decrease extremely quickly when the forbidden region
becomes higher or wider.

Conversely, approximate localized states can be constructed from the numerical
parity eigenstates:

$$
|L\rangle\approx\frac{|0\rangle+|1\rangle}{\sqrt2},
\qquad
|R\rangle\approx\frac{|0\rangle-|1\rangle}{\sqrt2}.
$$

An eigenvector has an arbitrary global sign, so code may need to swap these
labels after checking which combination has more probability at $x<0$.

These localized combinations are not exact energy eigenstates. They are most
useful when $E_0$ and $E_1$ form a nearly degenerate pair that is well
separated from higher levels. If the barrier is low, the lowest two states
need not produce sharply localized left and right combinations.

## Tunneling dynamics

If the system begins in the approximate left-localized state, the relative
phase between the two energy eigenstates changes in time. The state transfers
between wells with angular frequency $\Delta E/\hbar$. The probability has
period

$$
T=\frac{2\pi\hbar}{\Delta E}.
$$

More explicitly, start with

$$
|\psi(0)\rangle=|L\rangle
=\frac{|0\rangle+|1\rangle}{\sqrt2}.
$$

Its time evolution is

$$
|\psi(t)\rangle
=e^{-iE_0t/\hbar}
\frac{|0\rangle+e^{-i\Delta E t/\hbar}|1\rangle}{\sqrt2}.
$$

Projecting onto the localized states gives

$$
P_L(t)=\cos^2\left(\frac{\Delta E\,t}{2\hbar}\right),
\qquad
P_R(t)=\sin^2\left(\frac{\Delta E\,t}{2\hbar}\right).
$$

The first complete transfer from left to right occurs at

$$
t_{L\to R}=\frac{\pi\hbar}{\Delta E}=\frac{T}{2}.
$$

Thus a smaller stationary energy splitting corresponds to a longer tunneling
time. This direct relation is why resolving a very small $\Delta E$
accurately is numerically important.

This chapter computes the stationary ingredients. Explicit time propagation
belongs in the later time-evolution phase.

The real Hamiltonian permits real stationary eigenvectors, although the
time-dependent superposition acquires complex phases.
