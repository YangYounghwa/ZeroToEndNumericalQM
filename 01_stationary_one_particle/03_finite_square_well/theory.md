# Quantum Theory: Finite Square Well

## Physical system

Use a symmetric attractive well with half-width $a$ and depth $V_0>0$:

$$
V(x)=
\begin{cases}
-V_0, & |x|<a,\\
0, & |x|\ge a.
\end{cases}
$$

The physical domain is the whole real line. The points $x=\pm a$ are changes
in the potential, not hard walls. Therefore, the wavefunction is generally
nonzero there.

## Schrödinger equation and the allowed bound-state range

The stationary equation is

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2}+V(x)\psi=E\psi.
$$

A bound state must have

$$
-V_0<E<0.
$$

The lower inequality follows because $-V_0$ is the minimum value of the
potential. The upper inequality is required for decay in the outside region,
where $V=0$. We now solve the equation separately in the three regions.

## Region I: left of the well

For $x<-a$, the potential is zero. The Schrödinger equation becomes

$$
-\frac{\hbar^2}{2m}\psi''=E\psi.
$$

Because a bound state has $E<0$, define

$$
\kappa=\frac{\sqrt{-2mE}}{\hbar}>0.
$$

Then

$$
\psi''-\kappa^2\psi=0,
$$

whose general solution is

$$
\psi_{\mathrm I}(x)=F e^{\kappa x}+G e^{-\kappa x}.
$$

As $x\to-\infty$, the term $e^{-\kappa x}$ diverges. Normalizability therefore
requires $G=0$:

$$
\psi_{\mathrm I}(x)=F e^{\kappa x}.
$$

## Region II: inside the well

For $-a<x<a$, the potential is $-V_0$. The equation is

$$
-\frac{\hbar^2}{2m}\psi''-V_0\psi=E\psi.
$$

Rearranging gives

$$
\psi''+\frac{2m(E+V_0)}{\hbar^2}\psi=0.
$$

Since $E>-V_0$, define

$$
k=\frac{\sqrt{2m(E+V_0)}}{\hbar}.
$$

The equation is therefore

$$
\psi''+k^2\psi=0,
$$

with general solution

$$
\psi_{\mathrm{II}}(x)=A\cos(kx)+B\sin(kx).
$$

The sine and cosine do not appear as an assumption. They are the two
independent solutions of the constant-coefficient equation inside the well.

## Region III: right of the well

For $x>a$, the equation is again

$$
\psi''-\kappa^2\psi=0,
$$

with general solution

$$
\psi_{\mathrm{III}}(x)=C e^{-\kappa x}+D e^{\kappa x}.
$$

The term $e^{\kappa x}$ diverges as $x\to+\infty$, so normalizability requires
$D=0$:

$$
\psi_{\mathrm{III}}(x)=C e^{-\kappa x}.
$$

Thus the bound state oscillates inside the well and has exponentially decaying
tails outside. A nonzero tail in a classically forbidden region is quantum
penetration.

## Why both matching conditions are required

The potential jumps at $x=\pm a$, but its value remains finite. The
wavefunction must be continuous. If it had a finite jump, its first derivative
would contain a delta distribution and its second derivative would contain a
derivative of a delta. No term in the Schrödinger equation could balance that
singularity for a finite step potential.

To find the condition on the derivative, integrate the Schrödinger equation
across a small interval around $x=a$:

$$
-\frac{\hbar^2}{2m}
\int_{a-\epsilon}^{a+\epsilon}\psi''(x)\,dx
+\int_{a-\epsilon}^{a+\epsilon}V(x)\psi(x)\,dx
=E\int_{a-\epsilon}^{a+\epsilon}\psi(x)\,dx.
$$

The first integral is

$$
\int_{a-\epsilon}^{a+\epsilon}\psi''(x)\,dx
=\psi'(a+\epsilon)-\psi'(a-\epsilon).
$$

Because $V$, $E$, and $\psi$ are finite, the other two integrals approach zero
as $\epsilon\to0$. Therefore,

$$
\psi'(a^+)=\psi'(a^-).
$$

The same argument applies at $x=-a$. The complete matching conditions are

$$
\psi(a^-)=\psi(a^+),
\qquad
\psi'(a^-)=\psi'(a^+).
$$

## Using parity to simplify the piecewise solutions

The potential satisfies $V(-x)=V(x)$, so its stationary states can be chosen
with definite parity.

For an even state,

$$
\psi(-x)=\psi(x).
$$

Inside the well, $\cos(kx)$ is even and $\sin(kx)$ is odd. Therefore, even
parity requires $B=0$:

$$
\psi_{\mathrm{II}}(x)=A\cos(kx).
$$

It is convenient to write the right exterior solution using its value $C_a$ at
$x=a$:

$$
\psi_{\mathrm{III}}(x)=C_a e^{-\kappa(x-a)},
\qquad x>a.
$$

Continuity of the wavefunction at $x=a$ gives

$$
A\cos(ka)=C_a.
$$

The derivatives at the boundary are

$$
\psi'_{\mathrm{II}}(a)=-Ak\sin(ka),
$$

and

$$
\psi'_{\mathrm{III}}(a)=-\kappa C_a.
$$

Derivative continuity therefore gives

$$
-Ak\sin(ka)=-\kappa C_a.
$$

Divide this equation by $A\cos(ka)=C_a$ to remove the unknown amplitudes:

$$
k\tan(ka)=\kappa.
$$

This is the even-state energy condition.

For an odd state,

$$
\psi(-x)=-\psi(x).
$$

Odd parity requires the cosine coefficient to vanish, leaving

$$
\psi_{\mathrm{II}}(x)=B\sin(kx).
$$

Write the right exterior solution again as

$$
\psi_{\mathrm{III}}(x)=C_a e^{-\kappa(x-a)}.
$$

Wavefunction continuity gives

$$
B\sin(ka)=C_a.
$$

Derivative continuity gives

$$
Bk\cos(ka)=-\kappa C_a.
$$

Dividing the derivative equation by the wavefunction equation gives

$$
k\cot(ka)=-\kappa,
$$

or equivalently

$$
-k\cot(ka)=\kappa.
$$

This is the odd-state energy condition. Matching at $x=-a$ gives no new
equation because parity already relates the left and right halves.

## Dimensionless energy equations

Define

$$
z=ka,
\qquad
z_0=\frac{a\sqrt{2mV_0}}{\hbar}.
$$

The definitions of $k$ and $\kappa$ give

$$
k^2+\kappa^2
=\frac{2m(E+V_0)}{\hbar^2}+\frac{-2mE}{\hbar^2}
=\frac{2mV_0}{\hbar^2}.
$$

Multiplying by $a^2$ gives

$$
z^2+(\kappa a)^2=z_0^2,
$$

so

$$
\kappa a=\sqrt{z_0^2-z^2}.
$$

Multiplying the even and odd matching equations by $a$ produces the
dimensionless equations

$$
z\tan z=\sqrt{z_0^2-z^2}
$$

for even states and

$$
-z\cot z=\sqrt{z_0^2-z^2}
$$

for odd states. Once a root is known,

$$
E=-V_0+\frac{\hbar^2z^2}{2ma^2}.
$$

These equations usually require numerical root finding. They are an
independent reference for the finite-difference eigenvalues.

## Bound versus continuum states

For this energy convention, $E<0$ means bound. States with $E\ge0$ belong to
the physical continuum. A finite computational box turns that continuum into
discrete box-dependent energy levels. Those positive levels are not additional
bound states.

As in earlier chapters, the real Hamiltonian allows stationary eigenfunctions
to be chosen real. General time-dependent states can still be complex.
