# Quantum Theory

## 1. Time-dependent Schrödinger equation

For a free particle in one dimension,

$$
i\hbar\frac{\partial\psi(x,t)}{\partial t}
=-\frac{\hbar^2}{2m}\frac{\partial^2\psi(x,t)}{\partial x^2}.
$$

Unlike a stationary eigenfunction, a localized packet contains many momentum
and energy components. Its probability density changes with time.

The wavefunction is a probability amplitude, not a probability itself. The
probability of finding the particle in a small interval is approximately
$|\psi(x,t)|^2dx$. Time evolution changes the amplitudes and their relative
phases; interference between components then changes the density. A single
energy eigenstate only acquires a global phase, which cancels in its density.

## 2. Initial Gaussian packet

Use

$$
\psi(x,0)=\frac{1}{(2\pi\sigma^2)^{1/4}}
\exp\left[-\frac{(x-x_0)^2}{4\sigma^2}
+ik_0(x-x_0)\right].
$$

Then `|psi|^2` is a normal distribution with

$$
\langle x\rangle_0=x_0,
\qquad \Delta x_0=\sigma,
\qquad \langle p\rangle_0=\hbar k_0.
$$

Its momentum uncertainty is

$$
\Delta p=\frac{\hbar}{2\sigma},
$$

so the packet is a minimum-uncertainty state:

$$
\Delta x\,\Delta p=\frac{\hbar}{2}.
$$

This equality refers to the initial Gaussian. The real envelope sets where
the particle is likely to be, while the factor $e^{ik_0(x-x_0)}$ sets its mean
momentum without changing the initial density. Narrowing the envelope requires
a wider range of Fourier wave numbers, explaining why smaller position width
means larger momentum uncertainty.

## 3. Exact free evolution

Every momentum component evolves with

$$
E(k)=\frac{\hbar^2k^2}{2m}.
$$

The exact infinite-domain packet is

$$
\psi(x,t)=
\frac{(2\pi\sigma^2)^{-1/4}}{\sqrt{1+i\tau}}
\exp\left[
-\frac{(x-x_0-vt)^2}{4\sigma^2(1+i\tau)}
+ik_0(x-x_0)-i\frac{\hbar k_0^2t}{2m}
\right],
$$

where

$$
v=\frac{\hbar k_0}{m},
\qquad
\tau=\frac{\hbar t}{2m\sigma^2}.
$$

Therefore,

$$
\langle x\rangle(t)=x_0+vt
$$

and

$$
\Delta x(t)=\sigma\sqrt{1+
\left(\frac{\hbar t}{2m\sigma^2}\right)^2}.
$$

The center moves at constant group velocity while the packet spreads because
different momentum components have different velocities.

To see where the solution comes from, Fourier transform the initial Gaussian,
multiply each wave-number amplitude by $e^{-iE(k)t/\hbar}$, then transform back.
The momentum probabilities stay fixed: only their phases change. The quadratic
dependence of energy on wave number produces the complex width $1+i\tau$.
Taking the absolute square removes the phase and gives the width above.

The characteristic spreading time is $2m\sigma^2/\hbar$. At that time the
position width has grown by a factor of $\sqrt2$. A heavier particle or a wider
initial packet spreads more slowly. Spreading does not imply energy gain:
the conserved mean energy includes both the mean momentum and its variance,

$$
\langle H\rangle=\frac{\langle p\rangle^2+(\Delta p)^2}{2m}
=\frac{\hbar^2}{2m}\left(k_0^2+\frac{1}{4\sigma^2}\right).
$$

## 4. Unitary evolution and invariants

For a time-independent Hermitian Hamiltonian,

$$
\psi(t)=\exp\left(-\frac{iHt}{\hbar}\right)\psi(0).
$$

The propagator is unitary, so

$$
\langle\psi(t)|\psi(t)\rangle=1.
$$

Energy is also conserved:

$$
\frac{d}{dt}\langle H\rangle=0.
$$

Backward propagation uses $U(-t)=U(t)^\dagger$. Exact forward evolution followed
by the same duration backward returns the original state. This inverse-evolution
check should be distinguished from physically time-reversing a state, which
also involves complex conjugation for this spinless, real-potential problem.

## 5. Finite-domain model

The numerical grid imposes zero Dirichlet boundaries. These are artificial
walls, not part of the infinite free-particle problem. The simulation must end
before the packet or a significant tail reaches either boundary. Otherwise,
reflection from the walls invalidates comparison with the analytical solution.
