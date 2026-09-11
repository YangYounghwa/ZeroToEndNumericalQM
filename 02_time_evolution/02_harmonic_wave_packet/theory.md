# Quantum Theory

## 1. Harmonic Hamiltonian

The time-dependent equation is

$$
i\hbar\frac{\partial\psi}{\partial t}
=\left[-\frac{\hbar^2}{2m}\frac{\partial^2}{\partial x^2}
+\frac12m\omega^2x^2\right]\psi.
$$

The potential confines the particle and produces equally spaced stationary
energies

$$
E_n=\hbar\omega\left(n+\frac12\right).
$$

## 2. Coherent initial state

The oscillator ground-state probability width is

$$
\sigma_x=\sqrt{\frac{\hbar}{2m\omega}}.
$$

Displace this Gaussian to `x_0` and give it mean momentum `p_0 = hbar k_0`:

$$
\psi(x,0)\propto
\exp\left[-\frac{(x-x_0)^2}{4\sigma_x^2}
+ik_0(x-x_0)\right].
$$

This is a coherent state. It is a superposition of stationary oscillator
states, but its probability density retains a Gaussian form.

The specific width matters. Displacing the ground state changes its mean
position, and multiplying by a linear phase changes its mean momentum, while
preserving the ground state's fluctuations. An arbitrary Gaussian is not
necessarily coherent. If its width differs from this value, the oscillator
can periodically squeeze and expand it even with exact evolution.

## 3. Center motion

Ehrenfest's theorem gives

$$
\frac{d\langle x\rangle}{dt}=\frac{\langle p\rangle}{m},
\qquad
\frac{d\langle p\rangle}{dt}=-m\omega^2\langle x\rangle.
$$

Therefore the quantum expectation follows the classical trajectory exactly:

$$
\boxed{
\langle x\rangle(t)=x_0\cos(\omega t)
+\frac{p_0}{m\omega}\sin(\omega t)
}.
$$

The period is

$$
T=\frac{2\pi}{\omega}.
$$

For a general potential, Ehrenfest's force is $-\langle V'(x)\rangle$, which
need not equal $-V'(\langle x\rangle)$. The harmonic force is linear, so those
expressions agree exactly. Differentiating the first expectation equation and
inserting the second gives the classical oscillator equation for the center.
This center trajectory holds for any oscillator state, even one whose shape
changes; it does not by itself prove that the state is coherent.

## 4. No spreading

A free Gaussian spreads because its momentum components move with different
velocities. In a coherent oscillator state, the harmonic dynamics preserve the
ground-state uncertainty:

$$
\Delta x(t)=\sqrt{\frac{\hbar}{2m\omega}}.
$$

The packet changes position and phase but not shape. A visibly breathing width
indicates spatial error, time-step error, boundary interference, or an initial
Gaussian whose width is not the coherent-state width.

One way to understand the fixed width is to evolve the centered position
operator: it is a combination of the initial position times $\cos(\omega t)$
and initial momentum times $\sin(\omega t)/(m\omega)$. In the coherent state,
their initial covariance is zero and their variances satisfy
$(\Delta p)^2=m^2\omega^2(\Delta x)^2$. The resulting position variance is
therefore proportional to $\cos^2(\omega t)+\sin^2(\omega t)=1$.

## 5. Conserved quantities and return

The Hamiltonian is time independent and Hermitian, so norm and energy are
constant. After one period the coherent state returns to the same physical
state. Its vector can differ by a global phase, so return is measured by

$$
F=|\langle\psi(0)|\psi(T)\rangle|^2.
$$

An ideal return has `F = 1`.

The energy separates into the ground-state energy and the classical energy
of the displaced center:

$$
\langle H\rangle=\frac{\hbar\omega}{2}
+\frac{p_0^2}{2m}+\frac12m\omega^2x_0^2.
$$

The first term remains even for a stationary center because position and
momentum still fluctuate. After one period, each eigenstate phase is
$e^{-i2\pi(n+1/2)}=-1$. Thus the full state vector changes sign while its
density and all measurement probabilities return. Fidelity correctly ignores
that common phase.

## 6. Finite-domain limitations

The exact oscillator lives on the entire real line. Zero boundaries at
`x_min` and `x_max` are harmless only when the displaced packet and its tails
remain far from them throughout the orbit. Large displacement or momentum
requires a larger domain.
