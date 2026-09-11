# Quantum Theory: Barrier Scattering

## 1. Physical problem

The Hamiltonian is

$$
\hat H=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x),\qquad
V(x)=\begin{cases}V_0,&|x|<a/2,\\0,&|x|\ge a/2.\end{cases}
$$

Here a is the **full width**, not a half-width, and V0 is nonnegative. The
physical scattering problem extends over the whole real line. The numerical
box is an approximation, and its walls are not part of the barrier.

We solve the time-dependent equation

$$i\hbar\,\partial_t\psi=\hat H\psi.$$

The initial, infinite-domain normalized Gaussian is

$$
\psi(x,0)=(2\pi\sigma^2)^{-1/4}
\exp\left[-\frac{(x-x_0)^2}{4\sigma^2}+ik_0(x-x_0)\right].
$$

Its position standard deviation is sigma. Choose x0 well to the left of the
barrier and k0 positive. The code constructs the envelope and phase, then
normalizes their sampled values on the actual grid. A significant initial tail
inside the barrier changes the preparation and weakens a free-incident-packet
comparison.

The free group velocity near k0 is hbar*k0/m. Use it to estimate the collision
time, then check the actual trajectory. Different momenta travel at different
velocities and scatter differently.

## 2. Probability current and conservation

Let rho = psi* psi. Multiply the Schrödinger equation by psi*, multiply its
complex conjugate by psi, and subtract. For real V, the potential terms cancel:

$$
\partial_t\rho=\frac{i\hbar}{2m}
\left(\psi^*\psi''-\psi\psi^{*\prime\prime}\right)
=-\partial_x j,
$$

$$j=\frac{\hbar}{m}\operatorname{Im}(\psi^*\partial_x\psi).$$

Integrating over an interval gives

$$\frac{d}{dt}\int_{x_L}^{x_R}\rho\,dx=j(x_L)-j(x_R).$$

Probability leaves through a boundary when its outward current is positive.
For a plane wave A exp(ikx), j = hbar*k*|A|^2/m. This explains why scattering
coefficients are ratios of currents. In this chapter the potential is zero on
both sides, so incoming and transmitted wave numbers are equal.

Density and current answer different questions: density measures how much
probability is present, while current measures its flow. The incoming and
reflected waves overlap on the left and can form interference fringes. Those
fringes are not additional particles or a failure of probability conservation.
Current keeps track of the opposite propagation directions even in this
overlap region.

For a closed system with time-independent Hermitian H,
d< H >/dt = 0 and the total norm is constant. Tunnelling through a static barrier
does not require an energy increase or temporary energy borrowing.

## 3. Deriving plane-wave transmission

Translate the barrier to [0, a] for this derivation. Translation changes an
amplitude phase but not the reflection or transmission probabilities.
For E>0 define

$$k=\frac{\sqrt{2mE}}{\hbar},\qquad q^2=\frac{2m(E-V_0)}{\hbar^2}.$$

Take a unit incoming amplitude and no incoming wave from the right:

$$
\phi_L=e^{ikx}+r e^{-ikx},\quad
\phi_B=C\cos(qx)+D\frac{\sin(qx)}q,\quad
\phi_R=t e^{ikx}.
$$

The interior form also has a regular q=0 limit: phi_B = C + Dx.
Both phi and phi' are continuous at the two finite potential steps. A jump
in the derivative would create a delta function in the second derivative,
with no matching term in the finite potential; a jump in the wavefunction is
also incompatible with this equation. These conditions apply to amplitudes and
derivatives, not just to densities, because their phases determine interference.

Applying them at the left and right edges gives

$$1+r=C,\qquad ik(1-r)=D,$$

$$C\cos(qa)+D\frac{\sin(qa)}q=t e^{ika},$$

$$-Cq\sin(qa)+D\cos(qa)=ik t e^{ika}.$$

Eliminating C, D, and r yields

$$
t=\frac{e^{-ika}}{\cos(qa)-i\frac{k^2+q^2}{2kq}\sin(qa)}.
$$

Since both exterior velocities are equal, T(E)=|t|^2 and R(E)=|r|^2=1-T(E).
For E>V0, q is real and

$$
T(E)=\left[1+\frac{V_0^2\sin^2(qa)}{4E(E-V_0)}\right]^{-1}.
$$

For 0<E<V0, write q=i*kappa, where
kappa=sqrt(2m(V0-E))/hbar. Using sin(i*z)=i*sinh(z) gives

$$
T(E)=\left[1+\frac{V_0^2\sinh^2(\kappa a)}{4E(V_0-E)}\right]^{-1}.
$$

The wave decays inside a sub-barrier region, but for finite height and width
there is nonzero transmission. For E>V0, reflection can still occur; perfect
transmission occurs at q*a=n*pi for positive integer n. The finite-barrier
matching approach is also developed in the University of Manchester's
[square-barrier notes](https://oer.physics.manchester.ac.uk/QM/Notes/jsmath/Notesse22.html).

At E=V0>0, take the limit sin(qa)/q -> a:

$$T(V_0)=\left[1+\frac{m V_0 a^2}{2\hbar^2}\right]^{-1}.$$

For positive height and width, T tends to zero as E tends to zero. Removing the
barrier (zero height or width) gives T=1. These limits are valuable tests.

Above the barrier, the two interfaces each reflect part of the wave. Their
reflected amplitudes can cancel when the phase accumulated inside has the
resonant value $qa=n\pi$. That is why complete transmission can occur even
though the potential is nonzero. Below the barrier, the interior wave number
is imaginary and the trigonometric oscillations become exponential behavior;
the same matching calculation describes both regimes.

## 4. A Gaussian packet is not a single energy

Fourier transforming the Gaussian gives the normalized wave-number density

$$f(k)=\sqrt{\frac2\pi}\,\sigma\exp[-2\sigma^2(k-k_0)^2],\qquad
\Delta k=\frac1{2\sigma}.$$

The central energy is hbar^2*k0^2/(2m), while the mean free energy includes the
spread:

$$\langle E\rangle=\frac{\hbar^2}{2m}\left(k_0^2+\frac1{4\sigma^2}\right).$$

For a packet initially far to the left, its asymptotic transmitted probability
is approximately

$$T_{\rm packet}=\int_0^\infty f(k)\,T\left(\frac{\hbar^2k^2}{2m}\right)dk.$$

This follows by applying the stationary transmission amplitude to each incoming
momentum component and using the momentum-space norm after packet separation.
Do not replace this integral by T evaluated at the central or mean energy unless
the spectrum is narrow enough and T varies little across it.

The negative-k weight is

$$P_{k<0}=\tfrac12\operatorname{erfc}(\sqrt2\sigma k_0).$$

The reference integrates the positive-k part without renormalizing away this
weight. Negative-k components initially travel away from the barrier. They are
negligible for the default sigma=2 and k0=2, but can matter for broader spectra.

The default Gaussian includes above-barrier energies even though its central
and mean energies are below V0. Its total transmitted probability therefore
combines sub-barrier tunnelling and above-barrier transmission. To isolate the
former in a later experiment, control the spectral weight above V0 as well as
the mean energy.

## 5. Finite-time measurements

Choose c=a/2+padding and partition the wavefunction:

$$P_L=\int_{x<-c}|\psi|^2dx,\quad
P_N=\int_{-c\le x\le c}|\psi|^2dx,\quad
P_R=\int_{x>c}|\psi|^2dx.$$

These three probabilities always sum to the current norm. Initially PL is
mostly the incident packet, so calling it reflection is incorrect. During
collision, PN measures unresolved probability near the barrier.

Use PL≈R and PR≈T only when PN is sufficiently small, the outgoing probabilities
have stabilized, and the box walls have not affected the state. Increasing the
measurement time alone is not enough: eventually the walls reflect the outgoing
packets. A small final boundary density cannot rule out an earlier reflection.

## 6. Units

Choose a length L0, energy E0=hbar^2/(m L0^2), and time t0=hbar/E0.
The code's m=hbar=1 examples are dimensionless versions of the equations in
these units. Barrier height, packet width, momentum, and time must use the same
consistent scales. The APIs retain m and hbar for comparisons in other units.
