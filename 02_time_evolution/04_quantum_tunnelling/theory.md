# Quantum tunnelling with a controlled incident spectrum

## 1. Physical system and units

Solve the one-dimensional time-dependent Schrödinger equation

\[
i\hbar\partial_t\psi=H\psi,
\qquad H=-\frac{\hbar^2}{2m}\partial_x^2+V(x),
\qquad V(x)=V_0 e^{-x^2/(2a^2)}.
\]

The Gaussian barrier is smooth, with peak height `V0` and scale `a`. Its full
width at half maximum is \(2\sqrt{2\ln2}\,a\). It is a different potential from
Chapter 3's rectangular barrier; the rectangular transmission formula does not
apply. Smoothness helps us isolate Fourier discretization error.

Choose length unit \(\ell\), energy unit \(\hbar^2/(m\ell^2)\), and time unit
\(m\ell^2/\hbar\). The dimensionless equation has `m=hbar=1`. The default peak is
2.5, barrier scale 0.8, and central incident wave number 1.5.

## 2. Why transmission below the barrier is possible

Substitute \(\psi(x,t)=\phi(x)e^{-iEt/\hbar}\) into the time-dependent equation:

\[
-\frac{\hbar^2}{2m}\phi''+V\phi=E\phi,
\qquad \phi''=\frac{2m}{\hbar^2}(V-E)\phi.
\]

Where `V<E`, constant-potential solutions oscillate. Where `V>E`, they are
exponentials rather than identically zero. Matching the wavefunction and its
derivative across a finite barrier permits a nonzero outgoing amplitude.
Classical motion at the same energy would turn around.

For the Gaussian barrier and `0<E<V0`, the turning points satisfy

\[
x_\pm=\pm a\sqrt{2\ln(V_0/E)}.
\]

The slowly varying barrier estimate has an exponential factor
\(\exp[-2\int_{x_-}^{x_+}\sqrt{2m(V(x)-E)}\,dx/\hbar]\). This explains suppression
with barrier width/height, but is not an exact reference, especially near the
barrier top. We validate the implemented dynamics with matrix exponentials and
independently converged finite differences instead.

## 3. A packet has an energy distribution

The incoming state is

\[
\psi(x,0)=(2\pi\sigma^2)^{-1/4}
e^{-(x-x_0)^2/(4\sigma^2)}e^{ik_0(x-x_0)}.
\]

Its density has position standard deviation `sigma`. Taking the Fourier
transform with convention \(\widetilde\psi(k)=(2\pi)^{-1/2}\int e^{-ikx}\psi(x)dx\)
gives a Gaussian momentum density:

\[
f(k)=|\widetilde\psi(k)|^2
=\sqrt{\frac{2\sigma^2}{\pi}}e^{-2\sigma^2(k-k_0)^2},
\qquad \Delta k=\frac{1}{2\sigma}.
\]

Far from the barrier the free energy is \(E(k)=\hbar^2 k^2/(2m)\). Thus

\[
E_0=\frac{\hbar^2 k_0^2}{2m},\qquad
\langle E\rangle=\frac{\hbar^2}{2m}\left(k_0^2+\frac{1}{4\sigma^2}\right).
\]

`E0<V0` alone does not make every momentum component sub-barrier. Define
\(k_c=\sqrt{2mV_0}/\hbar\). Integrating the Gaussian tails gives

\[
P(k<0)=\tfrac12\operatorname{erfc}(\sqrt2\sigma k_0),
\]

\[
P(E>V_0)=\tfrac12\operatorname{erfc}[\sqrt2\sigma(k_c-k_0)]
+\tfrac12\operatorname{erfc}[\sqrt2\sigma(k_c+k_0)].
\]

The second term includes negative high-energy momenta. For an asymptotically
incoming packet, the above-barrier contribution to transmission is bounded by
this total incident weight because each energy's transmission lies between zero
and one. These free-packet estimates require negligible initial overlap with
the barrier and boundaries. They are not estimates of the instantaneous kinetic
energy distribution while the packet is interacting with the potential.

The default `sigma=3` gives \(\Delta k=1/6\), \(E_0=1.125\), mean energy 1.138889,
and above-barrier weight about \(5.02\times10^{-6}\). Transmission is about 0.016,
far larger than that tail. Subject to the numerical checks, it is predominantly
sub-barrier tunnelling. A finite Gaussian never has a strictly bounded spectrum.

Increasing `sigma` narrows the momentum spread but widens the packet. Move the
initial center farther away and enlarge the box when needed. A width sweep
changes the physical state; it is not numerical grid convergence.

## 4. What to measure

Integrate density on `x<-cut`, `|x|<=cut`, and `x>cut`:

\[
P_L+P_N+P_R=1.
\]

The default cut is 4, where the potential is already small but not zero.
Early `P_L` contains the incoming packet. Interpret late `P_L` and `P_R` as
reflection and transmission only after `P_N` is small and `P_R` has plateaued.
Their asymptotic sum approaches one. Check this time window in a larger domain.

The FFT evolves a periodic box: probability crossing the right edge reappears
at the left edge, and conversely. Such wraparound preserves norm and can create
false transmission. Track edge strips over time, not just the final norm.

## 5. Connection to the numerical method

The kinetic energy is diagonal in momentum space and the potential is diagonal
in position space. Alternating the two representations lets us apply short-time
evolution without building a Hamiltonian matrix. The next document derives the
splitting and explains its accuracy limits.
