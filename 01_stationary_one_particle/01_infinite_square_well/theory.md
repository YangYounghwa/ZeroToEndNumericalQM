# Quantum Theory: Infinite Square Well

## Physical system

A particle of mass $m$ is confined to the interval

$$
0 < x < L.
$$

Its potential energy is

$$
V(x) =
\begin{cases}
0, & 0 < x < L, \\
\infty, & x \leq 0 \text{ or } x \geq L.
\end{cases}
$$

The particle cannot exist outside the well. Therefore, its wavefunction must
satisfy the boundary conditions

$$
\psi(0)=0, \qquad \psi(L)=0.
$$

## Time-independent Schrödinger equation

For a stationary state, the Schrödinger equation is

$$
\hat H\psi(x)=E\psi(x),
$$

where the Hamiltonian is

$$
\hat H=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x).
$$

Inside the well, $V(x)=0$, so

$$
-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2}=E\psi.
$$

Rearranging gives

$$
\frac{d^2\psi}{dx^2}+k^2\psi=0,
\qquad
k^2=\frac{2mE}{\hbar^2}.
$$

## Analytical solution

The general solution inside the well is

$$
\psi(x)=A\sin(kx)+B\cos(kx).
$$

The boundary condition $\psi(0)=0$ gives

$$
B=0.
$$

The second boundary condition gives

$$
\psi(L)=A\sin(kL)=0.
$$

A nonzero wavefunction requires

$$
kL=n\pi,
\qquad n=1,2,3,\ldots
$$

Therefore,

$$
k_n=\frac{n\pi}{L}.
$$

Substitution into the definition of $k$ gives the allowed energies:

$$
E_n=\frac{n^2\pi^2\hbar^2}{2mL^2}.
$$

The energy is quantized because only wavelengths that satisfy both boundary
conditions are allowed.

## Wavefunction normalization

The probability of finding the particle somewhere inside the well must be one:

$$
\int_0^L |\psi_n(x)|^2\,dx=1.
$$

Using

$$
\int_0^L \sin^2\left(\frac{n\pi x}{L}\right)dx=\frac{L}{2},
$$

the normalized eigenfunctions are

$$
\psi_n(x)=\sqrt{\frac{2}{L}}
\sin\left(\frac{n\pi x}{L}\right).
$$

They obey orthonormality:

$$
\int_0^L \psi_n^*(x)\psi_j(x)\,dx=\delta_{nj}.
$$

## Observables used for validation

For the state $n$:

$$
\langle x\rangle=\frac{L}{2},
$$

and

$$
E_n \propto n^2.
$$

The numerical solution should reproduce the analytical energies,
normalization, orthogonality, and symmetry about the center of the well.

## Units and foundations used here

A state is a column of sampled amplitudes. Matrix multiplication applies an
operator, and an eigenvector of the Hamiltonian has a definite energy.
Hermitian conjugation means transpose plus complex conjugation. This chapter's
Hamiltonian is real, but later states will be complex.

For a well of length L, set q = x/L and E_scale = hbar^2/(m L^2).
Then H/E_scale = -(1/2) d^2/dq^2 on [0, 1], and E_n/E_scale = n^2 pi^2/2.
The dimensionless wavefunction is phi(q) = sqrt(L) psi(Lq), so its integral
probability remains one. Setting m = hbar = L = 1 expresses this choice of units.

An eigensolver returns columns v with sum(abs(v)^2) = 1. Sampled wavefunctions
instead satisfy dx * sum(abs(psi)^2) = 1, so psi = v/sqrt(dx). The inner product
is dx * psi.conj().T @ psi. Use the same weight in residuals and observables.
