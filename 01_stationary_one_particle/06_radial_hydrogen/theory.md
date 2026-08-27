# Quantum Theory

## 1. Physical system and units

A particle of mass `m` moves in the attractive Coulomb potential of a fixed
nucleus with charge parameter `Z`:

$$
V(r)=-\frac{Z}{r}.
$$

The implementation uses atomic units by default, so `m = hbar = 1`. In these
units, hydrogen has `Z = 1`, length is measured in Bohr radii, and energy in
Hartree. The code retains `m` and `hbar` parameters to make their positions in
the equation explicit.

The stationary Schrödinger equation is

$$
\left[-\frac{\hbar^2}{2m}\nabla^2-\frac{Z}{r}\right]\psi
=E\psi.
$$

## 2. Separation in spherical coordinates

Because the potential depends only on `r`, write

$$
\psi(r,\theta,\phi)=R_\ell(r)Y_\ell^m(\theta,\phi),
$$

where the spherical harmonic satisfies

$$
\hat L^2Y_\ell^m=\hbar^2\ell(\ell+1)Y_\ell^m.
$$

The radial part of the Laplacian is

$$
\nabla^2=
\frac{1}{r^2}\frac{\partial}{\partial r}
\left(r^2\frac{\partial}{\partial r}\right)
-\frac{\hat L^2}{\hbar^2r^2}.
$$

Substitution gives

$$
-\frac{\hbar^2}{2m}
\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR_\ell}{dr}\right)
+\frac{\hbar^2\ell(\ell+1)}{2mr^2}R_\ell
-\frac{Z}{r}R_\ell
=ER_\ell.
$$

## 3. Reduced radial wavefunction

Define

$$
u_\ell(r)=rR_\ell(r).
$$

Since `R = u/r`, direct differentiation gives

$$
\frac{1}{r^2}\frac{d}{dr}\left(r^2\frac{dR}{dr}\right)
=\frac{1}{r}\frac{d^2u}{dr^2}.
$$

Multiplying the radial equation by `r` produces a one-dimensional-looking
equation on the half-line:

$$
\boxed{
\left[
-\frac{\hbar^2}{2m}\frac{d^2}{dr^2}
+\frac{\hbar^2\ell(\ell+1)}{2mr^2}
-\frac{Z}{r}
\right]u_\ell(r)=Eu_\ell(r)
}.
$$

The effective potential is therefore

$$
V_{\mathrm{eff}}(r)=
\frac{\hbar^2\ell(\ell+1)}{2mr^2}-\frac{Z}{r}.
$$

The first term is the centrifugal barrier. It vanishes for `ell = 0` and
repels higher-angular-momentum states from the origin.

## 4. Boundary conditions and normalization

The physical wavefunction must remain finite at the origin. Since `R = u/r`,
this requires

$$
u(0)=0.
$$

A bound state also satisfies `u(r) -> 0` as `r -> infinity`. Numerically, the
half-line is truncated at `r_max` and the second condition becomes

$$
u(r_{\max})=0.
$$

The three-dimensional normalization simplifies because the spherical
harmonics are normalized:

$$
1=\int |\psi|^2d^3r
=\int_0^\infty |R(r)|^2r^2dr
=\int_0^\infty |u(r)|^2dr.
$$

Thus `|u(r)|^2 dr`, not `|R(r)|^2 dr`, is the radial probability.

## 5. Analytical reference results

The principal quantum number obeys

$$
n=1,2,3,\ldots,\qquad \ell=0,1,\ldots,n-1.
$$

For the potential convention used here,

$$
E_n=-\frac{mZ^2}{2\hbar^2n^2}.
$$

The energy depends on `n` but not on `ell` or the magnetic quantum number.
Consequently, the `2s` and `2p` radial sectors have the same exact energy.
Finite-grid errors break this equality slightly; the difference should shrink
under refinement.

The exact mean radius is

$$
\langle r\rangle_{n\ell}
=\frac{\hbar^2}{2mZ}\left[3n^2-\ell(\ell+1)\right].
$$

For the hydrogen `1s` state, `E = -1/2` and `<r> = 3/2` in atomic units.

## 6. Expected behavior and limitations

- Bound-state energies are negative and approach zero from below as `n`
  increases.
- Higher states extend farther from the nucleus and need a larger `r_max`.
- Increasing `ell` adds a repulsive short-distance barrier.
- The origin is never sampled because both `-Z/r` and the centrifugal term
  are singular there.
- This chapter solves only the radial equation for a fixed `ell`; reconstructing
  the full three-dimensional wavefunction also requires spherical harmonics.
- Relativistic effects, spin, finite nuclear mass beyond the supplied `mass`,
  and fine structure are outside this nonrelativistic model.
