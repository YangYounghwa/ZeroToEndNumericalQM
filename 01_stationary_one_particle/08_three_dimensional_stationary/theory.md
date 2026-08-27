# Quantum Theory

## 1. Three-dimensional equation

The Cartesian stationary Schrödinger equation is

$$
\left[-\frac{\hbar^2}{2m}
\left(\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}
+\frac{\partial^2}{\partial z^2}\right)
+V(x,y,z)\right]\psi=E\psi.
$$

Normalization requires

$$
\iiint |\psi(x,y,z)|^2\,dx\,dy\,dz=1.
$$

Unlike radial hydrogen, this Cartesian formulation does not assume spherical
symmetry. It can represent an arbitrary sampled real `V(x,y,z)`.

## 2. Separable systems

For

$$
V(x,y,z)=V_x(x)+V_y(y)+V_z(z),
$$

the eigenfunctions factorize and energies add:

$$
\psi=X_{n_x}Y_{n_y}Z_{n_z},
\qquad
E=E_{n_x}^{(x)}+E_{n_y}^{(y)}+E_{n_z}^{(z)}.
$$

This separability provides an exact reference even though the implementation
constructs and solves the full 3D grid problem.

## 3. Anisotropic oscillator reference

For

$$
V=\frac12m(\omega_x^2x^2+\omega_y^2y^2+\omega_z^2z^2),
$$

the energies are

$$
E_{n_xn_yn_z}=\hbar\left[
\omega_x\left(n_x+\frac12\right)
+\omega_y\left(n_y+\frac12\right)
+\omega_z\left(n_z+\frac12\right)
\right].
$$

Unequal frequencies reduce accidental degeneracies. In the isotropic case,
the three first-excited states `(1,0,0)`, `(0,1,0)`, and `(0,0,1)` are
degenerate.

## 4. Observables and symmetry

Position expectations use the volume probability density. For a potential
symmetric under reflection of each coordinate, parity eigenstates satisfy
`<x> = <y> = <z> = 0`.

Within an exactly degenerate subspace, a numerical eigensolver may return any
orthonormal linear combinations. Energies and the subspace are well defined,
but individual eigenvector shapes are not unique.

## 5. Physical and numerical scope

This chapter handles one spinless nonrelativistic particle on a rectangular
Cartesian domain. It does not include magnetic vector potentials, spin,
relativistic corrections, or adaptive meshes. Uniform 3D grids are useful for
learning but become expensive quickly.
