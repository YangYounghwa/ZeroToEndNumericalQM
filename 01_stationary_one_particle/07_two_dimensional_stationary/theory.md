# Quantum Theory

## 1. Two-dimensional equation

For a particle moving in the `xy` plane,

$$
\left[-\frac{\hbar^2}{2m}
\left(\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}\right)
+V(x,y)\right]\psi(x,y)=E\psi(x,y).
$$

The wavefunction is normalized by

$$
\int\int |\psi(x,y)|^2\,dx\,dy=1.
$$

Zero Dirichlet boundaries are imposed on the four sides of a finite rectangle.

## 2. Separable potentials

If

$$
V(x,y)=V_x(x)+V_y(y),
$$

then a product state

$$
\psi_{n_xn_y}(x,y)=X_{n_x}(x)Y_{n_y}(y)
$$

has energy

$$
E_{n_xn_y}=E_{n_x}^{(x)}+E_{n_y}^{(y)}.
$$

This sum structure is the physical reason for the Kronecker-sum matrix used in
the code.

## 3. Analytical reference

For

$$
V(x,y)=\frac12m\left(\omega_x^2x^2+\omega_y^2y^2\right),
$$

the exact energies are

$$
E_{n_xn_y}=\hbar\omega_x\left(n_x+\frac12\right)
+\hbar\omega_y\left(n_y+\frac12\right).
$$

Using unequal frequencies makes the ordering of low states easier to inspect.
When `omega_x = omega_y`, exchanging `n_x` and `n_y` leaves the energy
unchanged. The first excited states `(1,0)` and `(0,1)` are therefore
degenerate.

## 4. Observables

The position expectations are

$$
\langle x\rangle=\int\int x|\psi|^2dxdy,
\qquad
\langle y\rangle=\int\int y|\psi|^2dxdy.
$$

They vanish for parity eigenstates of a potential symmetric under independent
`x` and `y` reflection.

## 5. Why this is not radial hydrogen

Radial hydrogen is a 3D physical problem reduced to one coordinate by spherical
symmetry and known spherical harmonics. Here no angular separation is assumed.
The numerical unknown genuinely occupies a 2D Cartesian grid, so an arbitrary
real `V(x,y)` can break rotational and reflection symmetries.
