# Numerical Method

## 1. Hamiltonian construction

Use the same interior grid and second-order kinetic matrix as Chapter 1. Add
the sampled potential to the main diagonal:

$$
H=T+\operatorname{diag}\left(\frac12m\omega^2x_i^2\right).
$$

The matrix remains real symmetric and tridiagonal.

## 2. Time propagation

Crank-Nicolson applies without changing its structure:

$$
\left(I+\frac{i\Delta t}{2\hbar}H\right)\psi^{n+1}
=\left(I-\frac{i\Delta t}{2\hbar}H\right)\psi^n.
$$

Because the harmonic Hamiltonian is time independent, factorize the left
matrix once. The method remains unitary for the discrete Hermitian `H` and has
global `O(dt^2)` error.

## 3. Stronger physical validation

The free packet checked translation and spreading. A coherent harmonic packet
adds three precise checks:

1. `<x>(t)` follows the analytical sinusoid.
2. `Delta x(t)` stays at the ground-state width.
3. Fidelity with the initial state approaches one after `2*pi/omega`.

These tests depend on both kinetic and potential terms and therefore catch
errors that norm conservation alone cannot detect.

## 4. Matrix-exponential reference

On a small grid, compare the final Crank-Nicolson state with

$$
\exp(-iHT/\hbar)\psi(0).
$$

After aligning global phase, halving `dt` should reduce the final-state error
by about four.

## 5. Spatial versus temporal error

Even an extremely small `dt` cannot correct a coarse spatial Hamiltonian.
Finite differences slightly alter oscillator level spacings. Over a full
period, these small energy errors accumulate as phase error, shifting the
packet center and reducing return fidelity.

Refine space when center or width errors stop improving under time-step
refinement. Expand the domain if the packet tail approaches a boundary.

## 6. Computational costs

The sparse NumPy path stores `O(N)` Hamiltonian entries and performs one sparse
factorization followed by triangular solves. The PyTorch reference uses dense
`O(N^2)` storage but advances several initial-state columns together. Storing
all times costs `O(num_steps * N)` per state.
