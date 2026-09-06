# Quantum Theory: General Two-Dimensional Potentials

## 1. What changes in two dimensions

Chapters 1–6 already established the stationary eigenvalue equation,
normalization, Hermiticity, boundary conditions, and finite-difference
Hamiltonians. The new issue is not the origin of

$$
\hat H\psi_n=E_n\psi_n.
$$

The new issue is that the unknown is now a function over an area:

$$
\psi_n=\psi_n(x,y),
$$

and the Hamiltonian contains two spatial derivatives:

$$
\boxed{
\hat H=
-\frac{\hbar^2}{2m}
\left(
\frac{\partial^2}{\partial x^2}
+\frac{\partial^2}{\partial y^2}
\right)
+V(x,y)
}.
$$

The solver must determine the full 2D shape of every eigenfunction. It cannot
assume that the state is a product of two one-dimensional functions.

The numerical domain is the rectangle

$$
\Omega=[x_{\min},x_{\max}]\times[y_{\min},y_{\max}],
$$

with zero Dirichlet values on all four edges. The normalization is

$$
\int_\Omega |\psi_n(x,y)|^2\,dx\,dy=1.
$$

For a real potential, these boundary conditions make the Hamiltonian
Hermitian. Energies are real, different nondegenerate eigenstates are
orthogonal, and the eigenfunctions may be chosen real.

## 2. General versus separable potentials

A special potential has the additive form

$$
V(x,y)=V_x(x)+V_y(y).
$$

Then the product ansatz

$$
\psi(x,y)=X(x)Y(y)
$$

reduces the 2D equation to two 1D equations. The energy is a sum:

$$
E=E_x+E_y.
$$

This is useful for constructing analytical tests, but it is not the general
case.

For a general potential,

$$
V(x,y)\ne V_x(x)+V_y(y).
$$

Changing $x$ can change the force and confinement experienced along $y$, and
vice versa. No pair of independent 1D eigenproblems contains the full physics.
The complete 2D partial differential equation must be solved.

The code therefore accepts one callable that evaluates the entire potential:

```text
V(x_mesh, y_mesh) -> array with shape (Ny, Nx)
```

It does not accept separate `Vx` and `Vy` components.

## 3. Main problem: a coupled quartic potential

The main numerical example uses $m=\hbar=1$ and

$$
\boxed{
V(x,y)=\frac12\left(\omega_x^2x^2+\omega_y^2y^2\right)
+\lambda x^2y^2
},
$$

with

$$
\omega_x=1,
\qquad
\omega_y=1.4,
\qquad
\lambda=0.08.
$$

The harmonic terms confine the particle near the origin. The positive quartic
term penalizes configurations in which both $|x|$ and $|y|$ are large.

### Proof that the potential is nonseparable

Every additive potential obeys

$$
V(x,y)-V(x,0)-V(0,y)+V(0,0)=0.
$$

For the coupled potential,

$$
V(x,y)-V(x,0)-V(0,y)+V(0,0)=\lambda x^2y^2.
$$

This is nonzero for generic $x$ and $y$, proving that the potential cannot be
written as $V_x(x)+V_y(y)$.

The same obstruction appears directly in the Schrödinger equation. Substitute
$\psi=XY$ and divide by $XY$:

$$
-\frac{\hbar^2}{2m}\frac{X''}{X}
+\frac12m\omega_x^2x^2
-\frac{\hbar^2}{2m}\frac{Y''}{Y}
+\frac12m\omega_y^2y^2
+\lambda x^2y^2
=E.
$$

The first two terms depend only on $x$, and the next two depend only on $y$.
The final product depends on both. It cannot be absorbed into either 1D
equation, so ordinary separation of variables fails.

## 4. Geometry of the coupled potential

The potential has one minimum:

$$
V(0,0)=0.
$$

Along the coordinate axes,

$$
V(x,0)=\frac12\omega_x^2x^2,
\qquad
V(0,y)=\frac12\omega_y^2y^2.
$$

Away from the axes, the effective curvature changes. At fixed $x$,

$$
V(x,y)
=\frac12\omega_x^2x^2
+\frac12\left(\omega_y^2+2\lambda x^2\right)y^2.
$$

Thus the effective $y$ frequency increases with $|x|$:

$$
\omega_{y,\mathrm{eff}}^2(x)=\omega_y^2+2\lambda x^2.
$$

Similarly,

$$
\omega_{x,\mathrm{eff}}^2(y)=\omega_x^2+2\lambda y^2.
$$

This makes the physical meaning of coupling explicit: displacement in one
direction changes confinement in the other direction.

Because $V\to+\infty$ in every direction, the spectrum is discrete. The
positive coupling suppresses probability in the diagonal corner regions
relative to the uncoupled oscillator.

## 5. Exact symmetries and expected eigenstates

The potential is independently even in both coordinates:

$$
V(-x,y)=V(x,y),
\qquad
V(x,-y)=V(x,y).
$$

The Hamiltonian commutes with the two reflection operators, so eigenstates can
be classified by

$$
(P_x,P_y)=(+,+),(+,-),(-,+),(-,-).
$$

These symmetries remain exact even though the equation is nonseparable.

Expected behavior:

- The nondegenerate ground state has parity $(+,+)$ and no nodes.
- An $x$-odd state has a nodal line at $x=0$.
- A $y$-odd state has a nodal line at $y=0$.
- Every parity eigenstate satisfies
  $\langle x\rangle=\langle y\rangle=0$.
- The coupling changes energies and shapes but does not mix states with
  different parity pairs.

The last statement follows because $x^2y^2$ is even in both coordinates.

## 6. Weak-coupling theory as a partial analytical check

The full coupled spectrum has no simple closed form. Perturbation theory gives
an analytical approximation when the coupling changes the unperturbed states
only weakly.

Write

$$
\hat H=\hat H_0+\lambda\hat W,
\qquad
\hat W=x^2y^2,
$$

where $\hat H_0$ is the separable anisotropic oscillator. Its unperturbed
states are $|n_x,n_y\rangle$.

Writing $\hat H=\hat H_0+\lambda\hat W$ is only a decomposition of the
Hamiltonian. It does **not** mean that every method used afterward is
perturbative. The distinction is what is done with $\lambda\hat W$:

- perturbation theory expands the answer in powers of $\lambda$ and truncates
  the expansion;
- direct numerical diagonalization places the complete term in the matrix and
  solves the resulting eigenproblem without truncating powers of $\lambda$.

### 6.1 Energy and state expansions

Perturbation theory assumes expansions of the form

$$
E_n(\lambda)
=E_n^{(0)}+\lambda E_n^{(1)}
+\lambda^2E_n^{(2)}+\cdots,
$$

$$
|\psi_n(\lambda)\rangle
=|n^{(0)}\rangle
+\lambda|n^{(1)}\rangle
+\lambda^2|n^{(2)}\rangle+\cdots.
$$

First-order perturbation theory keeps only the first two energy terms:

$$
E_{n_xn_y}
\approx E_{n_xn_y}^{(0)}
+\lambda
\langle n_x|x^2|n_x\rangle
\langle n_y|y^2|n_y\rangle.
$$

For a nondegenerate state, the first correction to the eigenvector is

$$
|n^{(1)}\rangle
=\sum_{m\ne n}
\frac{\langle m^{(0)}|\hat W|n^{(0)}\rangle}
{E_n^{(0)}-E_m^{(0)}}
|m^{(0)}\rangle.
$$

This equation shows the physical meaning of perturbative mixing. The coupling
adds small components of other unperturbed states. Small energy denominators
produce strong mixing, so ordinary nondegenerate perturbation theory can fail
near a degeneracy even when $\lambda$ appears numerically small.

The second-order energy correction is

$$
E_n^{(2)}
=\sum_{m\ne n}
\frac{|\langle m^{(0)}|\hat W|n^{(0)}\rangle|^2}
{E_n^{(0)}-E_m^{(0)}}.
$$

Stopping at first order discards the contribution $\lambda^2E_n^{(2)}$ and
all higher terms.

### 6.2 What “small coupling” means

The numerical value of $\lambda$ alone is not a general measure of smallness,
because $\lambda$ has units unless dimensionless units have already been
chosen. A useful state-dependent condition is

$$
\epsilon_{mn}
=\frac{|\lambda\langle m^{(0)}|\hat W|n^{(0)}\rangle|}
{|E_n^{(0)}-E_m^{(0)}|}
\ll1
$$

for every state $m$ that couples appreciably to $n$.

For the oscillator, characteristic lengths are

$$
\ell_x=\sqrt{\frac{\hbar}{m\omega_x}},
\qquad
\ell_y=\sqrt{\frac{\hbar}{m\omega_y}}.
$$

The characteristic coupling energy is therefore of order

$$
E_{\mathrm{coupling}}
\sim\lambda\ell_x^2\ell_y^2
=\lambda\frac{\hbar^2}{m^2\omega_x\omega_y}.
$$

It should be small compared with relevant oscillator energy gaps. This is a
scale estimate, not a proof of convergence.

### 6.3 First-order result for the coupled oscillator

For a one-dimensional oscillator,

$$
\langle n|x^2|n\rangle
=\frac{\hbar}{m\omega}\left(n+\frac12\right).
$$

Therefore,

$$
\boxed{
E_{n_xn_y}
\approx
\hbar\omega_x\left(n_x+\frac12\right)
+\hbar\omega_y\left(n_y+\frac12\right)
+\lambda\frac{\hbar^2}{m^2\omega_x\omega_y}
\left(n_x+\frac12\right)
\left(n_y+\frac12\right)
}.
$$

For the ground state in the chapter's units,

$$
E_{00}^{(0)}=\frac12(1+1.4)=1.2,
$$

and

$$
\Delta E_{00}^{(1)}
=\frac{\lambda}{4\omega_x\omega_y}
=\frac{0.08}{4(1)(1.4)}
\approx0.01429.
$$

Hence weak-coupling theory predicts approximately

$$
E_{00}\approx1.21429
$$

on the infinite plane. A finite-difference result will also contain spatial and
domain errors, so it should approach this perturbative estimate only when the
coupling is sufficiently weak and the grid is converged.

The operator $x^2$ connects oscillator quantum numbers differing by $0$ or
$\pm2$. Consequently, $x^2y^2$ mixes product states while preserving both
parities. This explains why the exact coupled eigenfunction is usually a sum
of several oscillator product states rather than one product.

### 6.4 Why direct diagonalization contains higher-order mixing

Suppose two unperturbed states are coupled. In their subspace the Hamiltonian
has the form

$$
H=
\begin{pmatrix}
E_a^{(0)} & \lambda W_{ab}\\
\lambda W_{ab}^* & E_b^{(0)}
\end{pmatrix}.
$$

Direct diagonalization gives

$$
E_{\pm}
=\frac{E_a^{(0)}+E_b^{(0)}}{2}
\pm
\sqrt{
\left(\frac{E_a^{(0)}-E_b^{(0)}}{2}\right)^2
+\lambda^2|W_{ab}|^2
}.
$$

The square root contains the effect of the coupling without expanding it in a
finite power series. The grid calculation performs the same type of direct
diagonalization for many coupled basis states at once.

For this reason, the numerical result is nonperturbative with respect to
$\lambda$ for the chosen finite grid. It is not exact for the continuous
problem: it still has grid-spacing, finite-domain, and eigensolver errors.

## 7. Hellmann–Feynman prediction

For an exact normalized eigenstate of the coupled Hamiltonian,

$$
\frac{dE_n}{d\lambda}
=\left\langle\psi_n\left|
\frac{\partial\hat H}{\partial\lambda}
\right|\psi_n\right\rangle.
$$

Since

$$
\frac{\partial\hat H}{\partial\lambda}=x^2y^2,
$$

we obtain

$$
\boxed{
\frac{dE_n}{d\lambda}=\langle x^2y^2\rangle_n\ge0
}.
$$

Every isolated energy level must therefore increase as the positive coupling
is strengthened. This gives a useful numerical experiment even without exact
energies.

## 8. Why the kinetic operator is still a Kronecker sum

The potential is nonseparable, but the Cartesian Laplacian always satisfies

$$
\nabla^2=\partial_x^2+\partial_y^2.
$$

With the chapter's C-order flattening, the discrete space is ordered as
$\mathcal H_y\otimes\mathcal H_x$. The kinetic operator therefore has the form

$$
\hat T=I_y\otimes\hat T_x+\hat T_y\otimes I_x.
$$

This identity concerns the derivative operator only. It does not imply

$$
\hat V=I_y\otimes\hat V_x+\hat V_y\otimes I_x.
$$

For the coupled problem, the potential is instead one full diagonal operator:

$$
\hat V=\operatorname{diag}\bigl(V(x_i,y_j)\bigr).
$$

The complete Hamiltonian acts on the full $N_xN_y$-dimensional state vector.
The eigenvectors are reshaped back into 2D functions only after the full matrix
eigenproblem has been solved.

## 9. Separable oscillator as a calibration problem

Setting $\lambda=0$ gives the anisotropic oscillator

$$
V_0(x,y)=\frac12m\omega_x^2x^2
+\frac12m\omega_y^2y^2.
$$

Its exact spectrum is

$$
E_{n_xn_y}^{(0)}
=\hbar\omega_x\left(n_x+\frac12\right)
+\hbar\omega_y\left(n_y+\frac12\right).
$$

This case calibrates the general solver. It tests whether the 2D indexing,
kinetic operator, potential sampling, normalization, and eigensolver reproduce
a known result.

For $\omega_x=\omega_y=\omega$, the states $(1,0)$ and $(0,1)$ are degenerate.
The discrete calculation preserves that degeneracy only when the domain and
grid treat $x$ and $y$ identically.

## 10. Observables for the coupled problem

For a normalized state,

$$
\langle A\rangle_n
=\int_\Omega\psi_n^*(x,y)
\hat A\psi_n(x,y)\,dx\,dy.
$$

Useful position moments include

$$
\langle x\rangle,
\quad
\langle y\rangle,
\quad
\langle x^2\rangle,
\quad
\langle y^2\rangle,
\quad
\langle x^2y^2\rangle.
$$

The first two check parity. The next two measure the extent of the state along
each axis. The final moment measures the coupling contribution to the energy:

$$
\langle V_{\mathrm{coupling}}\rangle
=\lambda\langle x^2y^2\rangle.
$$

A useful measure of coordinate correlation is

$$
C_{x^2y^2}
=\langle x^2y^2\rangle
-\langle x^2\rangle\langle y^2\rangle.
$$

It vanishes for a probability density that factorizes exactly into independent
$x$ and $y$ parts. A nonzero value shows that the spatial distribution contains
coordinate correlations.

## 11. Validation without an exact spectrum

The nonseparable problem must be supported by several independent checks:

1. The Hamiltonian is Hermitian.
2. Eigenvectors are normalized and mutually orthogonal.
3. Eigenpair residual norms are small.
4. Energies stabilize when both $N_x$ and $N_y$ increase.
5. Energies stabilize when the four boundaries move outward.
6. NumPy and PyTorch agree on the same small grid.
7. Eigenstates obey the exact parity predictions.
8. Energies increase with positive $\lambda$.
9. Weak-coupling results approach perturbation theory.
10. Setting $\lambda=0$ reproduces the separable oscillator spectrum.

A refined-grid result is a numerical reference, not an exact answer. It must
use a sufficiently large domain and smaller spacings than every calculation
being judged against it.

## 12. Scope of this chapter

The chapter treats one spinless particle on a uniform Cartesian grid with a
real scalar potential. It does not cover magnetic vector potentials, complex
absorbers, periodic boundaries, adaptive meshes, or position-dependent mass.

Unlike radial hydrogen, no symmetry is used to eliminate a coordinate. The
unknown genuinely occupies a 2D mesh. A 3D Cartesian discretization would use
the same construction with a third Kronecker term and a volume weight. It is
not a separate chapter because that extension adds scaling cost rather than a
new numerical idea.
