# Two-dimensional wave packets

## 1. Equation, coordinates, and probability

For one particle in a plane, write \(\mathbf r=(x,y)^T\) and

\[
i\hbar\partial_t\psi(x,y,t)=
\left[-\frac{\hbar^2}{2m}(\partial_x^2+\partial_y^2)+V(x,y)\right]\psi(x,y,t).
\]

This follows from the classical Hamiltonian \(H=(p_x^2+p_y^2)/(2m)+V\) by
replacing each momentum with \(-i\hbar\partial_\alpha\). The two derivative
terms act on independent spatial directions. A real potential gives Hermitian
evolution with appropriate boundaries.

The density is probability per area:

\[
\int\!\int |\psi|^2 dx\,dy=1,
\qquad \langle f\rangle=\int\!\int\psi^*f\psi\,dx\,dy.
\]

Multiplying Schrödinger's equation by \(\psi^*\), subtracting its conjugate,
and collecting derivatives yields \(\partial_t|\psi|^2+\nabla\cdot\mathbf j=0\),
where \(\mathbf j=(\hbar/m)\operatorname{Im}(\psi^*\nabla\psi)\).
The periodic computational box conserves probability but allows wraparound.

As before, use length unit \(\ell\), time unit \(m\ell^2/\hbar\), and energy
unit \(\hbar^2/(m\ell^2)\) to obtain `m=hbar=1`. Momentum arguments in this
chapter mean physical/dimensionless `p`, not wave number `k`: \(p=\hbar k\).

## 2. Free anisotropic Gaussian

When `V=0`, the Hamiltonian is a sum of independent x and y operators. An initial
product state remains a product of two free Gaussian solutions.
For coordinate \(u\in\{x,y\}\), position width \(\sigma_u\), center \(u_0\),
and momentum \(p_u\), define

\[
z_u(t)=1+\frac{i\hbar t}{2m\sigma_u^2},\qquad
\phi_u(u,t)=\frac{(2\pi\sigma_u^2)^{-1/4}}{\sqrt{z_u}}
\exp\left[-\frac{(u-u_0-p_ut/m)^2}{4\sigma_u^2z_u}
+\frac{i}{\hbar}\left(p_u(u-u_0)-\frac{p_u^2t}{2m}\right)\right].
\]

Substituting into the free Schrödinger equation verifies the solution; it is
also obtained by multiplying the initial Gaussian Fourier transform by
\(e^{-i\hbar k_u^2t/(2m)}\) and completing the square in the inverse transform.
Then \(\psi(x,y,t)=\phi_x(x,t)\phi_y(y,t)\), and

\[
\langle u\rangle=u_0+\frac{p_u}{m}t,\qquad
\operatorname{Var}(u)=\sigma_u^2+\frac{\hbar^2t^2}{4m^2\sigma_u^2},\qquad
\operatorname{Cov}(x,y)=0.
\]

Use unequal widths, momenta, and grid spacings so an accidental axis swap is
visible. The infinite-plane formula is a valid finite-box reference only when
the packet tails remain negligible at the seams.

## 3. Coupled quadratic potential

Choose

\[
V(\mathbf r)=\frac m2\mathbf r^TK\mathbf r,
\qquad
K=\begin{pmatrix}\omega_x^2&c\\c&\omega_y^2\end{pmatrix}.
\]

The cross term in the potential is `m*c*x*y`. Positive frequencies and
\(|c|<\omega_x\omega_y\) make `K` positive definite and the oscillator confined.
At the default values `omega_x=1`, `omega_y=1.3`, `c=0.35`, the potential is
\((x^2+0.7xy+1.69y^2)/2\).

This potential cannot be split into `Vx(x)+Vy(y)`. It can, however, be solved
by rotating coordinates. A real symmetric matrix has an orthogonal
eigendecomposition

\[
K=Q\operatorname{diag}(\nu_1^2,\nu_2^2)Q^T.
\]

Set \(\mathbf q=Q^T\mathbf r\). Orthogonal rotation preserves the Laplacian
and integration measure, so the Hamiltonian becomes two independent oscillators
with frequencies \(\nu_1,\nu_2\). These are the normal modes. Nonseparability
in the original coordinates does not mean the problem lacks an exact solution.

## 4. Ground-state covariance and a displaced packet

Define \(\Omega=Q\operatorname{diag}(\nu_1,\nu_2)Q^T\), the positive square root
of `K`. The ground state has the Gaussian form

\[
\psi_0(\mathbf r)=C\exp[-m\mathbf r^T\Omega\mathbf r/(2\hbar)],
\quad C=\left(\frac{m}{\pi\hbar}\right)^{1/2}(\det\Omega)^{1/4}.
\]

To check it, differentiate the Gaussian:

\[
\nabla^2\psi_0=
\left[\frac{m^2}{\hbar^2}\mathbf r^T\Omega^2\mathbf r
-\frac m\hbar\operatorname{tr}\Omega\right]\psi_0.
\]

The quadratic terms cancel in `H psi0` because \(\Omega^2=K\), leaving ground
energy \(E_g=\hbar(\nu_1+\nu_2)/2\).
Reading the covariance from the probability Gaussian gives

\[
\Sigma=\frac{\hbar}{2m}\Omega^{-1}.
\]

Its off-diagonal element generally differs from zero: the density ellipse is
tilted in x/y. Independent x/y Gaussian widths would not be this coupled
ground state.

Displace its center and multiply by a momentum phase:

\[
\psi(\mathbf r,0)=C
e^{-m(\mathbf r-\mathbf r_0)^T\Omega(\mathbf r-\mathbf r_0)/(2\hbar)}
e^{i\mathbf p_0^T(\mathbf r-\mathbf r_0)/\hbar}.
\]

In each normal mode this is a coherent state. Its covariance stays fixed, while
its center and momentum obey the classical oscillator equations.
Ehrenfest's theorem also gives these exactly:
\(\dot{\langle\mathbf r\rangle}=\langle\mathbf p\rangle/m\) and
\(\dot{\langle\mathbf p\rangle}=-mK\langle\mathbf r\rangle\).

## 5. Exact motion and state comparison

With \(\mathbf q_0=Q^T\mathbf r_0\) and \(\boldsymbol\pi_0=Q^T\mathbf p_0\),
each mode satisfies

\[
q_j(t)=q_{0j}\cos(\nu_jt)+\frac{\pi_{0j}}{m\nu_j}\sin(\nu_jt),
\]

\[
\pi_j(t)=\pi_{0j}\cos(\nu_jt)-m\nu_jq_{0j}\sin(\nu_jt).
\]

Rotate back with \(\mathbf r(t)=Q\mathbf q(t)\), \(\mathbf p(t)=Q\boldsymbol\pi(t)\).
Substitute these into the displaced Gaussian above to obtain the evolved state
up to a global phase. Remove that phase before comparing numerical states.
Do not assume the two frequencies give a common finite return period.

The conserved energy is

\[
E=\frac\hbar2(\nu_1+\nu_2)+\frac{\mathbf p_0^T\mathbf p_0}{2m}
+\frac m2\mathbf r_0^TK\mathbf r_0.
\]

This provides checks of full states, centers, correlations, and energy, rather
than only whether an animation looks plausible. The generic numerical
propagator also accepts sampled potentials that cannot be diagonalized this way.
