# Spin-1/2 in a static magnetic field

## 1. From a spatial wavefunction to a spinor

We now keep only one particle's spin degree of freedom. Its position is not
part of the model. Spin is intrinsic angular momentum, not a rotating material
ball. A measurement of its z component has two possible outcomes, `+hbar/2`
and `-hbar/2`, with orthonormal basis states

$$
|+z\rangle=\begin{pmatrix}1\\0\end{pmatrix},\qquad
|-z\rangle=\begin{pmatrix}0\\1\end{pmatrix}.
$$

A pure state is $|\psi\rangle=a|+z\rangle+b|-z\rangle$. Its coefficients
are complex amplitudes, normalized by $|a|^2+|b|^2=1$. There is no `dx` or
area element. The two basis vectors span this model's complete Hilbert space.

The bra $\langle\psi|=(a^*,b^*)$ is the conjugate transpose of the ket.
The inner product is $\langle\psi|\phi\rangle=a^*c+b^*d$, and the expected
value of an operator is $\langle A\rangle=\langle\psi|A|\psi\rangle$.
Multiplying the whole state by $e^{i\chi}$ changes no measurement probability.
Changing the relative phase of `a` and `b` generally does.

Up to a global phase, parameterize a normalized pure spin as

$$
|\psi(\theta,\phi)\rangle=
\begin{pmatrix}\cos(\theta/2)\\e^{i\phi}\sin(\theta/2)\end{pmatrix},
\qquad 0\leq\theta\leq\pi.
$$

For example, `theta=pi/2, phi=0` is `+x`, and `theta=pi/2, phi=pi/2` is `+y`.
Both give equal probabilities when measured along z, but have different phases.

## 2. Spin operators and Pauli algebra

In this basis, $S_a=(\hbar/2)\sigma_a$, where

$$
\sigma_x=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
\sigma_y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
\sigma_z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.
$$

Each matrix is Hermitian and squares to the identity. Direct multiplication
gives $\sigma_a\sigma_b=\delta_{ab}I+i\epsilon_{abc}\sigma_c$, hence

$$
[\sigma_a,\sigma_b]=2i\epsilon_{abc}\sigma_c,\qquad
S^2=S_x^2+S_y^2+S_z^2=\frac34\hbar^2I.
$$

The eigenvalues of each `S_a` are `+hbar/2` and `-hbar/2`. The components do
not commute, so a spin cannot have definite values for all three simultaneously.
The Pauli representation of a two-state magnetic Hamiltonian is discussed in
[Feynman, Volume III, Chapter 11](https://www.feynmanlectures.caltech.edu/III_11.html).

## 3. Hamiltonian and sign convention

Let the magnetic moment be $\boldsymbol\mu=\gamma\mathbf S$, where `gamma`
is the signed gyromagnetic ratio. The interaction energy is

$$
H=-\boldsymbol\mu\cdot\mathbf B
=-\frac{\gamma\hbar}{2}\mathbf B\cdot\boldsymbol\sigma
=\frac\hbar2\boldsymbol\Omega\cdot\boldsymbol\sigma,
\qquad\boldsymbol\Omega=-\gamma\mathbf B.
$$

Keep the sign of `gamma`: changing it reverses precession. The examples use
`gamma=+1` as a generic dimensionless case, not an electron-specific constant.
The angular frequency magnitude is $\omega=|\gamma|\,|\mathbf B|$.
`gamma*B` has units of inverse time; `hbar*omega` has units of energy.
Numerical examples use `hbar=1` and an arbitrary consistent time/field scale.

Writing the matrix explicitly gives

$$
H=-\frac{\gamma\hbar}{2}
\begin{pmatrix}B_z&B_x-iB_y\\B_x+iB_y&-B_z\end{pmatrix}.
$$

Its two energies are $E_\pm=\pm\hbar\omega/2$. An energy eigenstate gains
only a phase during time evolution, while a superposition can change its
measurement probabilities.

## 4. Derive the exact constant-field propagator

For constant `H`, Schrödinger's equation has solution
$|\psi(t)\rangle=U(t)|\psi(0)\rangle$, with $U(t)=e^{-iHt/\hbar}$.
If $\mathbf n=\boldsymbol\Omega/\omega$, the Pauli algebra gives
$(\mathbf n\cdot\boldsymbol\sigma)^2=I$. Separate even and odd terms in
the exponential series:

$$
U(t)=\cos(\omega t/2)I
-i\sin(\omega t/2)\mathbf n\cdot\boldsymbol\sigma.
$$

The expression is continuous at zero field, where `U=I`. A form that avoids
dividing by `omega` is

$$
U(t)=\cos(\omega t/2)I
-i\frac{t}{\hbar}\operatorname{sinc}(\omega t/2)H,
\qquad\operatorname{sinc}(z)=\frac{\sin z}{z}.
$$

PyTorch's `sinc(u)` means `sin(pi*u)/(pi*u)`, so the code passes `z/pi`.
The exact rotation conserves norm and the expectation of this static `H`.

## 5. Bloch vectors and spin measurements

Define the dimensionless Bloch vector $\mathbf r=\langle\boldsymbol\sigma\rangle$.
For the spinor above,

$$
\mathbf r=(2\operatorname{Re}(a^*b),\,2\operatorname{Im}(a^*b),\,|a|^2-|b|^2)
=(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta).
$$

A normalized pure state has $|\mathbf r|=1$, and its expected angular momentum
is $\langle\mathbf S\rangle=(\hbar/2)\mathbf r$. The Bloch sphere is a
representation of spin states; it is not the particle's spatial trajectory.

For a unit measurement axis $\mathbf a$, the projectors are
$P_\pm=(I\pm\mathbf a\cdot\boldsymbol\sigma)/2$. Squaring them verifies
that they are projectors, and their sum is the identity. Therefore

$$
p_\pm=\langle\psi|P_\pm|\psi\rangle
=\frac{1\pm\mathbf a\cdot\mathbf r}{2}.
$$

These are probabilities of outcomes `+hbar/2` and `-hbar/2`, not the values
of the outcomes themselves. The code computes probabilities without sampling
random outcomes or evolving a post-measurement state.

## 6. Derive precession and check its direction

For a time-independent operator, $d\langle A\rangle/dt=(i/\hbar)\langle[H,A]\rangle$.
Insert the Pauli commutator to obtain

$$
\dot{\mathbf r}=\boldsymbol\Omega\times\mathbf r.
$$

Its solution is Rodrigues' rotation formula:

$$
\mathbf r(t)=\mathbf r_0\cos(\omega t)
+(\mathbf n\times\mathbf r_0)\sin(\omega t)
+\mathbf n(\mathbf n\cdot\mathbf r_0)[1-\cos(\omega t)].
$$

For `B=(0,0,1)`, `gamma=1`, and initial `+x`, this yields
$\mathbf r(t)=(\cos t,-\sin t,0)$. The minus sign in `r_y` is a useful
independent check. For `B=(1,0,0)` and initial `+z`, the probability of `-z` is
$\sin^2(t/2)$. More examples of spin time evolution and precession appear in
[MIT's time-evolution notes](https://ocw.mit.edu/courses/22-51-quantum-theory-of-radiation-interactions-fall-2012/485cf9e8ede00033f1c7d7b6eb6fecb2_MIT22_51F12_Ch5.pdf).

A rotation through `2*pi` gives `U=-I`: the spinor changes sign, while its
Bloch vector and probabilities return. At `4*pi`, the spinor itself returns.
The single-spin global sign is unobservable in these probabilities; it must
not be counted as a state error when comparing physical states up to phase.

## 7. Scope

The field is uniform and static, and the spin remains pure. There is no orbital
motion, relaxation, decoherence, or driving. Time-dependent fields require a
time-ordered evolution method and different energy checks. Composite states and
their density matrices are introduced with coupled spins in the next chapter.
