# Two coupled spin-1/2 sites

## 1. The composite Hilbert space

Each site has basis `|+z>` and `|-z>`. The joint space is the tensor product
$\mathcal H_1\otimes\mathcal H_2$, with dimension `2*2=4`. Fix the order

$$
|++\rangle,\quad |+-\rangle,\quad |-+\rangle,\quad |--\rangle.
$$

The first sign always belongs to site 1. A general joint state has four
complex amplitudes and norm $\sum_{i,j}|c_{ij}|^2=1$. The sites are separately
addressable spin degrees of freedom; this effective model does not describe
particle motion or require antisymmetrizing spatial wavefunctions.

For separate spinors $|a\rangle=(a_+,a_-)^T$, $|b\rangle=(b_+,b_-)^T$,
their product state is

$$
|a\rangle\otimes|b\rangle=
(a_+b_+,\ a_+b_-,\ a_-b_+,\ a_-b_-)^T.
$$

The inner product factorizes:
$(\langle a|\otimes\langle b|)(|c\rangle\otimes|d\rangle)
=\langle a|c\rangle\langle b|d\rangle$.
Normalized factors therefore produce a normalized joint state. A general
four-amplitude state need not have this product form.

An array containing two independent spinors is a batch, not a joint state.
Keeping only two spinors cannot represent all correlations of the coupled pair.

For example, in $(|+-\rangle+|-+\rangle)/\sqrt2$, each individual z outcome
is random, but the pair always has opposite signs. Assigning both sites their
own equal-superposition spinor would also produce `|++>` and `|-->` outcomes,
so it would describe a different preparation. The four joint amplitudes retain
information that separate local probabilities lose.

## 2. Local operators

An operator acting on the first site must leave the second untouched:

$$
\sigma_{1a}=\sigma_a\otimes I,\qquad
\sigma_{2a}=I\otimes\sigma_a,\qquad
S_{sa}=\frac\hbar2\sigma_{sa}.
$$

For example, $(\sigma_x\otimes I)|+-\rangle=|--\rangle$, whereas
$(I\otimes\sigma_x)|+-\rangle=|++\rangle$.
The Kronecker-product rule
$(A\otimes B)(C\otimes D)=AC\otimes BD$ shows that operators on different
sites commute. Operators on the same site retain the Pauli commutators.

## 3. Exchange and local magnetic fields

Use the isotropic exchange model

$$
H=\frac{J}{\hbar^2}\mathbf S_1\cdot\mathbf S_2
-\gamma(\mathbf B_1\cdot\mathbf S_1+\mathbf B_2\cdot\mathbf S_2)
=\frac J4\sum_a\sigma_a\otimes\sigma_a
-\frac{\gamma\hbar}{2}\sum_a(B_{1a}\sigma_{1a}+B_{2a}\sigma_{2a}).
$$

`J` has units of energy, so the exchange frequency scale is `|J|/hbar`.
`gamma` is signed, with the same convention as the previous chapter.
Both sites use the same gamma here, but may have different fields. The main
example sets `J=hbar=1` and both fields to zero.

This Hamiltonian is a specified effective interaction. Deriving `J` from
microscopic electron motion belongs to later lattice models. The word exchange
does not mean that this calculation moves the particles between sites.

Multiplying the Pauli matrices explicitly gives

$$
H_{ex}=\frac J4
\begin{pmatrix}
1&0&0&0\\
0&-1&2&0\\
0&2&-1&0\\
0&0&0&1
\end{pmatrix}.
$$

The off-diagonal elements couple `|+->` and `|-+>`. They cannot be obtained
from two independent single-spin evolutions.

The diagonal entries supply phases to the basis states; the off-diagonal
entries transfer amplitude between them. In particular, the exchange term
flips the two opposite signs together and preserves their total z spin.
The parallel states are already exchange eigenstates. An initial opposite-spin
product is not an eigenstate, which is why it can develop nontrivial dynamics.

## 4. Derive singlet and triplet energies

Let $\mathbf S_{tot}=\mathbf S_1+\mathbf S_2$. Expanding its square yields

$$
\mathbf S_1\cdot\mathbf S_2
=\tfrac12(S_{tot}^2-S_1^2-S_2^2),\qquad
S_1^2=S_2^2=\tfrac34\hbar^2I.
$$

Two spin halves combine into total spin one or zero. Equivalently, directly
diagonalize the displayed four-by-four matrix to find

$$
|t_+\rangle=|++\rangle,\quad
|t_0\rangle=(|+-\rangle+|-+\rangle)/\sqrt2,\quad
|t_-\rangle=|--\rangle,
$$

$$
|s\rangle=(|+-\rangle-|-+\rangle)/\sqrt2.
$$

The triplet has $S_{tot}^2=2\hbar^2$, giving `E_t=J/4`. The singlet has
$S_{tot}^2=0$, giving `E_s=-3J/4`. At zero field positive `J` favors the
singlet (antiferromagnetic sign convention); negative `J` favors the triplet.
Different texts can define the exchange coefficient with a different sign or
factor, so always compare their Hamiltonians before comparing values of `J`.

## 5. Exact exchange dynamics

Start with $|+-\rangle=(|t_0\rangle+|s\rangle)/\sqrt2$. Give each energy
eigenstate its phase and transform back:

$$
|\psi(t)\rangle=e^{iJt/(4\hbar)}
\left[\cos\left(\frac{Jt}{2\hbar}\right)|+-\rangle
-i\sin\left(\frac{Jt}{2\hbar}\right)|-+\rangle\right].
$$

Thus $P_{-+}(t)=\sin^2(Jt/(2\hbar))$, and complete transfer first occurs
at $t=\pi\hbar/|J|$ when `J` is nonzero. Probabilities return at
$2\pi\hbar/|J|$; the spinor may differ by a global phase.

At half the transfer time for `J>0`, ignoring global phase, the state is
$(|+-\rangle-i|-+\rangle)/\sqrt2$. Both local spin averages vanish, but
the spins are perfectly anticorrelated when measured along z.

The transfer is interference between the singlet and triplet contributions.
Their energy difference is J, so their relative phase advances at rate
$J/\hbar$. Recombining them in the product basis produces the cosine and sine
amplitudes above. This explains why the transfer time is determined by an
energy difference, and why a common shift of both energies cannot change the
transfer probability.

## 6. Local expectations, correlations, and product states

Calculate the local Bloch vectors $r_{sa}=\langle\sigma_{sa}\rangle$,
joint correlations $C_{ab}=\langle\sigma_{1a}\sigma_{2b}\rangle$, and
connected correlations $C^c_{ab}=C_{ab}-r_{1a}r_{2b}$.
For a product pure state, the inner-product factorization makes every connected
correlation zero. Exchange dynamics can make them nonzero.

For the example above, $r_{1z}=\cos(Jt/\hbar)$, $r_{2z}=-r_{1z}$,
$C_{zz}=-1$, and $C^c_{zz}=-\sin^2(Jt/\hbar)$.
Joint z-measurement probabilities are the four squared amplitudes. Marginal
probabilities follow by summing over the other site's sign.

A direct product test for a nonzero pure state uses its coefficient matrix:

$$
M=\begin{pmatrix}c_{++}&c_{+-}\\c_{-+}&c_{--}\end{pmatrix}.
$$

A product has `M=a*b.T`, rank one, so `det(M)=0`. Conversely, a nonzero two-by-two
matrix with zero determinant has rank one and can be factored. A nonzero
determinant therefore proves that this pure state is entangled. For a normalized
state, `|det(M)|` ranges from zero to one half. The half-transfer state reaches
one half. This is a pure-state test; connected correlations alone are not a
general mixed-state entanglement criterion. Bell states and reduced density
matrices will make that distinction precise in the next chapter.

## 7. Unequal longitudinal fields

For z-directed fields, total z spin is conserved and `|+->,|-+>` form a closed
subspace. Define the energy detuning
$\delta=\gamma\hbar(B_{1z}-B_{2z})$. In that subspace,

$$
H_0=-\frac J4 I+\frac12(J\sigma_x-\delta\sigma_z).
$$

The common part of the longitudinal field cancels because total z spin is zero.
Apply the single-spin Pauli exponential with $D=\sqrt{J^2+\delta^2}$:

$$
|\psi(t)\rangle=e^{iJt/(4\hbar)}
\left[\left(\cos\frac{Dt}{2\hbar}+i\frac\delta D\sin\frac{Dt}{2\hbar}\right)|+-\rangle
-i\frac JD\sin\frac{Dt}{2\hbar}|-+\rangle\right].
$$

Consequently, $P_{-+}=J^2/(J^2+\delta^2)\,\sin^2(Dt/(2\hbar))$.
Detuning reduces the maximum transfer below one. This is a physical change,
not a numerical error. The zero-splitting limit is evaluated continuously.

## 8. Conservation laws depend on the fields

Norm and energy are conserved for all static Hermitian Hamiltonians here.
At zero field, isotropic exchange also conserves all total-spin components and
total spin squared. A uniform field conserves total spin squared and the total
component along that field; it need not conserve the laboratory z component.
Unequal longitudinal fields conserve total z but can mix singlet and triplet.
General unequal transverse fields need conserve neither of these spin quantities.
Check commutators before choosing an invariant as a numerical test.
