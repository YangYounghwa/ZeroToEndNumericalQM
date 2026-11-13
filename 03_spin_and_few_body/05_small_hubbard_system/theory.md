# The spinful Fermi-Hubbard dimer

## Model and units

There are two sites and one spatial orbital on each site. Each orbital admits
one up electron and one down electron. Two equal-spin electrons cannot occupy
the same orbital. Opposite spins can share a site. This differs from the
spin-polarized model in the previous chapter.

## Many-electron states and occupation numbers

A **spin orbital** specifies both a spatial orbital and a spin state.
This dimer has four spin orbitals:

$$
(0\uparrow,\ 0\downarrow,\ 1\uparrow,\ 1\downarrow).
$$

Each spin orbital is either empty or occupied by one electron. Write a basis
state as $|n_{0\uparrow},n_{0\downarrow},n_{1\uparrow},n_{1\downarrow}\rangle$,
where each occupation number is 0 or 1. For example:

| Occupation state | Meaning |
| --- | --- |
| $|0,0,0,0\rangle$ | No electrons |
| $|1,0,0,0\rangle$ | One up electron at site 0 |
| $|1,0,0,1\rangle$ | One up electron at site 0 and one down electron at site 1 |
| $|1,1,0,0\rangle$ | Two electrons at site 0, with opposite spins |

The entries label spin orbitals, not individual electrons. We do not assign
names such as electron A and electron B. The occupation basis describes
identical electrons while the operator rules enforce their exchange signs.

A many-electron state can be a superposition of occupation states. For example,

$$
|\Psi\rangle=\alpha|1,0,0,1\rangle+\beta|0,1,1,0\rangle,
\qquad |\alpha|^2+|\beta|^2=1.
$$

Both terms contain two electrons. Their coefficients are probability
amplitudes for whole configurations. They are not separate wavefunctions for
two labeled electrons. A measurement of all occupations gives the first
configuration with probability $|\alpha|^2$ and the second with $|\beta|^2$.
The relative phase also matters for measurements that mix these configurations.

The **vacuum**, $|\mathrm{vac}\rangle=|0,0,0,0\rangle$, is the normalized state
with no electrons in these four spin orbitals. The **Fock space** contains all
occupation states, including different total electron counts. Its dimension
is $2^4=16$. The subspace with exactly two electrons has dimension
$\binom42=6$. Requiring one up and one down electron leaves four states.
The two other two-electron states have equal spins on different sites.

## Creation and annihilation operators

The letter $c$ denotes an operator, not a probability amplitude.
The dagger distinguishes creation from annihilation:

| Operator | Action |
| --- | --- |
| $c^\dagger_{i\sigma}$ | Adds an electron to spin orbital $(i,\sigma)$ |
| $c_{i\sigma}$ | Removes an electron from spin orbital $(i,\sigma)$ |

The dagger means Hermitian adjoint. In a matrix representation, it means
complex conjugation followed by transposition. Creation and annihilation
are adjoints of each other. Neither is generally Hermitian by itself.

First consider just one spin orbital. Its basis is $|0\rangle$ for empty and
$|1\rangle$ for occupied. The operators act as

$$
c^\dagger|0\rangle=|1\rangle,\qquad c|1\rangle=|0\rangle,
\qquad c^\dagger|1\rangle=0,\qquad c|0\rangle=0.
$$

The last two results enforce exclusion and the absence of an electron to
remove. Here $0$ is the **zero vector**. It is not the normalized empty state
$|0\rangle$. The zero vector cannot describe a physical state.

In the ordered basis $(|0\rangle,|1\rangle)$, these rules give

$$
c=\begin{pmatrix}0&1\\0&0\end{pmatrix},\qquad
c^\dagger=\begin{pmatrix}0&0\\1&0\end{pmatrix}.
$$

With several spin orbitals, the same occupation changes apply, but a fermion
sign can appear. The next section on fermion signs derives that sign from a
fixed orbital order. Copying these two-by-two matrices to different orbitals
with ordinary tensor products alone would miss the required signs.

### Counting and moving electrons

Operators act on the state from right to left. The number operator is

$$
n_{i\sigma}=c^\dagger_{i\sigma}c_{i\sigma}.
$$

For an occupied spin orbital, annihilation removes its electron and creation
restores it. The two fermion signs cancel. For an empty spin orbital,
annihilation gives zero. Thus the number operator has eigenvalues 1 and 0.
For the single-orbital matrices above, $c^\dagger c=\operatorname{diag}(0,1)$.
Its expectation value is the probability that the spin orbital is occupied.

Now consider $c^\dagger_{0\sigma}c_{1\sigma}$. It first removes an electron
from site 1, then adds an electron with the same spin at site 0. This is a
hopping operator. It gives zero if the source spin orbital is empty or the
destination spin orbital is already occupied. Its adjoint moves the electron
in the reverse direction. Adding both directions gives a Hermitian term.

For example, the one-electron state obeys

$$
c^\dagger_{0\uparrow}c_{1\uparrow}|0,0,1,0\rangle
=|1,0,0,0\rangle.
$$

The intermediate state is the vacuum. The initial and final states both
contain one electron. Likewise, hopping in a two-electron state passes through
a one-electron intermediate state and returns to two electrons. These are
algebraic steps in applying the operator product, not a physical sequence in
which an electron leaves the closed system.

Finally, $n_{i\uparrow}n_{i\downarrow}$ is 1 only when both spin orbitals at
site $i$ are occupied. Multiplication by $U$ assigns energy $U$ to that
double occupancy. This counts an electron pair on a site, not two separate
energy contributions of $U$.

## Hamiltonian and conserved electron counts

Use $\hbar=1$. Let $t$ be the real hopping energy, $U$ the on-site interaction,
and $\Delta=v_0-v_1$ the site bias. Set $v_0=\Delta/2$ and $v_1=-\Delta/2$.
Positive $U$ penalizes double occupancy. Negative $U$ favors it.
There is one open bond and no coupling to an environment.

$$
H=-t\sum_{\sigma=\uparrow,\downarrow}
(c^\dagger_{0\sigma}c_{1\sigma}+c^\dagger_{1\sigma}c_{0\sigma})
+U\sum_{i=0}^1 n_{i\uparrow}n_{i\downarrow}
+\frac{\Delta}{2}(n_0-n_1).
$$

Here $n_{i\sigma}=c^\dagger_{i\sigma}c_{i\sigma}$ and
$n_i=n_{i\uparrow}+n_{i\downarrow}$. Hopping moves an electron without changing
its spin. Each spin count commutes with $H$. Restrict to
$N_\uparrow=N_\downarrow=1$. This is the half-filled dimer: two electrons on two
sites. The restriction is exact for initial states in this sector.

Explicitly, $N_\sigma=\sum_i n_{i\sigma}$ and $N=N_\uparrow+N_\downarrow$.
Creation raises N by one; annihilation lowers it by one. Each hopping term
contains one of each, while the other terms only count occupations. Therefore
the complete Hamiltonian preserves N even though an individual $c$ does not.

## Spin orbitals and fermion signs

Order the spin orbitals as $(0\uparrow,0\downarrow,1\uparrow,1\downarrow)$.
For compact notation, label them by $q=0,1,2,3$ in that order. Thus
$c_0=c_{0\uparrow}$, $c_1=c_{0\downarrow}$, $c_2=c_{1\uparrow}$, and
$c_3=c_{1\downarrow}$. In the occupation ket below, $n_q$ denotes a spin-orbital
occupation. This differs from the site count $n_i=n_{i\uparrow}+n_{i\downarrow}$
in the Hamiltonian.
A mask stores the four occupations with orbital zero as its lowest bit.
Define each Fock state by creation operators in ascending orbital order:

$$
|n_0n_1n_2n_3\rangle=
(c_0^\dagger)^{n_0}(c_1^\dagger)^{n_1}
(c_2^\dagger)^{n_2}(c_3^\dagger)^{n_3}|\mathrm{vac}\rangle.
$$

Fermion operators satisfy $\{c_p,c_q^\dagger\}=\delta_{pq}I$ and
$\{c_p,c_q\}=0$. Here $\{A,B\}=AB+BA$ is the anticommutator, and
$\delta_{pq}$ is 1 for $p=q$ and 0 otherwise. For different orbitals,
interchanging two creation operators changes the sign:

$$
c_p^\dagger c_q^\dagger=-c_q^\dagger c_p^\dagger.
$$

This algebra builds fermion antisymmetry into the occupation description.
Moving $c_q$ past the occupied lower orbitals gives

$$
c_q|n\rangle=(-1)^{\sum_{p<q}n_p}n_q|n-e_q\rangle.
$$

Here $e_q$ changes only occupation q by one. The right side is zero when
$n_q=0$; no state with negative occupation is introduced. Creation has the
same sign and the factor $1-n_q$. For example,

$$
c_0|1,1,0,0\rangle=|0,1,0,0\rangle,\qquad
c_1|1,1,0,0\rangle=-|1,0,0,0\rangle.
$$

Removing orbital 1 passes one occupied lower orbital, so it gives a minus sign.
Removing orbital 0 passes none. The signs depend on the fixed basis convention;
physical predictions do not change when that convention is changed consistently.

Creation and annihilation change particle number. Build them on the full
16-state Fock space before forming number-conserving products. Projecting each
single operator into the fixed-number sector first would give zero.

## Four-state Hamiltonian

Choose the sector basis

$$
|D_0\rangle=c_0^\dagger c_1^\dagger|\mathrm{vac}\rangle,\quad
|a\rangle=c_0^\dagger c_3^\dagger|\mathrm{vac}\rangle,\quad
|b\rangle=c_1^\dagger c_2^\dagger|\mathrm{vac}\rangle,\quad
|D_1\rangle=c_2^\dagger c_3^\dagger|\mathrm{vac}\rangle.
$$

Their masks are $(3,9,6,12)$. For example, $c_2^\dagger c_0|D_0\rangle=-|b\rangle$.
Multiplication by $-t$ gives the positive matrix entry below.

$$
H=\begin{pmatrix}
U+\Delta&-t&t&0\\
-t&0&0&-t\\
t&0&0&t\\
0&-t&t&U-\Delta
\end{pmatrix}.
$$

The diagonal interaction counts shared sites, not individual electrons.
The bias gives energies $+\Delta$ and $-\Delta$ to the two doublons.
It gives zero to states with one electron on each site.

## Exact symmetric dimer

Set $\Delta=0$. Define
$|S\rangle=(|a\rangle-|b\rangle)/\sqrt{2}$,
$|T_0\rangle=(|a\rangle+|b\rangle)/\sqrt{2}$, and
$|D_\pm\rangle=(|D_0\rangle\pm|D_1\rangle)/\sqrt{2}$.
The triplet has energy zero. The odd doublon has energy $U$.
In the $(S,D_+)$ basis the remaining block is

$$
H_{S,D_+}=\begin{pmatrix}0&-2t\\-2t&U\end{pmatrix}.
$$

Its characteristic equation is $E^2-UE-4t^2=0$. Thus

$$
E_\pm=\frac{U\pm\sqrt{U^2+16t^2}}{2}.
$$

The four sector energies are $E_-,0,U,E_+$, sorted as needed.
The other two triplet spin components are outside this sector.
For nonzero $t$, the ground state is nondegenerate. The Hellmann-Feynman
identity gives its total double occupancy:

$$
D=\left\langle\sum_i n_{i\uparrow}n_{i\downarrow}\right\rangle
=\frac{\partial E_-}{\partial U}
=\frac12\left(1-\frac{U}{\sqrt{U^2+16t^2}}\right).
$$

At $U=0$, $D=1/2$. Repulsion suppresses $D$; attraction raises it toward one.
For $U>0$ and $U\gg |t|$, the singlet-triplet gap is

$$
J=0-E_- = \frac{8t^2}{\sqrt{U^2+16t^2}+U}
=\frac{4t^2}{U}+O(t^4/U^3).
$$

The rationalized expression avoids subtracting nearly equal numbers.
The effective spin interaction is $J(\mathbf S_0\cdot\mathbf S_1-1/4)$
in the singly occupied space. The constant makes the triplet energy zero.
The formula $4t^2/U$ is a large-repulsion approximation, not an exact gap.

## Dynamics and measurements

For eigenpairs $(E_k,v_k)$, evolution is
$\psi(\tau)=\sum_k v_k e^{-iE_k\tau}v_k^\dagger\psi(0)$.
Time is $\tau$ here; $t$ denotes hopping. A Hermitian, static Hamiltonian
conserves norm and energy. Negative time gives reverse evolution.
Starting in $S$ at zero bias gives the exact doublon probability

$$
D(\tau)=\frac{16t^2}{U^2+16t^2}
\sin^2\left(\frac{\sqrt{U^2+16t^2}\,\tau}{2}\right).
$$

This follows from the two-state block after subtracting $U I/2$.
For amplitudes $(d_0,a,b,d_1)$, $D=|d_0|^2+|d_1|^2$ and
$\langle n_0\rangle=2|d_0|^2+|a|^2+|b|^2$.
The two site counts sum to two. They do not describe a normalized
single-particle probability distribution.

## Scope and reading

This chapter solves a finite dimer exactly within a conserved sector.
It does not establish a bulk phase transition. No bosonic occupation cutoff
or continuum grid is used. A batch contains separate state vectors, not more
electrons in one system.

For a broader treatment of the same model, including site asymmetry, see
[Carrascal et al., The Hubbard Dimer](https://arxiv.org/abs/1502.02194).
The formulas above are derived directly from the displayed four-state matrix.
