# The spinful Fermi-Hubbard dimer

## Model and units

There are two sites and one spatial orbital on each site. Each orbital admits
one up electron and one down electron. Two equal-spin electrons cannot occupy
the same orbital. Opposite spins can share a site. This differs from the
spin-polarized model in the previous chapter.

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

## Spin orbitals and fermion signs

Order the spin orbitals as $(0\uparrow,0\downarrow,1\uparrow,1\downarrow)$.
A mask stores the four occupations with orbital zero as its lowest bit.
Define each Fock state by creation operators in ascending orbital order:

$$
|n_0n_1n_2n_3\rangle=
(c_0^\dagger)^{n_0}(c_1^\dagger)^{n_1}
(c_2^\dagger)^{n_2}(c_3^\dagger)^{n_3}|0\rangle.
$$

Fermion operators satisfy $\{c_p,c_q^\dagger\}=\delta_{pq}I$ and
$\{c_p,c_q\}=0$. Moving $c_q$ past the occupied lower orbitals gives

$$
c_q|n\rangle=(-1)^{\sum_{p<q}n_p}n_q|n-e_q\rangle.
$$

The creation rule has the same sign and the factor $1-n_q$.
Creation and annihilation change particle number. Build them on the full
16-state Fock space before forming number-conserving products. Projecting each
single operator into the fixed-number sector first would give zero.

## Four-state Hamiltonian

Choose the sector basis

$$
|D_0\rangle=c_0^\dagger c_1^\dagger|0\rangle,\quad
|a\rangle=c_0^\dagger c_3^\dagger|0\rangle,\quad
|b\rangle=c_1^\dagger c_2^\dagger|0\rangle,\quad
|D_1\rangle=c_2^\dagger c_3^\dagger|0\rangle.
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
