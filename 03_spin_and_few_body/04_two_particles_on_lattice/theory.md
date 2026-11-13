# Theory: two particles on a lattice

## 1. Sites and the one-particle Hamiltonian

Use an open chain with $L\ge2$ sites labeled $0,\ldots,L-1$. Each site has one
orthonormal orbital $|i\rangle$. The particle can hop between adjacent sites:

$$
h=-J\sum_{i=0}^{L-2}(|i+1\rangle\langle i|+|i\rangle\langle i+1|)
+\sum_{i=0}^{L-1}v_i|i\rangle\langle i|.
$$

$J$ and $v_i$ have units of energy. Positive $J$ gives negative off-diagonal
matrix entries. There is no link between the two end sites. A real $v_i$ makes
$h$ Hermitian. This is a discrete physical model, not a finite-difference
approximation to a continuous coordinate. Normalization has no spatial weight:

$$
|u\rangle=\sum_i u_i|i\rangle,\qquad \sum_i|u_i|^2=1.
$$

For $v_i=0$, the eigenvalue equation is
$-J(u_{i-1}+u_{i+1})=\epsilon u_i$. Set the missing endpoint amplitudes to
$u_{-1}=u_L=0$. A sine ansatz gives

$$
u_q(i)=\sqrt{\frac{2}{L+1}}\sin\frac{q\pi(i+1)}{L+1},\qquad
\epsilon_q=-2J\cos\frac{q\pi}{L+1},\quad q=1,\ldots,L.
$$

Substituting the sine into the difference equation uses
$\sin(a-b)+\sin(a+b)=2\sin a\cos b$. The endpoint condition selects the
allowed $q$. For $J>0$, $q=1$ has the lowest energy.

## 2. Distinguishable particles

First assign labels A and B. The product state $|i,j\rangle$ means A is at
site $i$ and B is at site $j$. There are $L^2$ states. Store them in C order:

$$
\operatorname{index}(i,j)=iL+j.
$$

Both particles experience the same $h$. With no interaction,

$$
H_{\mathrm{lab}}=h\otimes I+I\otimes h.
$$

The first term moves A and the second moves B. If $h|u_q\rangle=
\epsilon_q|u_q\rangle$, then

$$
H_{\mathrm{lab}}(|u_q\rangle\otimes|u_r\rangle)
=(\epsilon_q+\epsilon_r)|u_q\rangle\otimes|u_r\rangle.
$$

All ordered pairs $(q,r)$ occur. A batch of simulations is a separate array
axis; it does not add particles.

## 3. Exchange symmetry

### 3.1 Exchange is not spatial parity

The [oscillator chapter](../../01_stationary_one_particle/02_harmonic_oscillator/theory.md)
introduced spatial parity $\Pi$, with $(\Pi\psi)(x)=\psi(-x)$.
Here use $P_{12}$ for **particle exchange**. These operators answer different
questions:

| Operator | Action on a two-particle wavefunction | Meaning |
| --- | --- | --- |
| Spatial parity $\Pi$ | $\Psi(x_1,x_2)\mapsto\Psi(-x_1,-x_2)$ | Reflect both positions |
| Exchange $P_{12}$ | $\Psi(x_1,x_2)\mapsto\Psi(x_2,x_1)$ | Swap particle coordinates |

On the lattice, spatial reflection sends $|i,j\rangle$ to
$|L-1-i,L-1-j\rangle$. Exchange instead sends it to $|j,i\rangle$.
They can agree on a particular state without being the same operator.
Both square to the identity and have eigenvalues $\pm1$. This shared algebra
does not make their physical meanings identical. We use different symbols
to keep the distinction clear.

The [coupled-spin chapter](../02_two_coupled_spins/theory.md) also introduced
spin swap. For particles with spin, full particle exchange swaps both their
positions and their spin coordinates.

### 3.2 Why exchange eigenvalues are plus or minus one

In the labeled site basis,

$$
P_{12}|i,j\rangle=|j,i\rangle,\qquad P_{12}^2=I,\qquad
P_{12}^\dagger=P_{12}.
$$

Swapping twice restores the original state. If $P_{12}\Psi=\eta\Psi$, then
$P_{12}^2\Psi=\eta^2\Psi=\Psi$, so $\eta=+1$ or $-1$. Thus

$$
\Psi_B(x_2,x_1)=+\Psi_B(x_1,x_2),\qquad
\Psi_F(x_2,x_1)=-\Psi_F(x_1,x_2).
$$

The first state is symmetric under exchange. The second is antisymmetric.
The minus sign leaves the density unchanged, but it changes interference
between amplitudes. It cannot be discarded independently in each term of
an antisymmetric superposition.

### 3.3 What makes a particle a boson or fermion

Bosons have integer spin; fermions have half-integer spin. Electrons are
fermions. Photons are bosons. Under the usual assumptions of relativistic
quantum theory in three spatial dimensions, the spin-statistics theorem
connects these classes to symmetric and antisymmetric total states.
The lattice is a restricted model of such particles. Its Schrödinger equation
does not derive the spin-statistics theorem.

In this nonrelativistic calculation, we specify the particle type and impose
the corresponding state-space restriction. The equation $P_{12}^2=I$ explains
the two exchange eigenvalues. It does not decide which eigenvalue a physical
particle species must use.

Identical bosons require a symmetric total state. Identical fermions require
an antisymmetric total state. Here internal states are fixed and identical,
so the spatial state carries that symmetry. For fermions with different spin
states, antisymmetry applies to space and spin together. Such fermions can
occupy one spatial site in different spin orbitals. They are outside this model.
See [MIT's identical-particle notes](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/fb48dd49e8649557f1715f0415de4f4d_MIT8_06S18ch8.pdf).

### 3.4 Constructing allowed states and Pauli exclusion

For $i<j$, normalized spatial basis states are

$$
|i,j\rangle_+=\frac{|i,j\rangle+|j,i\rangle}{\sqrt2},\qquad
|i,j\rangle_-=\frac{|i,j\rangle-|j,i\rangle}{\sqrt2}.
$$

For two bosons on one site, use $|i,i\rangle_+=|i,i\rangle$. Do not divide
this state by $\sqrt2$. The antisymmetric combination with $i=j$ vanishes.
This gives Pauli exclusion for the single spin orbital at each site.

More generally, two identical fermions cannot occupy the same complete
one-particle state $|u\rangle$, since

$$
|u\rangle\otimes|u\rangle-|u\rangle\otimes|u\rangle=0.
$$

Zero is not a normalized physical state. Two bosons can occupy $|u\rangle$;
their normalized product is $|u\rangle\otimes|u\rangle$. Exclusion is a
restriction on states, not a repulsive potential added to H. Similarly,
boson bunching does not require attraction.

For electrons, a spatial orbital plus spin identifies a complete
one-particle state. Opposite spins can occupy the same spatial orbital.
For example, the allowed two-electron state

$$
\Psi(x_1,s_1;x_2,s_2)=\phi(x_1)\phi(x_2)
\frac{\uparrow(s_1)\downarrow(s_2)-\downarrow(s_1)\uparrow(s_2)}{\sqrt2}
$$

has symmetric space and antisymmetric spin. Its total exchange sign is minus.
For a factorized space-and-spin state, multiply their exchange signs to get
the total sign. A spin triplet has sign plus, so two electrons in a triplet
need antisymmetric space. Our fixed-spin lattice model keeps this latter
spatial restriction; it does not include the opposite-spin example above.

| Statistics | Basis pairs | Dimension |
| --- | --- | --- |
| Distinguishable | All ordered $(i,j)$ | $L^2$ |
| Boson | $i\le j$ | $L(L+1)/2$ |
| Fermion | $i<j$ | $L(L-1)/2$ |

These are complete two-particle sectors. No occupation cutoff is imposed.

### 3.5 Why the Hamiltonian does not select the statistics

For two identical particles in the same external potential,

$$
P_{12}(h\otimes I)P_{12}=I\otimes h,\qquad
P_{12}(I\otimes h)P_{12}=h\otimes I.
$$

Adding these equations gives $P_{12}H_{\mathrm{lab}}P_{12}=H_{\mathrm{lab}}$,
or $[H_{\mathrm{lab}},P_{12}]=0$. This remains true for an interaction
$W(x_1,x_2)=W(x_2,x_1)$; noninteraction is not required for exchange symmetry.

For time-independent H, $U(t)=e^{-iHt/\hbar}$ also commutes with $P_{12}$.
If $P_{12}\Psi(0)=\eta\Psi(0)$, then

$$
P_{12}\Psi(t)=P_{12}U(t)\Psi(0)
=U(t)P_{12}\Psi(0)=\eta\Psi(t).
$$

The Hamiltonian **preserves** the exchange sign. It does not choose it.
The labeled space contains both signs. For distinguishable particles, both
and their superpositions are allowed. For a fixed identical-particle species,
we restrict the total state to the required sector before solving.

For example, the lowest state of the full noninteracting labeled Hamiltonian
puts both particles in its lowest one-particle orbital. It is allowed for
bosons, but excluded for our fixed-spin fermions. Diagonalizing the full space
and taking its lowest eigenvalue would therefore give the wrong fermion ground
energy. This is why the code chooses a basis before selecting energies.

Exchange symmetry also does not require an even external potential. A site
potential can favor the left end and break spatial reflection while treating
both particles identically. H then commutes with $P_{12}$ but need not commute
with $\Pi$.

### 3.6 Projectors and the basis used in code

The operators

$$
\mathcal S=\frac{I+P_{12}}2,\qquad
\mathcal A=\frac{I-P_{12}}2
$$

extract symmetric and antisymmetric components. Using $P_{12}^2=I$ gives
$\mathcal S^2=\mathcal S$, $\mathcal A^2=\mathcal A$,
$\mathcal S\mathcal A=0$, and $\mathcal S+\mathcal A=I$.
Projection is not normalization: for $i\ne j$, $\mathcal S|i,j\rangle$
has norm $1/\sqrt2$, so normalize it to obtain $|i,j\rangle_+$.
Also, $\mathcal A|i,i\rangle=0$ cannot be normalized.

PyTorch uses normalized columns Q spanning one sector, so
$QQ^\dagger=\mathcal S$ for bosons and $QQ^\dagger=\mathcal A$ for fermions.
NumPy builds the allowed occupation basis directly. Its hopping factors and
signs already enforce the statistics. Thus exchange symmetry need not appear
as a separate projection step in every implementation.

## 4. Projection and occupation numbers

Let $Q$ contain the chosen normalized basis states as columns in the labeled
space. Then $Q^\dagger Q=I$. A sector coefficient vector $c$ represents
$\Psi=Qc$. The Schrödinger equation becomes

$$
i\hbar\dot c=H_s c,\qquad H_s=Q^\dagger H_{\mathrm{lab}}Q.
$$

This restriction is exact because the sector is invariant. $QQ^\dagger$ is
the sector projector in labeled space. It is not the identity there unless
the full distinguishable basis is used.

The pair $(i,j)$ also defines site counts $n_k=\delta_{ki}+\delta_{kj}$.
Their sum is two. Bosons allow $n_k=0,1,2$ in this sector. Fermions allow only
$0,1$. For identical particles, sorted pairs uniquely identify occupations.
For labeled particles, $(i,j)$ and $(j,i)$ have equal counts but remain distinct.

An independent construction uses

$$
H=\sum_{a,b}h_{ab}\,a_a^\dagger a_b.
$$

Boson operators obey $[a_i,a_j^\dagger]=\delta_{ij}$ and commute with
each other. For one mode, the normalized state is
$|n\rangle=(a^\dagger)^n|0\rangle/\sqrt{n!}$. Commuting $a$ through its
n creation operators gives

$$
a(a^\dagger)^n|0\rangle=n(a^\dagger)^{n-1}|0\rangle,
\qquad a|n\rangle=\sqrt n\,|n-1\rangle,
\qquad a^\dagger|n\rangle=\sqrt{n+1}\,|n+1\rangle.
$$

For bosons, removal from site $b$ gives $\sqrt{n_b}$. Addition at $a$ then
gives $\sqrt{n_a-\delta_{ab}+1}$. Thus the coefficient for this move is

$$
h_{ab}\sqrt{n_b(n_a-\delta_{ab}+1)}.
$$

When $a=b$, this is $h_{aa}n_a$. For a hop from a singly occupied site to
another singly occupied site, it is $-\sqrt2 J$. The factor comes from
normalization and exchange, not an attractive force.

For fermions, fix the convention

$$
|n_0,\ldots,n_{L-1}\rangle=
(a_0^\dagger)^{n_0}\cdots(a_{L-1}^\dagger)^{n_{L-1}}|0\rangle.
$$

Here $\{a_i,a_j^\dagger\}=\delta_{ij}$ and
$\{a_i^\dagger,a_j^\dagger\}=0$. Braces denote an anticommutator:
$\{A,B\}=AB+BA$. In particular, $(a_i^\dagger)^2=0$.

Anticommutation gives a minus sign each time an operator crosses an occupied
earlier orbital. First remove at $b$, producing $n'=n-e_b$. Then add at $a$.
The combined factor is

$$
(-1)^{\sum_{k<b}n_k+\sum_{k<a}n'_k}.
$$

Reject removal from an empty orbital and addition to an occupied orbital.
Use the updated counts for the second sign. Adjacent hops on this open chain
have positive operator factors, but longer hops can have negative factors.
The tests include a long hop across an occupied site to expose sign errors.

## 5. Exact noninteracting spectra

Symmetrizing or antisymmetrizing products of one-particle eigenstates leaves
their energy sum unchanged. Therefore the exact finite-chain spectra are

$$
E_{qr}=\epsilon_q+\epsilon_r,
\quad
\begin{cases}
q,r\text{ ordered},&\text{distinguishable},\\
q\le r,&\text{bosons},\\
q<r,&\text{fermions}.
\end{cases}
$$

For $J>0$ and zero site energies, the ground energies are $2\epsilon_1$ for
bosons or labeled particles, and $\epsilon_1+\epsilon_2$ for fermions.
Adding a constant $v_0$ to every site shifts every two-particle energy by
$2v_0$. Degenerate eigenvectors can rotate within their subspace. Compare
spectra, residuals, or propagators rather than individual eigenvectors.

## 6. Measurable probabilities

Normalize the sector vector with $\sum_d|c_d|^2=1$. Expanding with $Q$ gives
ordered-coordinate amplitudes $\Psi_{ij}$. Define

$$
p_{ij}=|\Psi_{ij}|^2,\qquad
\langle n_i\rangle=\sum_j(p_{ij}+p_{ji}),\qquad
P_{\mathrm{same}}=\sum_i p_{ii}.
$$

Then $\sum_{ij}p_{ij}=1$ and $\sum_i\langle n_i\rangle=2$.
For identical particles, an unordered different-site event has probability
$p_{ij}+p_{ji}$. A same-site event is counted only once. Equivalently,
$P_{\mathrm{same}}=\sum_i\langle n_i(n_i-1)/2\rangle$ for exactly two particles.

The coordinate labels for identical particles are bookkeeping devices.
Do not apply the previous chapter's distinguishable-subsystem entanglement
interpretation directly to their exchange symmetry.

## 7. Two-site experiment

Set $v_i=0$, $J>0$, and start with one particle on each of two sites. Let
$\tau=Jt/\hbar$, $c=\cos\tau$, and $s=\sin\tau$. Since $h=-J\sigma_x$,

$$
U(t)=\cos\tau\,I+i\sin\tau\,\sigma_x.
$$

For labeled particles initially in $|0,1\rangle$, the amplitude on each
same-site state is $ics$. Thus

$$
P_{\mathrm{same}}^{\mathrm{lab}}=2c^2s^2=\tfrac12\sin^2(2\tau).
$$

For bosons, evolve $(|0,1\rangle+|1,0\rangle)/\sqrt2$:

$$
|\psi_B(t)\rangle=\cos(2\tau)|0,1\rangle_+
+\frac{i\sin(2\tau)}{\sqrt2}(|0,0\rangle+|1,1\rangle).
$$

Therefore $P_{\mathrm{same}}^B=\sin^2(2\tau)$. At $\tau=\pi/4$, both bosons
share a site with probability one, split equally between the two sites.

The two-site fermion sector contains only $|0,1\rangle_-$. Both orbitals are
filled, so hopping cannot change the state and $P_{\mathrm{same}}^F=0$.
Its energy is zero for these site energies.

All three preparations have $\langle n_0\rangle=\langle n_1\rangle=1$.
Mean site counts alone miss their different pair correlations. These results
concern this preparation and Hamiltonian; bosons do not always share a site.

## 8. Size and accuracy

At fixed $J$ and lattice spacing, increasing $L$ lengthens the physical chain.
It does not refine a continuous-space grid. For two particles and $v_i=0$,
both ground energies approach the band edge $-4J$. Expanding the cosines gives

$$
E_B+4J\sim\frac{2J\pi^2}{(L+1)^2},\qquad
E_F+4J\sim\frac{5J\pi^2}{(L+1)^2}.
$$

The report checks this finite-size scaling against exact finite-chain formulas.
At each fixed size the full basis is used. Matrix exponentials evaluate each
requested time directly, so closer output times do not reduce a time-step error.
Remaining numerical errors come from floating-point arithmetic and linear algebra.

The next chapter adds interactions and internal spin where needed for Hubbard models.
