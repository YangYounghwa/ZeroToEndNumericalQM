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

The exchange operator swaps particle coordinates:

$$
P|i,j\rangle=|j,i\rangle,\qquad P^2=I,\qquad P^\dagger=P.
$$

Thus its eigenvalues are $+1$ and $-1$. Since both particles see the same
$h$, $[P,H_{\mathrm{lab}}]=0$. Evolution preserves either exchange sector.

Identical bosons require a symmetric total state. Identical fermions require
an antisymmetric total state. Here internal states are fixed and identical,
so the spatial state carries that symmetry. For fermions with different spin
states, antisymmetry applies to space and spin together. Such fermions can
occupy one spatial site in different spin orbitals. They are outside this model.
See [MIT's identical-particle notes](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/fb48dd49e8649557f1715f0415de4f4d_MIT8_06S18ch8.pdf).

For $i<j$, normalized spatial basis states are

$$
|i,j\rangle_+=\frac{|i,j\rangle+|j,i\rangle}{\sqrt2},\qquad
|i,j\rangle_-=\frac{|i,j\rangle-|j,i\rangle}{\sqrt2}.
$$

For two bosons on one site, use $|i,i\rangle_+=|i,i\rangle$. Do not divide
this state by $\sqrt2$. The antisymmetric combination with $i=j$ vanishes.
This gives Pauli exclusion for the single spin orbital at each site.

| Statistics | Basis pairs | Dimension |
| --- | --- | --- |
| Distinguishable | All ordered $(i,j)$ | $L^2$ |
| Boson | $i\le j$ | $L(L+1)/2$ |
| Fermion | $i<j$ | $L(L-1)/2$ |

These are complete two-particle sectors. No occupation cutoff is imposed.

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
