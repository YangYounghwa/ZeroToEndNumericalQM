# Transverse-field Ising chain

## Spins, units, and boundaries

There are L fixed spin-one-half sites on an open chain. Each site has a
Pauli Z eigenbasis: bit zero has eigenvalue +1; bit one has eigenvalue -1.
The Hilbert space has dimension D=2^L. These are site spin factors, not
occupation states for moving fermions. No exchange sign accompanies a bit flip.

Use Pauli matrices X and Z, not spin operators S=X/2 or Z/2. Set hbar=1.
Both J and g have energy units. The Hamiltonian is

$$
H=-J\sum_{i=0}^{L-2}Z_iZ_{i+1}-g\sum_{i=0}^{L-1}X_i.
$$

There are L-1 bonds. Do not add a bond between the last and first sites.
Positive J favors aligned Z values. Positive g favors X=+1.
The code permits either sign, but the parameter study uses J>0 and g>=0.
A single site has no interaction bond.

Z is diagonal with entries (1,-1). X interchanges the two basis states.
For one site, XZ=-ZX. Operators on different sites commute. The field and
interaction terms therefore compete: their preferred states do not generally
coincide. Total Z magnetization is not conserved when g is nonzero.

## Bit basis and matrix entries

Use the integer

$$
b=\sum_{i=0}^{L-1}2^i b_i,\qquad z_i(b)=1-2b_i.
$$

Site zero is the lowest bit and the rightmost tensor factor. Write a product
state in the order |b_(L-1),...,b_0>. This convention fixes local operators,
correlations, and comparisons between libraries.

Applying H to a basis vector gives

$$
H|b\rangle=-J\left[\sum_{i=0}^{L-2}z_i(b)z_{i+1}(b)\right]|b\rangle
-g\sum_{i=0}^{L-1}|b\mathbin{\mathrm{xor}}2^i\rangle.
$$

The first term is diagonal. Each field term flips one bit. Matrix entry
[destination, source] receives -g. Each flip is reversible, so H is real
symmetric and hence Hermitian. This directly gives the PyTorch algorithm.
The independent reference embeds X and Z with Kronecker products.

## Global spin-flip symmetry

Define P as the product of X over all sites. It complements all bits:

$$
P|b\rangle=|\bar b\rangle,\qquad \bar b=(2^L-1)\mathbin{\mathrm{xor}}b.
$$

P is Hermitian and P^2=I. Conjugating a single Z by P changes its sign;
a product of two Z operators keeps its sign. Each X commutes with P.
Thus [H,P]=0. This parity is global spin-flip parity. It is neither spatial
reflection nor fermion exchange.

Use one representative b<D/2 from each complementary pair. The vectors

$$
|b,p\rangle=\frac{|b\rangle+p|\bar b\rangle}{\sqrt2},\qquad p=\pm1,
$$

have parity p and unit norm. Let Q_p contain them as columns. Then
Q_p^dagger Q_p=I and H_p=Q_p^dagger H Q_p has dimension D/2.
The union of both block spectra must equal the full spectrum. Projection is
exact because the Hamiltonian preserves the sectors. Selecting a sector
without checking its physical relevance can omit the first excitation.

For a parity eigenstate, <Z_i>=0 because P Z_i P=-Z_i. Yet
<Z_i Z_j> can be nonzero. A finite symmetric state can therefore have strong
spin correlations and zero mean Z magnetization. This is not evidence that
correlations are absent. At g=0 the ground space is degenerate; an arbitrary
eigensolver vector need not have definite parity.

## Exact limits

At g=0, each domain wall changes one bond energy from -J to +J. For J>0,

$$
E(b)=-J(L-1)+2J w(b),
$$

where w counts adjacent unequal bits. The two aligned states minimize energy.
Their even and odd superpositions are also ground states. For open chains
with negative J, alternating states minimize energy. The ground energy is
-|J|(L-1). At J=g=0 every state is degenerate.

At J=0, the sites are independent in the X basis. If k sites have X=-1,

$$
E_k=-g(L-2k),\qquad \text{multiplicity}=\binom Lk.
$$

For g>0 the ground state is the uniform positive vector in the Z basis.
It has <X_i>=1 and <Z_i Z_j>=0 for distinct sites.

For L=2, use parity combinations of |00>,|11> and |01>,|10>.
The even block is [[-J,-2g],[-2g,J]]. The odd block is diagonal with entries
-J and J. The full spectrum is therefore

$$
\{-\sqrt{J^2+4g^2},\ -J,\ J,\ +\sqrt{J^2+4g^2}\},
$$

sorted when signs or degeneracies require it. This checks both coefficients
and the fact that an open two-site chain contains only one bond.

## Critical-chain reference

For a uniform open chain at g=J>0, an exact free-fermion solution gives

$$
\epsilon_m=4J\sin\frac{(2m-1)\pi}{4L+2},\quad m=1,\ldots,L,
\qquad H=\sum_m\epsilon_m(f_m^\dagger f_m-1/2).
$$

This is an independent reference, not the numerical method implemented here.
One route rotates the spin axes and applies the Jordan-Wigner transformation.
Its open-chain quadratic coupling matrix has diagonal g and neighboring
entries -J. The positive excitation energies are twice its singular values.
At g=J, the boundary-value equation for its squared singular values yields
the angles shown above. There is no periodic-boundary parity constraint on
these open-chain excitation occupations.

Summing the occupied vacuum contributions gives

$$
E_0=-\frac12\sum_m\epsilon_m
=J\left[1-\csc\frac{\pi}{4L+2}\right],\qquad
E_1-E_0=4J\sin\frac{\pi}{4L+2}.
$$

The energy formula follows from the finite sum of sines at odd multiples of
pi/(4L+2). It gives -J for L=1 and -sqrt(5)J for L=2.
As L grows, E0/L approaches -4J/pi and L(E1-E0) approaches pi*J.
The finite-chain gap is nonzero. A small-chain calculation does not by itself
establish a thermodynamic phase transition.

For the exact solution and finite-boundary analysis, see
[Pfeuty (1970)](https://doi.org/10.1016/0003-4916(70)90270-8) and
[Finite-size scaling of the quantum Ising chain (1985)](https://doi.org/10.1088/0305-4470/18/1/006).
The [Matrix Product Toolkit benchmark](https://mptoolkit.qusim.net/HOWTO/ExactEnergies)
also lists the critical open-chain ground energy with Pauli normalization.
That benchmark uses the opposite sign for the field. Conjugating by the
product of all Z operators reverses the field sign and preserves every ZZ
term, so the energy spectra agree.

## Observables and errors

For a normalized state psi, define

$$
m_x=\frac1L\sum_i\langle X_i\rangle,\qquad
m_z=\frac1L\sum_i\langle Z_i\rangle,\qquad
C_{ij}=\langle Z_iZ_j\rangle.
$$

These are Pauli observables. Spin observables have additional factors of 1/2
or 1/4. C_ii=1. The connected correlation is
C_ij-<Z_i><Z_j>; the code reports C_ij itself.

Diagonal observables use probabilities |psi_b|^2. X expectations use
conjugate(psi_b)*psi_(b xor 2^i). Parity uses the complemented index.
These formulas support complex states and batches with the basis axis last.

Check each normalized eigenpair with r=||Hv-Ev||_2. A small residual bounds
the distance to some exact eigenvalue. It does not prove the eigenvalue is
the ground state. Use ordered low-energy requests and independent references.
Compare degenerate eigenspaces or spectra, not individual eigenvectors.
Changing L changes the physical system. It is not mesh refinement.
