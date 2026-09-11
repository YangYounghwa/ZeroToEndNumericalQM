# Theory: density matrices and Bell-state entanglement

## 1. From a state vector to a density matrix

Keep the previous chapter's product basis `(++, +-, -+, --)`. The alternative
labels `(00, 01, 10, 11)` mean `0 = +z` and `1 = -z`; they do not denote particle
occupations. The first index belongs to spin A, and the second to spin B.

For a normalized ket, define

$$
\rho=|\psi\rangle\langle\psi|,\qquad
\rho_{ij}=\psi_i\psi_j^*.
$$

It is Hermitian, has trace one, and is positive semidefinite because
$v^\dagger\rho v=|\langle\psi|v\rangle|^2\ge0$. It also satisfies
$\rho^2=\rho$. Multiplying the ket by a global phase leaves its density matrix
unchanged. No spatial integration weight is used for this orthonormal basis.

The expectation value of an operator follows from expanding the trace:

$$
\operatorname{Tr}(\rho O)=\sum_{ij}\psi_i\psi_j^*O_{ji}
=\langle\psi|O|\psi\rangle.
$$

In particular, diagonal entries are probabilities for measurements in the
chosen basis. Off-diagonal entries encode coherence in that basis.

## 2. Statistical mixtures

If preparation chooses normalized ket $|\psi_k\rangle$ with classical
probability $p_k$, the density matrix is

$$
\rho=\sum_k p_k|\psi_k\rangle\langle\psi_k|,
\qquad p_k\ge0,\quad\sum_kp_k=1.
$$

Expectations are then the probability-weighted average of each preparation's
expectation. Adding amplitudes instead would describe a coherent superposition
and introduce cross terms. The mixture's constituent states need not be
orthogonal, and its decomposition into preparations is generally not unique.

A density matrix's eigenvalues are nonnegative and sum to one. Its purity is

$$
P=\operatorname{Tr}(\rho^2)=\sum_k\lambda_k^2,
\qquad 1/D\le P\le1.
$$

Purity equals one exactly when the state is pure; the maximally mixed state
$I_D/D$ has purity $1/D$. A mixed density matrix generally cannot be replaced
by a single state vector of that same system.

## 3. Deriving the partial trace

Write the joint entries as $\rho_{ab,a'b'}$, with flattened index
$i=a d_B+b$. We want a local matrix that reproduces every A-only measurement:

$$
\operatorname{Tr}[\rho(O_A\otimes I_B)]
=\sum_{aa'b}\rho_{ab,a'b}(O_A)_{a'a}
=\operatorname{Tr}(\rho_A O_A).
$$

This identifies

$$
(\rho_A)_{aa'}=\sum_b\rho_{ab,a'b},\qquad
(\rho_B)_{bb'}=\sum_a\rho_{ab,ab'}.
$$

Only one pair of subsystem indices is contracted. Taking a full trace instead
would return a scalar. Selecting a diagonal block instead would correspond to
one measurement outcome before normalization; it would omit the sum over the
ignored subsystem. A partial trace describes the local system without
conditioning on a remote measurement result.

For a pure state, reshape its coefficients into $C_{ab}$. Substituting the
outer product gives

$$
\rho_A=CC^\dagger,\qquad
(\rho_B)_{bb'}=\sum_a C_{ab}C^*_{ab'}.
$$

Thus in this ket-coefficient convention $\rho_B=C^T C^*$, not $C^\dagger C$.
The latter has the same eigenvalues but different complex off-diagonal entries.
Both reductions share the squared singular values of C as their nonzero
eigenvalues. For unequal subsystem dimensions, the larger reduction may have
additional zeros. Tests use complex coefficients and dimensions `(2,3)` so
that conjugation and axis mistakes cannot hide in symmetric real examples.

## 4. Bell states and classical correlations

The four normalized Bell states are

$$
|\Phi^\pm\rangle=(|00\rangle\pm|11\rangle)/\sqrt2,\qquad
|\Psi^\pm\rangle=(|01\rangle\pm|10\rangle)/\sqrt2.
$$

Their joint density matrices are pure, while both local matrices are $I_2/2$.
For example, $\Phi^+$ has entries $\rho_{00,00}=\rho_{11,11}=1/2$ and
coherences $\rho_{00,11}=\rho_{11,00}=1/2$. Tracing either subsystem removes
these cross terms.

Compare it with the explicitly separable mixture

$$
\rho_{\mathrm{cl}}=\tfrac12|00\rangle\langle00|
+\tfrac12|11\rangle\langle11|.
$$

The mixture has the same diagonal and the same local states, but no cross
terms. It represents a shared classical random bit. A state is separable when
it can be written as a probability mixture of product density matrices;
otherwise it is entangled. Being different from $\rho_A\otimes\rho_B$ alone
only establishes correlations. For a *pure* joint state, failure to factor as
a product ket does establish entanglement.

These definitions and the Bell-state reduced matrices are also developed in
[IBM Quantum's discussion of composite and reduced states](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/multiple-systems).

The previous chapter's Pauli correlations distinguish these examples:

| Joint state | xx | yy | zz | Joint purity | Local purity |
| --- | --- | --- | --- | --- | --- |
| Phi+ | 1 | -1 | 1 | 1 | 1/2 |
| Phi- | -1 | 1 | 1 | 1 | 1/2 |
| Psi+ | 1 | 1 | -1 | 1 | 1/2 |
| Psi- | -1 | -1 | -1 | 1 | 1/2 |
| Classical 00/11 mixture | 0 | 0 | 1 | 1/2 | 1/2 |

Here `xx` means $\operatorname{Tr}(\rho\,\sigma_x\otimes\sigma_x)$, a
dimensionless Pauli expectation. Multiply by $\hbar^2/4$ for a product of spin
angular momenta. All off-diagonal axis correlations vanish for these examples.
Their local Pauli means are zero, so connected and ordinary correlations agree.
An xx measurement distinguishes this particular Bell state from this particular
mixture; it is not a universal test of mixed-state entanglement.

## 5. Entropy and the pure-state restriction

Von Neumann entropy in bits is

$$
S(\rho)=-\operatorname{Tr}(\rho\log_2\rho)
=-\sum_k\lambda_k\log_2\lambda_k,
\qquad 0\log_2 0:=0.
$$

The eigenvalue definition avoids taking logarithms of individual matrix
entries. Entropy ranges from zero for a pure state to $\log_2D$ for $I_D/D$.
For a pure bipartite joint state, define entanglement entropy as

$$
E(|\psi\rangle)=S(\rho_A)=S(\rho_B).
$$

The equality follows from their matching nonzero eigenvalues. It is zero for
product kets and one bit for Bell states. The classical mixture also has
$S(\rho_A)=S(\rho_B)=1$, but it is separable. Its joint entropy is one bit;
a Bell state's joint entropy is zero. Therefore local entropy by itself is
not an entanglement measure for an arbitrary mixed joint state. The code's
`entanglement_entropy` accepts pure state vectors; `entropy` accepts general
density matrices and does not claim to diagnose entanglement.

The family

$$
|\psi(\theta,\phi)\rangle=\cos\theta|00\rangle
+e^{i\phi}\sin\theta|11\rangle
$$

has local eigenvalues $1-p,p$, where $p=\sin^2\theta$. Its entropy is the
binary entropy $h_2(p)=-p\log_2p-(1-p)\log_2(1-p)$. The phase changes the
joint coherence but not these eigenvalues. Entanglement is maximal at
$\theta=\pi/4$ and zero at $0,\pi/2$.

Under local unitaries $U_A\otimes U_B$, the reduced state becomes
$\rho'_A=U_A\rho_AU_A^\dagger$. Its eigenvalues and entropy do not change.
In particular, a unitary acting only on B leaves A's reduced matrix unchanged.

## 6. Connection to the exchange chapter

For zero fields and $H=(J/4)\boldsymbol\sigma_A\cdot\boldsymbol\sigma_B$,
the evolution from $|01\rangle$, up to a global phase, is

$$
|\psi(t)\rangle=\cos\alpha|01\rangle-i\sin\alpha|10\rangle,
\qquad\alpha=Jt/(2\hbar).
$$

The reduced eigenvalues are $\cos^2\alpha,\sin^2\alpha$, giving
$E(t)=h_2(\sin^2\alpha)$. With $J=\hbar=1$, entropy reaches one bit at
$t=\pi/2$, falls to zero at the complete swap $t=\pi$, and repeats.
The halfway state has a relative phase $-i$; it is locally equivalent to a
Bell state, but its correlation matrix need not match the real Bell table.
