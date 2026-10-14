# Numerical method

## Basis and shapes

Use at least two sites. Let $D$ be the sector dimension from the theory table.
Pairs follow lexicographic order. For identical particles, require the smaller
site first. Reject a repeated site for fermions.

| Quantity | Shape | Type |
| --- | --- | --- |
| One-particle Hamiltonian | `(L, L)` | `complex128` |
| Embedding Q | `(L*L, D)` | `complex128` |
| Sector Hamiltonian | `(D, D)` | `complex128` |
| Initial states | `(..., D)` | `complex128` |
| Requested times | `(T,)` | `float64` |
| State history | `(T, ..., D)` | `complex128` |
| Expanded pair amplitudes | `(..., L, L)` | `complex128` |
| Mean site counts | `(..., L)` | `float64` |
| Same-site probability | `(...)` | `float64` |

Leading state axes hold independent experiments. They do not label particles.
Construct inputs in double precision. Converting rounded float32 data later
does not recover lost digits. PyTorch uses an explicit device and stays in
PyTorch until comparison or reporting.

## PyTorch construction

1. Sample real site energies. Add `-hopping` on adjacent off-diagonals.
2. Build Q from normalized product, symmetric, or antisymmetric columns.
3. Form `kron(h, I) + kron(I, h)`.
4. Project with `Q.mH @ H_labeled @ Q`.
5. Check `Q.mH @ Q`, Hermiticity, and the exact pair-sum spectrum.

For bosons on the same site, put one unit entry in Q. For different sites,
put two entries of magnitude `1/sqrt(2)`. The second is negative for fermions.
Swapping the final two axes of a reshaped column must give its exchange sign.
For identical sectors, also check `H_labeled @ Q == Q @ H_sector`.

## Independent NumPy construction

Build each matrix column by moving particles out of its occupation state.
For each source and target, apply the removal factor before the addition
factor. For fermions, compute the second parity from the occupation after
removal. For bosons, use the square-root factors derived in theory.

The NumPy code does not call the PyTorch code or construct Q. For labeled
particles, it moves each coordinate separately. It calculates site counts and
coincidences directly from sector probabilities and occupation numbers.

Test complex Hermitian h and long-range hopping as well as the real open chain.
This detects conjugation and fermion-sign errors that nearest-neighbor tests
can miss. The generic builder is broader than the chapter's physical example.

## Evolution

PyTorch evaluates `torch.linalg.matrix_exp(-1j * H * t / hbar)` at every time.
It applies each propagator to all initial states with `einsum`. This is a
matrix exponential; elementwise `torch.exp` would be incorrect.
See the [PyTorch reference](https://docs.pytorch.org/docs/stable/generated/torch.linalg.matrix_exp.html).

NumPy diagonalizes $H=V\operatorname{diag}(E)V^\dagger$ once. It multiplies
energy-basis amplitudes by $\exp(-iEt/\hbar)$ and transforms back. No algorithm
uses successive time steps. Negative times are allowed for reversibility checks.

Reject nonsquare, nonfinite, or non-Hermitian matrices. Require a positive
finite `hbar`, real finite times, and normalized nonempty states of the right
dimension. Norm tolerance is `1e-10`; Hermiticity tolerance is `1e-12`, both
absolute. The routines reject bad states instead of silently renormalizing them.
These tolerances fit the small double-precision examples, not arbitrary scales.

## Cost and scope

The labeled matrix and Q need $O(L^4)$ memory. Dense projection costs up to
$O(L^6)$ work because $D=O(L^2)$. Dense diagonalization costs $O(D^3)$.
PyTorch stores T dense propagators, using $O(TD^2)$ memory. For B initial
states, stored history uses $O(TBD)$ memory. NumPy reuses one eigendecomposition.

These routines are teaching implementations for short chains. They do not
implement a sparse large-lattice solver or a batched family of Hamiltonians.
The batch support is for initial states sharing one Hamiltonian.

## Validation and finite-size study

- Check dimensions, Q orthonormality, sector invariance, and Hermiticity.
- Compare the full spectrum with sums of analytical open-chain energies.
- Check a uniform potential shift of `2*v0`.
- Compare independent Hamiltonians and complex state histories.
- Measure norm, static energy, total particle number, and reversibility.
- Check the two-site formulas for all three statistics, including `hbar != 1`.
- Compare batched initial states with separate evolutions.
- Check fixed-size spectra separately from the approach to the long-chain band edge.

The report's size study changes the physical chain. Its scaled band-edge errors
approach $2\pi^2$ and $5\pi^2$ for $J=1$. This is finite-size scaling, not an
eigensolver error. Roundoff is measured separately against the exact finite-size
answer. Sampling more output times only draws a denser curve.

Common mistakes are double-counting identical pairs, giving diagonal boson
states a `1/sqrt(2)` factor, omitting boson enhancement, using the wrong fermion
operator order, and normalizing mean site counts to one instead of two.
