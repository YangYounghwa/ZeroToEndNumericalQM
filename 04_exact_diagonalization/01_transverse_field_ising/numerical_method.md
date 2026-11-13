# Numerical method

## Dense PyTorch calculation

1. Enumerate all integers from zero to 2**L-1.
2. Extract each site's bit and its Z eigenvalue.
3. Sum adjacent Z products for the diagonal interaction.
4. Set field entries at single-bit-flipped row indices.
5. Solve the real symmetric matrix with `torch.linalg.eigh`.
6. Measure residuals, parity, magnetization, and pair correlations.
7. Project with Q.T @ H @ Q to check the two parity sectors.

Use float64 matrices. Observables accept real or complex normalized states
with shape `(...,2**L)`. They return one value per state. Device selection is
explicit. No NumPy conversion occurs in the PyTorch calculation.

The dense builder accepts 1 through 10 sites. Storage costs 8*4**L bytes for
one real Hamiltonian. Eigenvectors and solver workspace require more memory.
Dense diagonalization costs O(8**L). The explicit parity projection is useful
for learning, but it still forms the full matrix. It does not remove that
memory cost. A later solver can build each block directly.

## Independent sparse reference

The reference constructs local operators from right-to-left Kronecker
products. It sums X terms and neighboring Z products in SciPy CSR format.
This avoids copying the bit-flip construction. Each row has at most L+1
nonzero entries. Storage is O(L*2**L), including index arrays.

`eigsh` requests the smallest algebraic eigenvalues with `which="SA"`.
The default tolerance is 1e-12. A seeded random initial vector has components
in both parity sectors. An all-ones start would be even and could miss odd
states. The function returns sorted eigenpairs and absolute residual norms.
It propagates convergence failures to the caller. Krylov convergence depends
on spectral gaps and the requested eigenpairs; no fixed iteration count is
promised. Exact degeneracies need separate care when requesting multiplicities.

Sparse calculations are limited to 14 sites in this teaching interface.
Dense NumPy comparisons have the same 10-site limit as PyTorch.

## Validation and finite-size study

Check J=0, g=0, L=1, and the exact two-site spectrum first. Compare full
matrices and complex-state observables across independent implementations.
Check that the parity blocks reconstruct the full spectrum.

At g=J=1, compare E0 and the gap with the exact open-chain formulas. Report
residuals separately from E0/L and L*gap. Their approach to the bulk limits
is a physical finite-size effect. It does not indicate eigensolver failure.

At fixed L, vary the sparse tolerance and report measured residuals and
energy errors. A tighter tolerance need not strictly improve an error already
at roundoff. Report actual values instead of assuming monotonic improvement.
At fixed L, sweep g to compare magnetization and correlations. Do not fit a
critical exponent from a few small chains.
