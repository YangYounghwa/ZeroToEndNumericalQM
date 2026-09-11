# Coding hints

## 1. Density matrices

For a single ket, an outer product has two distinct axes. For a batch, use
`unsqueeze(-1)` and `conj().unsqueeze(-2)` so broadcasting creates those axes
without combining separate experiments. `.mH` conjugates and transposes the
last two matrix axes; `.T` is unsuitable for general batched matrices.

For an ensemble, the outer products have shape `(..., K, D, D)`. Expand weights
to `(..., K, 1, 1)` and sum over axis `-3`. Do not average state vectors first.
Test a mixture of `00` and `11`: the result should be diagonal with two entries
of one half, unlike the coherent `Phi+` projector.

## 2. Partial trace

Write `(a,b,a_prime,b_prime)` on paper before writing a reshape. Tracing B
means setting `b_prime=b`, summing over b, and retaining a and a_prime. It does
not mean summing all b and b_prime independently. A dimension of size two does
not identify a subsystem by itself: batch axes may also have size two.

Start with `|0> tensor |1>`. Keeping A should return the `|0>` projector;
keeping B should return the `|1>` projector. Then test a complex product with
dimensions `(2,3)`; a Bell state alone cannot reveal which subsystem was kept.

For a pure state's coefficient matrix C, `C @ C.mH` is an independent A
reference. For B, use `C.transpose(-2,-1) @ C.conj()`. Omitting the latter
conjugation convention gives the wrong off-diagonal phases.

## 3. Purity and entropy

`torch.linalg.eigvalsh` computes real eigenvalues of Hermitian matrices and
supports batches. Do not take `torch.log2(rho)` entry by entry. Check matrix
validity before using the Hermitian solver.

With nonnegative eigenvalues p, make `safe = where(p > 0, p, 1)` and sum
`-p * log2(safe)` on the final axis. Do not discard small positive p. First
verify `diag(1,0)`, `I/2`, and `diag(0.75,0.25)` against scalar formulas.

Use `entanglement_entropy` only with pure joint kets. A one-bit reduced entropy
for the classical mixture is a deliberate counterexample to interpreting all
local entropy as entanglement.

## 4. Correlations and local transformations

Build each `sigma_a tensor sigma_b` with `torch.kron`. Evaluate the trace with
row/column indices in the opposite order: `sum(rho_ij * O_ji)`. Conjugating rho
again is unnecessary. The expected result is real for Hermitian inputs.

Construct local unitaries as exponentials of Hermitian Pauli matrices. Check
both `U @ rho @ U.mH` and the projector of `U @ psi`. A transformation only on B
should leave A's complete reduced matrix unchanged, including off-diagonals.

## 5. Experiments

Batch `theta` values through `schmidt_states`; verify the binary entropy
without a Python loop over matrix operations. In the report, loops are fine
for formatting rows. Compare the phase-independent entropy with phase-dependent
coherence. Revisit exchange evolution and mark product states and maximum
entanglement on the exact time samples.

The starter keeps public function signatures but omits implementation imports
that are unused in the stubs. Add the imports you need. Keep scratch output in
`workbench/`, and write large experiment tables to Markdown.
