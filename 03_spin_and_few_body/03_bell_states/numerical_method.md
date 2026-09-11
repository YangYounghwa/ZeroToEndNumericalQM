# Numerical method

## Shapes and basis order

All state amplitudes and matrices use `complex128`; real parameters and
observables use `float64`. Leading axes describe independent experiments.
Only the final state axes describe quantum subsystems.

| Quantity | Shape | Meaning |
| --- | --- | --- |
| Pure states | `(..., D)` | One normalized ket per batch entry |
| Joint matrices | `(..., D, D)` | Row and column indices of the joint basis |
| Ensemble kets | `(..., K, D)` | K preparations in each experiment |
| Ensemble probabilities | `(..., K)` | Classical weights, sum over K is one |
| Bipartite matrix after reshape | `(..., dA, dB, dA, dB)` | `(a,b,a',b')` |
| A or B reduction | `(..., dA, dA)` or `(..., dB, dB)` | Remaining subsystem |
| Entropy or purity | `(...)` | One number per density matrix |
| Pauli correlations | `(..., 3, 3)` | x,y,z on A and x,y,z on B |

`bell_states()` returns four **rows**, ordered `Phi+, Phi-, Psi+, Psi-`.
These rows form a batch, not the columns used for the previous chapter's
singlet/triplet basis matrix. The product-basis index remains `a*dB+b`.

## Calculation order

1. Check a ket's norm and form its outer product with its complex conjugate.
   For mixtures, construct each projector and sum with classical probabilities.
2. Validate a density matrix: finite entries, square shape, Hermiticity,
   trace one, and nonnegative Hermitian eigenvalues within tolerance.
3. Reshape the final two axes into four subsystem axes. Compute
   `einsum('...abcb->...ac', tensor)` for A or
   `einsum('...abad->...bd', tensor)` for B. NumPy's reference instead sums
   `rho[..., a*dB+b, a_prime*dB+b]` and the corresponding B entries explicitly.
4. Compute purity from `Tr(rho @ rho)` and entropy from `eigvalsh(rho)`.
   Compute observables with `Tr(rho @ O)`, preserving batch axes.
5. Compare results to analytical states before using random mixtures.

The functions reject unnormalized states and weights rather than silently
changing the physical input. `partial_trace` validates a density matrix, so it
is an educational state routine, not a general partial-trace API for arbitrary
unnormalized operators. Repeated validation is inexpensive for these tiny
matrices; larger systems would need a different performance strategy.

A dense D-dimensional density matrix stores D squared complex numbers,
compared with D for a pure ket. Constructing a projector costs O(D squared);
dense Hermitian diagonalization costs O(D cubed). For D=dA*dB, the trace
contractions sum O(dA squared*dB) or O(dA*dB squared) entries, although the
full joint matrix is still stored. Batch memory scales with the number of
experiments. This is inexpensive for two spins but motivates more economical
representations for many-body states later.

## Zero eigenvalues and roundoff

The default absolute validation tolerance is `1e-12`. Eigenvalues below its
negative are rejected. Small negative eigenvalues within tolerance are clipped
to zero only when computing entropy. No matrix is projected onto a physical
state, symmetrized, or renormalized. `eigvalsh` assumes Hermiticity, which is why
the explicit Hermiticity check comes first.

For entropy, replace zero arguments to the logarithm with one before taking
`log2`; the original zero eigenvalue still multiplies that result. This avoids
`0 * -inf` without adding an epsilon to every probability. Keep all positive
eigenvalues, even those below the validation tolerance. At `p=1e-14`, the term
`-p*log2(p)` is small but meaningful and should appear in the precision study.

An almost pure rotated matrix may produce a tiny entropy error of either sign
because its computed eigenvalues and trace are imperfect. Use absolute error
near zero, not relative error. The tolerance is an input acceptance bound,
not a guarantee that every entropy error is below the same value. The examples
use double precision from construction; casting inaccurate single-precision
input later cannot restore lost digits. Autograd through entropy at zero
eigenvalues is outside this chapter's scope.

## Verification and study design

- Analytical checks: Bell projectors, Bell correlation table, product-state
  reductions, classical mixture, binary entropy, and exchange landmarks.
- Structural checks: different subsystem dimensions and complex off-diagonal
  entries, multiple batch axes, both trace directions, local expectations, and
  local-unitary invariance.
- Independent comparison: NumPy index loops versus PyTorch contractions on
  seeded mixtures. Library agreement supplements the analytical checks.
- Numerical sensitivity: exactly zero and small positive eigenvalues,
  alongside invalid density matrices that must be rejected.

There is no finite grid, truncated basis, or approximate time integrator here.
The file named `bell_states_convergence.py` follows the project layout but
reports physical parameter sweeps and floating-point checks. Refining the
theta or time sampling only draws a denser curve; it does not reduce a solver's
time-step error. Exchange samples use a closed formula checked against the
same finite Hamiltonian's matrix exponential.
