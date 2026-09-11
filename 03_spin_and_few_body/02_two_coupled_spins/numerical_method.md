# Numerical methods for two coupled spins

## 1. Fix the basis before writing code

Use `(++,+-,-+,--)`, so flattening the coefficient matrix in C order gives
`index = 2*first_spin_index + second_spin_index`. A four-component state can be
reshaped to `(2,2)` without changing this convention.

Build local operators with `torch.kron(sigma, identity)` and
`torch.kron(identity, sigma)`. The Kronecker product constructs matrix blocks;
it does not mean elementwise multiplication. See the
[PyTorch kron documentation](https://docs.pytorch.org/docs/stable/generated/torch.kron.html).

For batched spinor products, form a separate outer product for each pair:
`first[..., :, None] * second[..., None, :]`, then flatten only the last two
axes. Applying `kron` directly to two `(batch,2)` arrays combines their batch
axes too, which is a different calculation.

## 2. Tensor shapes

| Quantity | Shape |
| --- | --- |
| Exchange energies | `(B,)` |
| Local fields, site then x/y/z | `(B,2,3)` |
| First-site or second-site Pauli operators | `(3,4,4)` |
| Hamiltonians | `(B,4,4)` |
| Initial joint states | `(B,4)` |
| State history | `(T,B,4)` |
| Local Bloch vectors | `(T,B,2,3)` |
| Joint or connected correlations | `(T,B,3,3)` |
| Joint z probabilities | `(T,B,4)` |

`B` in shapes denotes the experiment batch size. Each batch member is already
a complete two-spin system. Use complex128 for states and operators, float64
for fields, exchange values, and times, on one explicit device.

## 3. Build and evolve the Hamiltonian

The local-operator arrays allow the exchange term to be computed as
`sum(first[a] @ second[a] for a in range(3))/4`, then scaled by `J`.
Build the two field terms with the signed factor `-gamma*hbar/2`.

The exact static-model evolution is `exp(-1j*H*t/hbar) @ initial`.
PyTorch uses batched `torch.linalg.matrix_exp`; NumPy reconstructs the same
propagator from `eigh`. Neither has a time-integration step error. Output-time
spacing affects how densely the trajectory is sampled, not the accuracy of
an individual requested time.

Do not compare individual numerical eigenvectors inside the degenerate
triplet. Compare the known eigenpair residuals, the triplet subspace, or the
reconstructed propagator. Eigenvector phases also disappear from the latter.

If `J=0`, local terms commute and the propagator factors exactly:

$$
e^{-it(H_1\otimes I+I\otimes H_2)/\hbar}
=e^{-itH_1/\hbar}\otimes e^{-itH_2/\hbar}.
$$

This provides an independent check of site order, fields, and tensor products.
It does not hold for a nonzero exchange interaction in general.

## 4. Correlations and nonfactorization

Use conjugate expectations for every observable. The local and pair operator
arrays can be contracted with arbitrary leading state axes via `einsum`.
Pair correlations have two physical axes, first-site component then second-site
component; they are not a two-by-two matrix of sites.

Calculate probabilities and expectations on normalized inputs. The observables
do not renormalize histories: their normalization behavior helps reveal drift.
Check `sum(abs(state)**2)` separately. Physical spin correlations are
`(hbar/2)**2` times the reported dimensionless Pauli correlations.

`product_determinant` evaluates `abs(c0*c3-c1*c2)` directly. It gives zero for
product pure states and one half for the half-exchange state. It avoids
introducing density matrices before the next chapter. Do not apply it to a
four-by-four density matrix or to an incoherent mixture of amplitudes.

## 5. Time-step comparison

The CN update is the same linear solve as before, now with 4-by-4 matrices:

$$
(I+iHdt/(2\hbar))\psi_{n+1}=(I-iHdt/(2\hbar))\psi_n.
$$

PyTorch reuses batched LU factors. NumPy solves once for each tiny step matrix.
Neither forms an explicit inverse. Normalize once before evolution and save
the initial/final states even when the snapshot interval does not divide steps.

For an energy `E`, the numerical phase per step is
`-2*atan(E*dt/(2*hbar))`, instead of `-E*dt/hbar`. The singlet and triplet
therefore accumulate an incorrect relative phase. Global phase alignment cannot
remove that error, which changes spin-swap timing. CN conserves norm and static
energy and is reversible, but is only second order in time at fixed duration.

Use the closed-form exchange state for the time study. For arbitrary fields,
compare against the exact same four-by-four Hamiltonian's exponential. Keep
field detuning and numerical time-step studies separate.

## 6. Scope and cost

There is no spatial grid or basis cutoff for this four-state spin model. Dense
linear algebra is small and transparent; another sparse solver adds no value
here. Stored propagators scale as `O(T*B)` with 16 complex entries per field and
time. CN working memory scales as `O(B)`, with `O(S*B)` stored snapshots.

The exponential size of a many-spin Hilbert space becomes important later:
`n` spin halves need `2**n` amplitudes. First verify site ordering and operators
in this small system before extending the same construction to longer chains.
