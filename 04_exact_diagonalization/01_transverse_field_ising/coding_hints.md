# Coding hints

Read the theory before filling the PyTorch starter.

- Use `torch.arange(2**L)` for basis masks.
- Extract bit i with `(masks >> i) & 1`.
- Flip bit i with `masks ^ (1 << i)`.
- Keep site zero as the rightmost Kronecker factor in the reference.
- Add L-1 interaction bonds for open boundaries.
- Use Pauli eigenvalues plus and minus one, without hidden spin factors.
- Use float64 for diagonalization and complex128 for general state checks.
- Use conjugation in every expectation value.
- Treat each leading state dimension as a batch axis.
- Compare eigenpair residuals before interpreting the gap.
- Form each parity column from a mask and its complement.
- Request smallest algebraic sparse eigenvalues, not smallest magnitudes.

Exercises: derive the L=2 blocks by hand. Check the noninteracting ground
state with uniform amplitudes. Compare a parity-even aligned-spin cat state
with a single aligned product state. Both have unit long-range ZZ correlation,
but only the product state has nonzero Z magnetization. Explain why a
finite symmetric ground state does not directly display broken symmetry.
