# Coding hints

Read the theory before filling the starters. Implement PyTorch first.

- Use a bit test to check whether an orbital is occupied.
- Count occupied lower bits with `int.bit_count()` to obtain the fermion sign.
- Store matrix elements as `[destination, source]`.
- Use `.mH` for the Hermitian transpose.
- Form creation-annihilation products before selecting the sector.
- Keep mask order `(3,9,6,12)` consistent in every function.
- Use `torch.einsum("tij,...j->t...i", propagators, states)` for batches.
- Sum squared magnitudes on the last axis to check normalization.
- Measure double occupancy with weights `(1,0,0,1)`.
- Measure site counts with weights `(2,1,1,0)` and `(0,1,1,2)`.
- Derive the four NumPy matrix rows independently from the hopping rules.
- Use a normalized singlet to test the analytical doublon oscillation.

Exercises: test the anticommutator for different orbitals. Recover the atomic
limit at nonzero bias. Compare a batch with separate calls. Explain why the
large-U gap error decreases while the exact eigensolver residual stays small.
