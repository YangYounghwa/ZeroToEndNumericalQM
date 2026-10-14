# Coding hints

## Start with PyTorch

1. Implement `basis_pairs`. Write down the pairs for three sites by hand.
2. Build `single_particle_hamiltonian`. Check that the end sites are not linked.
3. Implement `basis_embedding`. Check every column's norm and exchange sign.
4. Form and project the Kronecker sum.
5. Use `torch.linalg.eigvalsh` to compare energy sums.
6. Prepare a localized pair and evolve it with the matrix exponential.
7. Expand states to `(L,L)` amplitudes before measuring site counts.

For `L=2`, the boson pair order is `(0,0), (0,1), (1,1)`. Its hopping
matrix has `-sqrt(2)*J` between neighboring basis states. The fermion basis
contains only `(0,1)`.

## Shape checks

`Q` stores basis states in columns. State batches use the final axis for sector
coefficients. Use `einsum('pd,...d->...p', Q, states)` to expand them. Reshape
only the final axis into two site axes.

Square magnitudes before summing probabilities. Add row and column sums to
get mean site counts. Sum the diagonal once for same-site probability.
No `dx`, factorial, or extra factor of two belongs in the state norm.

## NumPy comparison

Construct occupation tuples from canonical pairs. Use a dictionary from tuples
to row indices. Each column is the initial configuration:

```text
for each initial occupation:
    for each occupied source:
        remove one particle
        for each allowed target:
            calculate removal and addition factors
            add h[target, source] * factor to the final row
```

For a fermion hop, count occupied earlier sites separately before removal and
before addition. Test a hop over an occupied middle site; an adjacent-hop test
alone will not expose every sign mistake.

## Exercises

- Check the basis dimensions for `L=2,3,4`.
- Recover both particles' energy shift when each site energy increases by a constant.
- Compare boson and fermion spectra on a three-site chain by hand.
- At `J*t/hbar=pi/4`, compare same-site probabilities with mean occupations.
- Batch several initial pairs under the same Hamiltonian.
- Change a site potential and compare both implementations.
- Explain why increasing L is a physical size study and why decreasing output
  time spacing is not an integration accuracy test.

Read the solutions after attempting the starters. The supplied tests import
the solutions; copy the tests and adjust imports to check your workbench code.
