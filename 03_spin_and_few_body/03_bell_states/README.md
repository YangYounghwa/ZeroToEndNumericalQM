# Phase 3, Chapter 3: Bell states

The previous chapter produced pure two-spin states whose local Bloch vectors
vanish. Here we calculate the state of each spin using a **reduced density
matrix**. This explains how a perfectly known joint state can have mixed local
states, and why classical correlations can produce the same local statistics.

The main learning target is PyTorch: complex outer products, batched tensor
reshaping, contractions, and Hermitian eigenvalues. NumPy checks the calculation
with explicit index loops. These tiny dense matrices need no new SciPy methods.

## Read and implement

1. Read [theory](theory.md), including the partial-trace derivation and the
   distinction between pure-state entanglement and mixed-state correlations.
2. Read the [numerical method](numerical_method.md) for basis and shape rules.
3. Copy [the PyTorch starter](boilerplates/bell_states_torch_starter.py) into your
   own folder under `workbench/`. Implement projectors, mixtures, and validation
   first; then partial traces, purity, entropy, and correlations. Use the
   [coding hints](coding_hints.md) when needed.
4. Compare with the [PyTorch solution](solutions/bell_states_torch_solution.py).
   Use the [NumPy starter](boilerplates/bell_states_numpy_starter.py) and
   [NumPy solution](solutions/bell_states_numpy_solution.py) as the comparison.
5. Complete the [study starter](boilerplates/bell_states_convergence_starter.py),
   then compare with the [report generator](solutions/bell_states_convergence.py)
   and [validation report](VALIDATION.md).

The supplied tests target the supplied solutions. To test your own exercise,
copy the tests and adjust their imports to your workbench modules. Add imports
needed by your implementation; starter files intentionally contain little code.

## Run from the repository root

```powershell
uv run python 03_spin_and_few_body/03_bell_states/solutions/bell_states_torch_solution.py
uv run python 03_spin_and_few_body/03_bell_states/solutions/bell_states_numpy_solution.py
uv run python 03_spin_and_few_body/03_bell_states/solutions/bell_states_convergence.py
uv run python -m pytest -q 03_spin_and_few_body/03_bell_states
```

The study writes tables to `VALIDATION.md`. Use `--output workbench/bell-study.md`
to save a separate copy. CPU is sufficient; CUDA comparisons run only if available.

## What you should be able to explain

- Why `psi[:, None] * psi.conj()[None, :]` is a density matrix for one ket,
  while a mixture adds projectors with probabilities rather than amplitudes.
- How the four axes `(a,b,a',b')` determine which subsystem is traced out.
- Why all four Bell states have local density matrices `I/2`.
- Why the classical `00/11` mixture also has local entropy one bit, despite
  being separable, and which joint measurements distinguish it from `Phi+`.
- Why pure-state entanglement peaks halfway through an exchange swap and
  disappears at a complete swap.
- Why there is no time-step or spatial-grid refinement in this chapter, and
  how tiny eigenvalues affect the reported entropy.

## Completion checks

Check all Bell correlations, product-state reductions, and an unequal-dimension
`2 x 3` example with complex amplitudes. Verify local expectation values before
and after tracing, local-unitary invariance, mixture positivity, batch shapes,
and the analytical binary entropy. Normalization alone cannot check these.

## Next chapter

Two particles on a lattice: spatial site bases, exchange symmetry, occupation
numbers, and particle statistics before a small Hubbard system.

[Previous: Two coupled spins](../02_two_coupled_spins/README.md) ·
[Phase overview](../README.md) · [Implementation plan](PLAN.md)
