# Phase 3, Chapter 4: Two particles on a lattice

Compare distinguishable particles, identical bosons, and spin-polarized
fermions on a finite open chain. Start without interactions. Learn site bases,
exchange symmetry, occupation numbers, and PyTorch basis projection.

PyTorch projects a labeled two-particle Hamiltonian into each symmetry sector.
NumPy independently builds occupation-number hopping matrix elements. The
two-site example shows why equal mean site counts can hide different pair
correlations.

## Read and implement

1. Read [theory](theory.md) for the model and derivations.
2. Read [numerical methods](numerical_method.md) for shapes, algorithms, and costs.
3. Copy the [PyTorch starter](boilerplates/two_particles_torch_starter.py) into
   `workbench/`. Use the [coding hints](coding_hints.md) as needed.
4. Compare with the [PyTorch solution](solutions/two_particles_torch_solution.py).
5. Complete the [NumPy starter](boilerplates/two_particles_numpy_starter.py).
   Compare with the [independent solution](solutions/two_particles_numpy_solution.py).
6. Complete the [study starter](boilerplates/two_particles_convergence_starter.py).
   Compare with the [report generator](solutions/two_particles_convergence.py)
   and [validation report](VALIDATION.md).

The tests target the supplied solutions. Copy them and change imports to test
your own implementation. Use Python files and `uv`; no notebook is needed.

## Run from the repository root

```powershell
uv run python 03_spin_and_few_body/04_two_particles_on_lattice/solutions/two_particles_torch_solution.py
uv run python 03_spin_and_few_body/04_two_particles_on_lattice/solutions/two_particles_numpy_solution.py
uv run python 03_spin_and_few_body/04_two_particles_on_lattice/solutions/two_particles_convergence.py
uv run python -m pytest -q 03_spin_and_few_body/04_two_particles_on_lattice
```

The study writes `VALIDATION.md`. Set `--output workbench/lattice-study.md` to
save another copy. CPU is sufficient. CUDA checks run only when available.

## Completion checks

- Explain dimensions $L^2$, $L(L+1)/2$, and $L(L-1)/2$.
- Derive the boson square-root factors and fermion signs.
- Match all noninteracting energies to allowed one-particle energy sums.
- Preserve norm, energy, exchange symmetry, and total particle number.
- Recover analytical two-site probabilities for all three statistics.
- Compare batch and single-state results and independent implementations.
- Distinguish finite-size physics from floating-point and solver error.

The finite-site basis is complete. Increasing the chain length changes the
physical system. Exact matrix evolution has no time-integration step to refine.

## Next chapter

A small Hubbard system adds interactions. Introduce spin orbitals before
allowing opposite-spin fermions to share a spatial site.

[Previous: Bell states](../03_bell_states/README.md) ·
[Phase overview](../README.md) · [Implementation plan](PLAN.md)
