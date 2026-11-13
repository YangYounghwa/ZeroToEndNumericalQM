# Small Hubbard system

Add on-site interactions to a spinful two-site fermion model. Learn Fock
operator signs, exact sector projection, double occupancy, and the connection
to exchange-coupled spins. The sector has one up and one down electron.

## Reading order

1. Read [theory](theory.md).
2. Read [numerical method](numerical_method.md).
3. Use [coding hints](coding_hints.md) with the PyTorch starter.
4. Compare with NumPy.
5. Read the generated [validation report](VALIDATION.md).

## Commands

Run from the repository root.

```powershell
uv run python 03_spin_and_few_body/05_small_hubbard_system/solutions/hubbard_torch_solution.py
uv run python 03_spin_and_few_body/05_small_hubbard_system/solutions/hubbard_numpy_solution.py
uv run python 03_spin_and_few_body/05_small_hubbard_system/solutions/hubbard_convergence.py
uv run pytest 03_spin_and_few_body/05_small_hubbard_system/tests
```

At hopping 1, interaction 4, and zero bias, the ground energy is about
-0.82842712. Total double occupancy is about 0.14644661. Each site has mean
count one. These checks test distinct properties.

The complete dimer basis needs no cutoff study. The experiment measures the
large-U exchange approximation and checks exact dynamics separately.
The next chapter is the Phase 4 transverse-field Ising chain. It introduces
bit-string sectors and many-body exact diagonalization on larger systems.
