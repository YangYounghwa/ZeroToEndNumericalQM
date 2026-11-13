# Transverse-field Ising chain

Begin many-body exact diagonalization with an open spin chain. Learn bit
bases, global spin-flip parity, sparse low-energy solves, and correlations.
Use Pauli operators in H=-J sum ZZ-g sum X.

## Reading order

1. Read [theory](theory.md).
2. Read [numerical method](numerical_method.md).
3. Use [coding hints](coding_hints.md) with the PyTorch starter.
4. Compare with the independent NumPy/SciPy solution.
5. Read the generated [validation report](VALIDATION.md).

## Commands

Run from the repository root.

```powershell
uv run python 04_exact_diagonalization/01_transverse_field_ising/solutions/ising_torch_solution.py
uv run python 04_exact_diagonalization/01_transverse_field_ising/solutions/ising_numpy_solution.py
uv run python 04_exact_diagonalization/01_transverse_field_ising/solutions/ising_convergence.py
uv run pytest 04_exact_diagonalization/01_transverse_field_ising/tests
```

The dense solution uses at most 10 sites. The sparse interface permits up to
14 sites. The experiment uses sparse solves through 12 sites. These are
complete finite spin spaces. No spatial grid or basis cutoff is present.

The next chapter is the Heisenberg spin chain. It introduces conserved total
Z magnetization and direct construction in fixed-magnetization sectors.
