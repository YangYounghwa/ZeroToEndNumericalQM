# Spin-1/2 in a magnetic field

Begin Phase 3 with a two-component quantum state. Learn Pauli matrices,
complex inner products, batched matrix exponentials, spin measurements, and
Bloch-vector precession in PyTorch. NumPy provides an eigenbasis comparison.

## Reading order

1. [Theory and derivations](theory.md)
2. [Numerical methods and sign convention](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. [PyTorch starter](boilerplates/spin_field_torch_starter.py) and
   [solution](solutions/spin_field_torch_solution.py)
5. [NumPy starter](boilerplates/spin_field_numpy_starter.py) and
   [comparison](solutions/spin_field_numpy_solution.py)
6. [Convergence starter](boilerplates/spin_field_convergence_starter.py),
   [experiments](solutions/spin_field_convergence.py), and [results](VALIDATION.md)

## Run from the repository root

```powershell
uv run python 03_spin_and_few_body/01_spin_in_magnetic_field/solutions/spin_field_torch_solution.py
uv run python 03_spin_and_few_body/01_spin_in_magnetic_field/solutions/spin_field_numpy_solution.py
uv run python 03_spin_and_few_body/01_spin_in_magnetic_field/solutions/spin_field_convergence.py
uv run pytest 03_spin_and_few_body/01_spin_in_magnetic_field
```

The experiment command writes `VALIDATION.md`. Use `--output workbench/spin.md`
to save a separate report. No new dependencies or SciPy methods are required.

## What changes from the wave-packet chapters?

The state is `(a,b)` in the `(+z,-z)` basis. Normalize with `|a|^2+|b|^2=1`;
there is no spatial grid or integration weight. Two spin states are the entire
model, not a grid approximation.

Use the explicit convention `H=-gamma*hbar*B.sigma/2`, with signed `gamma`.
The angular-velocity vector is `Omega=-gamma*B`. At `gamma=hbar=1`, a positive
z field rotates an initial +x Bloch vector toward -y.

The main example batches three independent field/state pairs and evaluates
their constant-field matrix exponentials. It checks analytical rotation,
energy, norm, and spin measurement probabilities. More requested output times
do not improve a matrix exponential at a fixed time; there is no integration
time step in that method.

## Accuracy experiments

The matrix exponential, Pauli formula, and NumPy eigenbasis states agree within
about `6e-15` in the tested cases. The largest Bloch discrepancy is about `1e-14`.

Crank-Nicolson is included as a time-error exercise. For the tilted-field test,
reducing `dt` from 0.2 to 0.1 reduces the final state error from about `5.60e-3`
to `1.40e-3`. Norm and energy remain near roundoff in both cases. The report
also shows how precession error accumulates over long runs.

## Completion criteria

- Pauli algebra, spin eigenvalues, and arbitrary-axis measurement probabilities.
- Exact signed precession, transverse spin flips, and zero-field evolution.
- Energy-eigenstate phases and the spinor's `2*pi`/`4*pi` rotation behavior.
- PyTorch/NumPy agreement, batch/single agreement, and reversibility.
- Second-order CN convergence and accumulated phase-error checks.

## Next chapter

[Two coupled spins](../02_two_coupled_spins/README.md): tensor products, local operators in a composite Hilbert
space, exchange interaction, and correlations. A joint two-spin state has four
amplitudes; it is not a batch of two independent two-component states.
