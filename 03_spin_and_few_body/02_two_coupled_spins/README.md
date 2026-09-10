# Two coupled spins

Learn tensor products, local operators, exchange interactions, and joint
correlations with a four-amplitude quantum state. Use PyTorch first, with a
NumPy eigenbasis comparison and analytical references.

## Reading order

1. [Theory and exchange derivation](theory.md)
2. [Numerical methods and basis ordering](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. [PyTorch starter](boilerplates/coupled_spins_torch_starter.py) and
   [solution](solutions/coupled_spins_torch_solution.py)
5. [NumPy starter](boilerplates/coupled_spins_numpy_starter.py) and
   [comparison](solutions/coupled_spins_numpy_solution.py)
6. [Convergence starter](boilerplates/coupled_spins_convergence_starter.py),
   [experiments](solutions/coupled_spins_convergence.py), and [results](VALIDATION.md)

## Run from the repository root

```powershell
uv run python 03_spin_and_few_body/02_two_coupled_spins/solutions/coupled_spins_torch_solution.py
uv run python 03_spin_and_few_body/02_two_coupled_spins/solutions/coupled_spins_numpy_solution.py
uv run python 03_spin_and_few_body/02_two_coupled_spins/solutions/coupled_spins_convergence.py
uv run pytest 03_spin_and_few_body/02_two_coupled_spins
```

The experiment command writes `VALIDATION.md`; use `--output workbench/coupled.md`
for a separate report. No new dependencies or sparse methods are required.

## Main example

Use `H=J*sigma1.sigma2/4`, `J=hbar=1`, no fields, and initial state `|+z,-z>`.
The basis order is `(++,+-,-+,--)`.

At `t=pi/2`, the state cannot be factored into two independent spinors: its
coefficient determinant has magnitude one half. At `t=pi`, the spin directions
have swapped completely. The numerical state agrees with the exact exchange
solution to about `1e-15` in the example.

The report also checks the singlet/triplet spectrum, suppression of transfer by
unequal fields, conditional conservation laws, and general-field NumPy results.
The CN state error falls from about `7.27e-3` to `1.82e-3` when `dt` changes
from 0.2 to 0.1 in its fixed-time study. Conserved norm and energy do not remove
this error in exchange timing.

## Completion criteria

- Correct tensor-product basis and site-specific operator actions.
- Analytical singlet/triplet energies and exact exchange dynamics.
- Factorized evolution in the noninteracting limit.
- Local means, joint probabilities, correlations, and pure-state product test.
- Unequal-field block reference and conservation checks appropriate to the field.
- PyTorch/NumPy agreement, batching, reversibility, and CN time convergence.

## Next chapter

Bell states, density matrices, partial traces, and entanglement entropy. Learn
how a joint pure state can have mixed local states, and why a classical mixture
can share some correlations with an entangled state.
