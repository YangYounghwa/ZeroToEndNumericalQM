# Two coupled spins: chapter plan

1. Define the ordered product basis (+z,+z), (+z,-z), (-z,+z), (-z,-z).
   Derive product states, local operators, and isotropic exchange using Kronecker
   products. Distinguish a joint four-amplitude state from an experiment batch.
2. Use H=J*(sigma1.sigma2)/4-gamma*hbar*(B1.sigma1+B2.sigma2)/2.
   J has energy units; positive J puts the singlet below the triplet.
3. Derive the zero-field singlet/triplet spectrum and exact exchange oscillation
   from |+z,-z>. Check the noninteracting limit with independent single-spin
   propagators, and unequal longitudinal fields with a two-state block solution.
4. Implement batched native PyTorch evolution and a NumPy eigenbasis comparison.
   Calculate local spin expectations, joint probabilities, and connected
   correlations. Use coefficient-matrix factorization as a bridge to Bell states.
5. Retain a small CN time-error exercise, separate from exact static-field
   evolution. Test conservation laws only for Hamiltonians that possess them.
6. Add theory, numerical method, hints, matching starters, tests, and a generated
   validation report. Update indexes and run all project checks.

No new dependencies or sparse solver are needed for four states. Defer density
matrices, partial traces, and entropy to the Bell-state chapter. The previous
spin chapter remains uncommitted; this request advances the material without
requesting a commit.

## Completion evidence

Implemented the theory, numerical method, coding hints, matching starters,
PyTorch/NumPy solutions, tests, and generated validation report. Full project
checks: 219 tests passed, five CUDA tests skipped on the CPU build; strict mypy
passed for 125 source files; Ruff lint/format and local documentation links
passed. Both examples and the report generator ran successfully.

Analytical exchange states agree at about 1e-15 in the example; the general-field
NumPy state comparison stays below 6e-15 in the report. CN errors approach a
fourfold reduction when dt is halved. The field sweep reproduces the exact
detuning-dependent transfer limits, and commutator checks distinguish the
conservation laws of different field configurations.
