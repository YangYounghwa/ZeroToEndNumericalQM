# Spin-1/2 in a magnetic field: chapter plan

1. Introduce the z-basis spinor, Pauli matrices, angular momentum, and ordinary
   discrete normalization. Derive H=-gamma*S.B with signed gamma and define
   Omega=-gamma*B so the precession direction is unambiguous.
2. Derive the constant-field propagator from the Pauli algebra, the Bloch vector,
   arbitrary-axis measurement probabilities, and Rodrigues precession.
3. Implement native PyTorch batched matrix exponentials, state/operator
   contractions, and a closed-form reference. Use NumPy eigendecomposition as
   the independent comparison. No new SciPy method or dependency is needed.
4. Reuse batched Crank-Nicolson on 2x2 matrices for a time-step experiment, with
   norm, energy, reversibility, and phase/observable errors checked separately.
5. Test Pauli algebra, measurement projectors, signed precession, zero field,
   eigenstates, batch/single agreement, SU(2) 2pi/4pi rotations, and NumPy parity.
6. Provide theory, numerical method, hints, matching starters, tests, and a
   reproducible validation report. Add Phase 3 to pytest discovery and indexes.

Keep the physical problem to one pure spin in a static uniform field. There is
no spatial grid, field time discretization, relaxation, or composite Hilbert
space in this chapter. The next chapter introduces two coupled spins.
Leave this new chapter uncommitted after the requested preceding chapter commit.

## Completion evidence

Implemented the theory, numerical method, hints, matching starters, PyTorch and
NumPy solutions, tests, and generated validation report. Phase 3 is included
in default pytest discovery. Full project checks: 202 tests passed, four CUDA
tests skipped on the CPU build; strict mypy passed for 117 source files; Ruff
lint/format, the CPU smoke test, examples, and local Markdown links passed.

Matrix-exponential, Pauli-formula, and NumPy states agree to about 6e-15 in the
reported cases. The time study approaches fourfold error reduction when dt is
halved, while the long-time study exposes phase drift despite conserved norm.
