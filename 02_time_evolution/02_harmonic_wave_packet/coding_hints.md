# Coding Hints

## Build on Chapter 1

The propagation algorithm is unchanged. The important additions are:

- calculate `potential = 0.5 * mass * omega**2 * grid**2`;
- add it to the kinetic main diagonal;
- choose `width = sqrt(hbar / (2 * mass * omega))`;
- compare the center with its exact sinusoidal trajectory;
- measure width variation and one-period fidelity.

## Shapes and dtypes

Use `float64` for grids and potentials and `complex128` for states and the
Crank-Nicolson matrices. Store NumPy histories as `(time, position)` and
PyTorch batched histories as `(time, position, batch)`.

## Fidelity

Do not compare `psi(T)` and `psi(0)` component by component without accounting
for global phase. Use

```text
overlap = spacing * vdot(psi_initial, psi_final)
fidelity = abs(overlap)**2
```

Both states must be discretely normalized.

## Tests to write

- The potential is nonnegative and quadratic.
- The Hamiltonian is Hermitian.
- The initial position width is `sqrt(hbar/(2*m*omega))`.
- Norm and energy are conserved.
- The center follows the exact trajectory.
- Width variation decreases under spatial refinement.
- One-period fidelity is near one.
- Forward/reverse propagation recovers the initial state.
- NumPy and PyTorch histories agree.

## Common mistakes

- Reusing an arbitrary free-packet width and calling it a coherent state.
- Forgetting to add the potential only to the main diagonal.
- Expecting exact periodic return from a coarse finite-difference spectrum.
- Using a domain too small for the classical turning points plus packet width.
- Tightening `dt` when the dominant error is spatial discretization.

## Foundations and PyTorch practice

Reuse complex propagation and batching, adding a potential diagonal. Compare the packet center and width with coherent-state dynamics. A global phase does not change the physical state, so use state_l2_error or fidelity. Test time, grid, and domain error independently.

Implement the PyTorch starter first, then compare with the NumPy reference.
