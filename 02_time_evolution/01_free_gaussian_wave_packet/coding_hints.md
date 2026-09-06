# Coding Hints

## Recommended order

1. Build the interior grid and sparse free Hamiltonian.
2. Construct and discretely normalize a `complex128` Gaussian.
3. Form the two Crank-Nicolson matrices.
4. Factorize the left matrix once.
5. Store the initial state, then propagate in a loop.
6. Calculate norms, position means, widths, and energies.
7. Compare a small calculation with the matrix exponential.

## Shapes and dtypes

```text
grid:          (N,)             float64
initial state: (N,)             complex128
history:       (num_steps+1, N) complex128
batch input:   (N, batch)       complex128
batch history: (num_steps+1, N, batch)
```

Do not let Python silently discard the phase by storing a complex state in a
real array.

## Sparse NumPy/SciPy path

- Construct `H` with `scipy.sparse.diags`.
- Construct the identity with a complex dtype.
- Convert the left matrix to CSC format for `splu`.
- Reuse the returned factorization at every time step.

## PyTorch path

- Use `torch.complex128` for the Hamiltonian and states.
- Use `torch.linalg.lu_factor` once and `torch.linalg.lu_solve` repeatedly.
- Put independent initial states in columns so one solve advances the batch.
- Create tensors directly on the requested device.

## Tests to write

- The Gaussian has the requested center and standard deviation.
- `H` is Hermitian.
- Norm and energy drift remain near floating-point precision.
- The packet center and width follow their analytical formulas.
- NumPy and PyTorch histories agree.
- Halving `dt` reduces final-state error by about four.
- Propagating with `-dt` reverses propagation with `+dt`.

## Common mistakes

- Using `exp(-((x-x0)/sigma)**2 / 2)` for the wavefunction and then expecting
  the probability density to have standard deviation `sigma`.
- Refactorizing the left matrix inside the time loop.
- Comparing complex states without aligning their global phases.
- Allowing the packet to reach an artificial boundary during validation.
- Treating norm conservation as proof that the chosen time step is accurate.

## Foundations and PyTorch practice

Learn complex128 states, torch.exp for phases, conjugate inner products, torch.linalg.lu_factor, and torch.linalg.lu_solve. Reuse one factorization for all time steps and batch columns. Compare PyTorch matrix_exp with the SciPy sparse exponential action at identical parameters.

Implement the PyTorch starter first, then compare with the NumPy reference.
