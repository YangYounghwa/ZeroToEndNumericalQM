# Coding hints

## Suggested order

1. Construct the three complex128 Pauli matrices and test their products.
2. Construct a spinor from `theta` and `phi`. Check normalization and known
   `+x`, `+y`, `+z` measurement results.
3. Contract fields with Pauli matrices to build `H=-gamma*hbar*B.sigma/2`.
4. Use batched matrix exponentials for a common time vector and multiple fields.
5. Compute Bloch vectors, arbitrary-axis measurement probabilities, and energy.
6. Implement the Pauli closed form and Rodrigues rotation as independent checks.
7. Add reused-LU Crank-Nicolson and measure its time-step error.
8. Compare with the NumPy eigenbasis implementation.

The signed angular-velocity vector is `-gamma*field`. For a positive z field,
positive gamma, and initial +x, the initial motion must be toward -y.
This catches a sign error that norm and energy cannot detect.

## PyTorch details

- Use `.conj()` for bras and `.mH` for the last two matrix axes.
- Use `torch.linalg.matrix_exp`, not elementwise `torch.exp`, for a general matrix.
- `einsum("ba,aij->bij", fields_complex, pauli)` builds all Hamiltonians.
- For `(T,B,2,2)` propagators, reshape initial states as `(1,B,2,1)` before
  matrix multiplication and remove only the final singleton axis.
- A single field still needs a batch axis: `(1,3)`, paired with state `(1,2)`.
- Normalize over the spin axis, never over the experiment batch.
- `torch.sinc(z/pi)` evaluates `sin(z)/z`, including the limit at zero.
- `torch.linalg.eigh` returns eigenvectors as columns. Eigenvector signs and
  complex phases need not agree across libraries.

## Exercises

- Start at +z with a z field. Show that the state only acquires a phase.
- Start at +z with an x field. Verify `P(-z)=sin(omega*t/2)**2`.
- Change the sign of gamma. Show that the precession direction reverses.
- Set the field or gamma to zero. Verify the identity propagator.
- Change a spinor's global phase, then its relative phase. Explain which
  measurement probabilities change.
- Measure along the initial Bloch direction; a pure aligned state has P(+)=1.
- Check the spinor after `2*pi` and `4*pi` rotations, then compare Bloch vectors.
- Propagate two field/state pairs as a batch and as separate calculations.
- At a fixed final time, halve the CN time step. Compare phase-aligned state
  errors and spin observables, alongside norm and energy.

Copy the starter files to a subject folder under `workbench/`. They provide
public signatures and TODOs. Add imports as needed, including imports of your
own implementations in the convergence exercise.

The field is an externally specified input, not a quantum subsystem. A batch
of two spinors is therefore not the tensor-product state of two coupled spins;
that distinction becomes central in the next chapter.
