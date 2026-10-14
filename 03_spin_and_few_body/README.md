# Spin and few-body systems

Move from spatial grids to finite quantum state spaces. PyTorch remains the
main implementation; NumPy provides independent comparisons. Introduce the
required linear algebra and quantum concepts within each chapter.

## Available chapters

1. [Spin-1/2 in a magnetic field](01_spin_in_magnetic_field/README.md)
2. [Two coupled spins](02_two_coupled_spins/README.md)
3. [Bell states](03_bell_states/README.md)
4. [Two particles on a lattice](04_two_particles_on_lattice/README.md)

The first chapter introduces the spinor, Pauli algebra, signed Larmor
precession, measurement projectors, and the Bloch vector. It compares batched
matrix exponentials with exact rotations and NumPy eigendecomposition.
Crank-Nicolson provides a controlled time-error exercise.

The second chapter builds tensor-product states and local operators, adds
isotropic exchange and separate local fields, and measures joint correlations.
Analytical singlet/triplet energies and transfer dynamics validate the model.

The third chapter introduces density matrices, partial traces, and entropy.
Bell states and a separable classical mixture demonstrate why mixed local
states alone do not establish entanglement. Analytical entropy formulas and
independent NumPy index loops check the batched PyTorch contractions.

The fourth chapter introduces site bases, exchange symmetry, and occupation
numbers for two noninteracting particles. PyTorch symmetry projection and
NumPy occupation-number hopping agree with exact spectra and two-site dynamics.
The fermion model has one fixed spin state and one orbital per site.

## Planned progression

A small Hubbard system comes next. Add interactions and introduce spin
orbitals before allowing opposite-spin fermions to share a spatial site.

A batch of independent simulations is not a composite quantum state. Keep
that distinction explicit when moving from two amplitudes for one spin to
four amplitudes for a pair of spins.

## Checks

Use Hermiticity, normalization, exact small-system dynamics, measurement
probabilities, and independent library comparisons. Study time error only when
using an approximate integrator. A complete two-state spin basis needs no
spatial-grid or basis-cutoff convergence study.
For density matrices, also verify unit trace, positivity, subsystem indexing,
and the distinction between general entropy and pure-state entanglement.
