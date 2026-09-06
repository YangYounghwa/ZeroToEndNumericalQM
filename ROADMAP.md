# Learning Roadmap

## Goal

Learn PyTorch through numerical quantum mechanics. Use NumPy implementations
for comparison and SciPy only where sparse numerical methods are useful.

The subjects should not be learned as independent tracks. Each physics problem
introduces the mathematics, numerical method, and PyTorch tools needed to solve
it. Implement and validate the PyTorch calculation first, then compare it with
the NumPy reference. Do not treat SciPy as a separate course or require a third
implementation of every problem.

## Foundations Introduced Within Chapters

There is no Phase 0 folder or prerequisite course. Each chapter should explain
the foundations it needs in its theory, numerical-method, and coding-hints
documents. Use this list as a coverage checklist, not a gate before Phase 1.

### Mathematics

- Complex numbers
- Vectors, matrices, and tensors
- Inner products and orthogonality
- Eigenvalues and eigenvectors
- Tensor products
- Derivatives and integrals
- Fourier transforms
- Basic probability

### Quantum mechanics

- Wavefunctions and probability density
- State normalization
- Operators and observables
- Expectation values
- Bra-ket notation
- Hilbert spaces
- Boundary conditions
- Stationary and time-dependent Schrödinger equations

### Numerical methods

- Floating-point error
- Discretization error
- Finite differences
- Numerical quadrature
- Dense and sparse matrices
- Convergence studies
- Stability and conservation laws
- Nondimensionalization and units
- Separate domain, grid, time-step, and iterative-solver errors

### Python tools

- NumPy arrays, shapes, indexing, and broadcasting
- Vectorization
- Complex dtypes
- Matrix multiplication
- Basic plotting
- PyTorch tensors, dtypes, and devices
- PyTorch autograd and optimizers
- SciPy sparse matrix construction, low-energy eigensolvers, and reused LU
  factorizations when introduced by a problem
- Tests and reproducible experiments
- `uv` environments and commands

### Introduction points

- Infinite well: tensor shapes, matrix operators, eigenvalues, integration,
  normalization, and discretization error.
- Harmonic oscillator: units, nondimensionalization, finite-domain error, and
  a simple variational trial state.
- General 1D potential: batching, sparse-versus-dense comparison, and residuals.
- Radial hydrogen: reduced radial wavefunctions, the origin boundary condition,
  the centrifugal term, and the radial integration measure.
- 2D potentials: reshaping, Kronecker sums, and memory scaling.
- Time evolution: complex tensors, probability conservation, linear solves,
  matrix exponentials, and Fourier transforms when FFT propagation is used.
- Coupled spins: tensor products, density matrices, and partial traces.
- Variational optimization: autograd and optimizers; introduce small examples
  earlier when they help explain the current physics.

## Phase 1: One-Particle Stationary Problems

### Physical systems

1. Infinite square well
2. Harmonic oscillator
3. Finite square well
4. Double-well potential
5. General one-dimensional potential
6. Radial hydrogen equation
7. General two-dimensional stationary potential

### Methods

- Spatial grids
- Finite-difference derivatives
- Numerical quadrature
- Hamiltonian construction
- Dense diagonalization
- Sparse diagonalization
- Boundary-condition handling
- Kronecker-sum operators
- Multidimensional indexing and reshaping
- Dense-memory scaling estimates
- A small harmonic-oscillator basis expansion as a comparison with grid methods
- A Gaussian variational estimate before the later optimization phase

### Required checks

- Hamiltonian is Hermitian
- Wavefunctions are normalized
- Eigenstates are orthogonal
- Numerical energies agree with analytical results when available
- Error decreases as grid resolution increases
- Domain-size convergence is checked separately at fixed or controlled spacing
- Eigenpair residuals are small; normalization uses the integration weight
- Degenerate states are compared through subspace overlaps, allowing basis
  rotations and arbitrary eigenvector signs/phases
- Numerical units and the meaning of each grid boundary are documented

### Milestone

Create a reusable one-dimensional stationary Schrödinger solver. Verify it on
the infinite square well and harmonic oscillator before adding general
potentials. Add a SciPy sparse comparison to the reusable 1D solver and extend
the finite-difference ideas to sparse 2D Cartesian grids with explicit indexing
conventions. Use small dense and batched PyTorch problems to learn the tensor
operations. Radial hydrogen and 2D are extensions after the 1D core; 3D is optional
future work, not a completion requirement.

## Phase 2: Time Evolution

### Physical systems

1. Free Gaussian wave packet
2. Wave packet in a harmonic potential
3. Scattering from a potential barrier
4. Quantum tunnelling
5. Two-dimensional wave-packet propagation

### Methods

- Matrix exponential as a small-system reference
- Crank-Nicolson propagation
- Split-operator Fourier propagation
- Krylov-subspace propagation
- Fast Fourier transforms
- Reuse linear-system factorizations; do not form an explicit inverse
- Sparse matrix-exponential action as a larger-system reference

### Required checks

- Norm is conserved for closed, Hermitian evolution
- Energy is conserved for a time-independent Hamiltonian
- Error decreases with the time step
- Forward and reverse evolution recover the initial state within tolerance
- Compare state fidelity or phase-aligned state error and physical observables
  with analytical dynamics or a reference on the same spatial grid
- Study spatial resolution and domain size separately from time-step error
- Document boundary reflections; FFT propagation has periodic boundaries
- With absorbing boundaries, track removed probability instead of requiring
  norm conservation or exact reversibility

Norm conservation and reversibility alone do not establish accurate dynamics.
For scattering, measure reflection and transmission after packet separation and
account for probability remaining near the barrier or removed by absorbers.

## Phase 3: Spin and Few-Body Systems

### Physical systems

1. Spin-1/2 in a magnetic field
2. Two coupled spins
3. Bell states
4. Two particles on a lattice
5. A small Hubbard system

### Methods and concepts

- Pauli matrices
- Kronecker products
- Composite Hilbert spaces
- Sparse matrices
- Partial traces
- Reduced density matrices
- Lanczos eigensolver
- Correlation functions
- Entanglement entropy
- Identical particles, exchange symmetry, and occupation-number/Fock states
- Creation and annihilation operators and their (anti)commutation relations
- Fermionic signs before constructing a Hubbard Hamiltonian
- Bosonic occupation cutoffs and cutoff convergence

### Milestone

Construct composite operators without manually writing full matrices and verify
entanglement calculations on known two-qubit states.

## Phase 4: Many-Body Exact Diagonalization

### Models

1. Transverse-field Ising chain
2. Heisenberg spin chain
3. Bose-Hubbard model
4. Fermi-Hubbard model

### Methods

- Bit-string basis representations
- Sparse Hamiltonian construction
- Symmetry sectors
- Particle-number conservation
- Ground-state and excited-state calculations
- Correlation functions
- Finite-size scaling

### Milestone

Create exact-diagonalization results that can serve as reference data for later
variational, tensor-network, and Monte Carlo methods.

## Phase 5: Variational Methods

Build on the simple variational exercise introduced with stationary problems.
This phase develops optimization before DMRG and variational Monte Carlo. Other
Monte Carlo methods also need ideas such as imaginary-time projection; they are
not all variational minimization algorithms.

### Topics

- Rayleigh-Ritz variational principle
- Parameterized trial states
- Energy expectation values
- Gradient-based minimization
- Automatic differentiation with PyTorch
- Orthogonality constraints for excited states

### NumPy and PyTorch roles

The PyTorch implementation is the learning target and should use autograd and
optimizers. The NumPy implementation provides a transparent comparison. Check
gradients against an analytical derivative or finite differences on small cases.

## Phase 6: Tensor Networks

This is an optional specialization after exact diagonalization and variational
methods. It does not need to precede quantum Monte Carlo.

### Recommended order

1. Tensor reshaping and contraction
2. Singular-value decomposition
3. Schmidt decomposition
4. Matrix product states
5. Matrix product operators
6. Canonical forms
7. Time-evolving block decimation
8. Density matrix renormalization group

### Required checks

- Compare small systems with exact diagonalization
- Track normalization and truncation error
- Verify convergence with bond dimension
- Test tensor shapes and index ordering

## Phase 7: Quantum Monte Carlo

This is another specialization after small-system benchmarks and variational
methods. Tensor networks are not a prerequisite.

These methods should be separate chapters because they solve different problems
and have different sources of error.

### Recommended order

1. Random sampling and statistical error
2. Metropolis-Hastings sampling
3. Variational Monte Carlo
4. Imaginary-time projection
5. Diffusion Monte Carlo (optional advanced branch)
6. Auxiliary-field quantum Monte Carlo (optional advanced branch)

### Required topics

- Burn-in and autocorrelation
- Effective sample size
- Error bars
- Importance sampling
- Bias and variance
- Fermion sign and phase problems
- Local-energy estimators and stochastic reconfiguration for variational states
- Time-step, population, and constraint biases for the methods that use them

Auxiliary-field quantum Monte Carlo is an advanced endpoint, not a simple
extension of variational Monte Carlo.

## Phase 8: Modern Methods

Choose topics according to the PyTorch skill and physics question being studied.
Neural wavefunctions can follow variational Monte Carlo directly; DMC and AFQMC
are not prerequisites.

### Topics

- Neural-network wavefunctions
- Restricted Boltzmann machine states
- Autoregressive quantum states
- Differentiable physics
- GPU tensor contractions
- Differentiable quantum dynamics

This directory should be called `modern_methods` rather than `sota`, because the
state of the art changes over time.

## Suggested Directory Order

```text
01_stationary_one_particle/
02_time_evolution/
03_spin_and_few_body/
04_exact_diagonalization/
05_variational_methods/
06_tensor_networks/
07_quantum_monte_carlo/
08_modern_methods/
```

## Scope Rule

The first learning endpoint is a verified 1D stationary solver, wave-packet
evolution, coupled spins, a small spin-chain exact-diagonalization calculation,
and a variational calculation. Advanced branches are not required to reach it.

Do not implement all phases at once. Finish one chapter, including its tests and
convergence study, before starting the next chapter.
