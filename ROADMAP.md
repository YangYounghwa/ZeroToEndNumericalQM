# Learning Roadmap

## Goal

Learn quantum mechanics and numerical methods while developing practical skill
with NumPy and PyTorch.

The subjects should not be learned as independent tracks. Each physics problem
introduces the mathematics, numerical method, NumPy tools, and PyTorch tools
needed to solve it.

## Phase 0: Foundations

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

### Python tools

- NumPy arrays, shapes, indexing, and broadcasting
- Vectorization
- Complex dtypes
- Matrix multiplication
- Basic plotting
- PyTorch tensors, dtypes, and devices
- PyTorch autograd and optimizers
- Tests and reproducible experiments
- `uv` environments and commands

### Completion condition

Implement and test small exercises involving complex vectors, normalization,
matrix operators, eigenvalue problems, numerical derivatives, and integration.

## Phase 1: One-Particle Stationary Problems

### Physical systems

1. Infinite square well
2. Harmonic oscillator
3. Finite square well
4. Double-well potential
5. General one-dimensional potential
6. Radial hydrogen equation
7. General two-dimensional stationary potential
8. General three-dimensional stationary potential

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

### Required checks

- Hamiltonian is Hermitian
- Wavefunctions are normalized
- Eigenstates are orthogonal
- Numerical energies agree with analytical results when available
- Error decreases as grid resolution increases

### Milestone

Create a reusable one-dimensional stationary Schrödinger solver. Verify it on
the infinite square well and harmonic oscillator before adding general
potentials. Extend the same finite-difference ideas to sparse 2D and 3D
Cartesian grids while preserving explicit indexing conventions.

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

### Required checks

- Norm is conserved
- Energy is conserved for a time-independent Hamiltonian
- Error decreases with the time step
- Forward and reverse evolution recover the initial state within tolerance

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

This phase should come before tensor networks and quantum Monte Carlo because it
introduces the variational principle used by both.

### Topics

- Rayleigh-Ritz variational principle
- Parameterized trial states
- Energy expectation values
- Gradient-based minimization
- Automatic differentiation with PyTorch
- Orthogonality constraints for excited states

### NumPy and PyTorch roles

The NumPy implementation provides a transparent reference. The PyTorch
implementation should use autograd and optimizers rather than copying the NumPy
algorithm line for line.

## Phase 6: Tensor Networks

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

These methods should be separate chapters because they solve different problems
and have different sources of error.

### Recommended order

1. Random sampling and statistical error
2. Metropolis-Hastings sampling
3. Variational Monte Carlo
4. Diffusion Monte Carlo
5. Auxiliary-field quantum Monte Carlo

### Required topics

- Burn-in and autocorrelation
- Effective sample size
- Error bars
- Importance sampling
- Bias and variance
- Fermion sign and phase problems

Auxiliary-field quantum Monte Carlo is an advanced endpoint, not a simple
extension of variational Monte Carlo.

## Phase 8: Modern Methods

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
00_foundations/
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

Do not implement all phases at once. Finish one chapter, including its tests and
convergence study, before starting the next chapter.
