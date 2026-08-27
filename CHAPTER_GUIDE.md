# Chapter Structure and Workflow

## Standard Chapter Layout

Each subject should have its own organized folder.

```text
chapter_name/
├── theory.md
├── numerical_method.md
├── coding_hints.md
├── boilerplates/
│   ├── problem_name_numpy_starter.py
│   ├── problem_name_torch_starter.py
│   └── problem_name_convergence_starter.py
├── solutions/
│   ├── problem_name_numpy_solution.py
│   ├── problem_name_torch_solution.py
│   └── problem_name_convergence.py
└── tests/
    ├── test_problem_name_numpy.py
    └── test_problem_name_torch.py
```

Do not use `.ipynb` files.

## `theory.md`

Explain:

- the physical system;
- the assumptions;
- the governing equations;
- the boundary and initial conditions;
- important observables;
- known analytical results;
- expected physical behavior.

Include the required quantum-mechanics equation derivation directly in this
file. The learner is not required to reproduce the full derivation by hand.
Keep it detailed enough to explain every term used by the implementation, but
omit unrelated formalism that does not help solve or validate the problem.

## `numerical_method.md`

Explain:

- how the continuous equation becomes a discrete problem;
- the algorithm step by step;
- accuracy and error order;
- stability conditions;
- computational time and memory costs;
- numerical failure modes;
- appropriate validation tests.

## `coding_hints.md`

Provide guidance without giving the complete implementation:

- expected array shapes;
- recommended functions;
- pseudocode;
- dtype choices;
- boundary-condition handling;
- tests to write;
- common mistakes.

## NumPy Implementation

`problem_name_numpy_solution.py` is the reference implementation. It should
prioritize:

- clear equations;
- explicit array shapes;
- vectorized operations;
- correct complex dtypes;
- small, testable functions;
- minimal hidden state.

Avoid optimizing code before correctness and convergence have been established.

## PyTorch Implementation

`problem_name_torch_solution.py` should reproduce the verified NumPy result. It
must handle:

- explicit `dtype`;
- explicit device selection;
- tensor shape checks;
- conversion to NumPy only at clear boundaries;
- reproducible random seeds when randomness is used.

PyTorch should add value through at least one of:

- automatic differentiation;
- gradient-based optimization;
- GPU execution;
- batched calculations;
- large tensor contractions.

For a small dense eigenvalue problem, PyTorch may not be faster than NumPy. The
purpose of the early PyTorch chapters is API learning and result comparison, not
performance claims.

## Experiment File

`problem_name_convergence.py` should run controlled investigations such as:

- error versus grid spacing;
- error versus time-step size;
- runtime versus system size;
- memory use versus basis size;
- convergence versus iteration count;
- convergence versus tensor-network bond dimension.

Keep reusable algorithms out of the experiment file.

## Chapter Workflow

1. Read the physical explanation and equation derivation in `theory.md`.
2. Understand the discretization in `numerical_method.md`.
3. Implement the NumPy version using `coding_hints.md`.
4. Test NumPy against an analytical or trusted reference result.
5. Perform a convergence study.
6. Implement the PyTorch version.
7. Compare NumPy and PyTorch within a defined tolerance.
8. Record errors, limitations, and conclusions.
9. Start the next chapter only after the completion criteria pass.

## Definition of Done

A chapter is complete when:

- its physics, required equation derivation, and numerical method are
  documented;
- NumPy and PyTorch implementations run through `uv`;
- unit tests pass;
- NumPy and PyTorch agree within a justified tolerance;
- a convergence study is included;
- relevant physical invariants are checked;
- results are compared with analytical or exact reference data when available;
- limitations are documented.

A graph that looks physically reasonable is not sufficient evidence.

## First Chapter Plan

### Subject

One-dimensional infinite square well using finite differences.

### Learning goals

- Construct a spatial grid.
- Approximate the second derivative.
- Build a Hamiltonian matrix.
- Solve a Hermitian eigenvalue problem.
- Normalize discrete wavefunctions.
- Compare numerical and analytical energies.
- Measure convergence as the grid is refined.
- Reproduce the calculation with PyTorch tensors.

### Required tests

- Hamiltonian is symmetric or Hermitian.
- All computed energies are real and positive.
- Eigenvectors are orthonormal under the chosen discrete inner product.
- The lowest energies follow the analytical `n^2` pattern.
- Energy error decreases when grid spacing decreases.
- NumPy and PyTorch results agree within tolerance.

## Second Chapter Plan

### Subject

One-dimensional quantum harmonic oscillator using finite differences.

### Learning goals

- Add a position-dependent potential to the kinetic matrix.
- Approximate decay at infinity with a finite computational domain.
- Separate grid-discretization error from domain-truncation error.
- Verify energy, parity, normalization, and position expectation values.
- Reproduce the calculation with PyTorch tensors.

### Next step

Solve the finite square well. It introduces bound-state selection, continuum
box states, and matching to a transcendental analytical reference.

## Third Chapter Plan

### Subject

One-dimensional finite square well using finite differences and analytical
matching equations.

### Learning goals

- Distinguish physical potential edges from artificial outer boundaries.
- Select bound states using the continuum threshold.
- Measure probability leakage outside the well.
- Compare eigenvalues with even and odd transcendental equations.
- Recognize grid-alignment error at a discontinuous potential.

## Fourth Chapter Plan

### Subject

Symmetric quartic double-well potential and tunneling splitting.

### Learning goals

- Relate reflection symmetry to eigenstate parity.
- Measure the splitting of the lowest even-odd pair.
- Build approximate localized states from parity eigenstates.
- Study how barrier height changes the splitting.
- Validate a problem without a simple analytical spectrum.

### Next step

Extract a reusable one-dimensional stationary solver. The first four chapters
intentionally repeat the construction so the common interface is now visible.

## Fifth Chapter Plan

### Subject

Reusable stationary solver for general real one-dimensional potentials.

### Learning goals

- Pass the physical potential into a common solver interface.
- Support nonsymmetric finite domains and potentials.
- Validate potential arrays before Hamiltonian construction.
- Check eigenpair residuals as well as orthonormality.
- Verify the interface with a shifted harmonic oscillator.
- Use PyTorch batched eigendecomposition for multiple sampled potentials.
- State the accuracy and scaling limits of dense diagonalization.

### Next step

Solve the radial hydrogen equation. It introduces the reduced radial
wavefunction, a singular endpoint, and angular-momentum-dependent effective
potentials.

## Sixth Chapter Plan

### Subject

Reduced radial hydrogen equation on a finite half-line.

### Learning goals

- Derive the reduced radial equation using `u(r) = r R(r)`.
- Exclude the singular origin while enforcing `u(0) = 0`.
- Include the centrifugal effective potential for fixed angular momentum.
- Use a sparse tridiagonal Hamiltonian and request only low-energy eigenpairs.
- Verify hydrogenic energies, radial expectation values, and degeneracies.
- Separate radial-spacing error from outer-boundary truncation error.
- Compare the sparse NumPy/SciPy result with dense and batched PyTorch results.

### Next step

Build a genuine two-dimensional Cartesian solver without using symmetry to
reduce the number of coordinates.

## Seventh Chapter Plan

### Subject

General two-dimensional stationary potentials on a rectangular grid.

### Learning goals

- Store fields with shape `(Ny, Nx)` and document C-order flattening.
- Construct a sparse two-term Kronecker-sum kinetic operator.
- Accept an arbitrary sampled `V(x, y)` without assuming separability.
- Solve a coupled-quartic potential with an `x^2 y^2` interaction.
- Normalize with the area element and calculate 2D observables.
- Use separable oscillator energies only as an analytical validation case.
- Verify a nonseparable problem against a refined numerical reference.
- Compare sparse NumPy with small dense and batched PyTorch calculations.

## Eighth Chapter Plan

### Subject

General three-dimensional stationary potentials on a Cartesian grid.

### Learning goals

- Extend the flattening convention to `(Nz, Ny, Nx)`.
- Construct a sparse three-term Kronecker-sum Hamiltonian.
- Normalize with the volume element and verify 3D observables.
- Verify anisotropic oscillator energies and isotropic triplet degeneracy.
- Calculate dense-memory requirements before selecting a grid.
- Restrict dense PyTorch calculations to deliberately small references.

### Next phase

Begin time evolution with a free Gaussian wave packet. Introduce complex
wavefunctions, unitary propagation, norm conservation, and time-step
convergence before adding external potentials.

## Phase 2, First Chapter Plan

### Subject

Free Gaussian wave-packet evolution.

### Learning goals

- Construct a normalized complex Gaussian initial state.
- Derive its analytical center motion and spreading.
- Implement sparse Crank-Nicolson propagation with a reused factorization.
- Use a dense matrix exponential as a small-system reference.
- Verify norm, energy, reversibility, and second-order time convergence.
- Propagate batches of initial momenta with PyTorch.

## Phase 2, Second Chapter Plan

### Subject

Coherent Gaussian wave packet in a harmonic potential.

### Learning goals

- Add a position-dependent potential to the time-evolution Hamiltonian.
- Construct a coherent state with the oscillator ground-state width.
- Verify sinusoidal center motion and constant packet width.
- Check periodic return using global-phase-independent fidelity.
- Separate spatial phase error from time-step error.
- Batch several coherent initial displacements with PyTorch.

### Next step

Study scattering from a potential barrier. Measure reflected and transmitted
probabilities only after the outgoing packets are spatially separated.
