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
