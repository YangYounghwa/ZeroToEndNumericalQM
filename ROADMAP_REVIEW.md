# Review of the numerical quantum mechanics roadmap

Historical review, written before the PyTorch-first goal was clarified.
The updated [roadmap](ROADMAP.md) and [change report](UPDATE_REPORT.md) describe
the current direction and implemented changes.

The roadmap is a good overall structure. Its strongest feature is requiring analytical benchmarks, tests, and convergence studies alongside physics problems. I would keep that approach and most of the chapter order.

My main concern is scope: the later chapters form a substantial computational many-body physics program. They should be optional specializations after a clear introductory endpoint. The recommendations below are curriculum judgments; linked sources support the technical points.

## Changes I would prioritize

### 1. Make Phase 0 smaller and teach prerequisites when needed

Begin with complex vectors, matrix multiplication, eigenvalues, probability, derivatives, integration, and basic NumPy. Introduce tensor products with coupled spins, Fourier transforms with propagation, and autograd with variational optimization.

Reason: the stated goal is to learn physics and tools together, but Phase 0 currently resembles several prerequisite courses. Move a simple spin-1/2 example near the beginning: it teaches states, observables, and evolution without grid errors.

### 2. Explicitly add SciPy

Use NumPy for arrays and small dense problems, SciPy for sparse numerical methods, and PyTorch when differentiation or tensor computation serves the problem. SciPy provides sparse eigensolvers, linear solvers, and matrix-exponential action routines. See its [sparse linear algebra documentation](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html).

Reason: sparse diagonalization and Crank–Nicolson are already central to the roadmap, but the tool list omits their natural scientific Python implementation. Avoid requiring two implementations of every elementary exercise.

### 3. Strengthen the numerical accuracy requirements

Add the following to stationary problems:

- Nondimensionalization and explicit units.
- Separate grid-spacing convergence from computational-domain convergence.
- Eigenpair residuals, such as the norm of `H @ psi - E * psi`, with an appropriate scale for reporting relative error.
- Distinguish Euclidean vector normalization from quadrature-weighted wavefunction normalization.
- Compare degenerate eigenspaces rather than demanding identical individual eigenvectors.

Reason: refining the mesh inside a box that is too small can converge to the wrong physical answer. Normalization and Hermiticity alone do not establish accuracy.

In time evolution, vary spatial resolution and time step independently and compare observables or state errors with a reference solution. Norm conservation and reversibility can both hold while accumulated phase error is large.

Make conservation requirements conditional: norm conservation applies to closed, unitary evolution. An absorbing boundary intentionally removes probability. Standard FFT propagation also imposes periodicity, so scattering simulations need a domain large enough to avoid wraparound or a suitable absorber. The [Algorithm Archive implementation](https://www.algorithm-archive.org/contents/split-operator_method/split-operator_method.html) explains this periodic boundary behavior.

### 4. Add a prerequisite section before Hubbard models

Teach identical particles, bosonic and fermionic exchange symmetry, occupation-number states, creation and annihilation operators, their algebra, and fermionic signs before constructing Hubbard Hamiltonians. Include a local occupation cutoff and its convergence for bosons.

Reason: tensor products alone do not explain the particle statistics needed for these models. MIT's [identical-particle notes](https://ocw.mit.edu/courses/8-06-quantum-physics-iii-spring-2018/fb48dd49e8649557f1715f0415de4f4d_MIT8_06S18ch8.pdf) provide a physics reference for this prerequisite.

### 5. Introduce basic variational ideas earlier

Place one simple Rayleigh–Ritz exercise after the first stationary solvers: optimize a trial wavefunction and compare its energy with the grid result. Retain the later phase for more substantial optimization and PyTorch autograd.

Also add one basis-expansion calculation, such as an anharmonic oscillator in a truncated harmonic-oscillator basis, and compare basis-size convergence with grid convergence.

Reason: this makes numerical quantum mechanics broader than finite differences and connects approximation quality to the choice of representation.

### 6. Treat the advanced phases as branches

After small-system exact diagonalization and variational methods, choose tensor networks or Monte Carlo according to interest. Neither requires completing the other first. DMRG is particularly connected to matrix product states and one-dimensional many-body systems; see [Schollwöck's review](https://arxiv.org/abs/1008.3477).

Keep VMC as the first quantum Monte Carlo project. Add imaginary-time propagation before DMC or ground-state AFQMC. The claim that all QMC uses the variational principle is too broad: ground-state AFQMC uses imaginary-time projection, as described in this [AFQMC research paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11137827/). Variational methods are useful preparation, but not a complete explanation of QMC.

Neural wavefunctions can follow VMC directly; completing DMC and AFQMC first is unnecessary. For this branch, add local-energy estimators and stochastic reconfiguration, which [NetKet documents](https://netket.readthedocs.io/en/stable/user-guides/sr.html) as a geometry-based optimization method.

## A practical first endpoint

Complete these projects before treating advanced methods as required:

1. A two-state spin calculation with analytical checks.
2. A reusable 1D stationary solver with separate grid and domain studies.
3. Wave-packet evolution with reference comparisons and a barrier-scattering probability budget.
4. Coupled spins with reduced density matrices and known entanglement checks.
5. A small spin-chain exact-diagonalization calculation.
6. A variational calculation benchmarked against an exact result.

Keep radial hydrogen and 2D as extensions; make 3D optional. The roadmap currently promises a 3D extension in Phase 1's milestone even though its listed systems stop at 2D. Radial hydrogen also needs an explicit explanation of the reduced radial wavefunction, origin boundary condition, centrifugal term, and normalization measure.

These changes would preserve the roadmap's strengths while making the first learning target manageable and its numerical results easier to trust.
