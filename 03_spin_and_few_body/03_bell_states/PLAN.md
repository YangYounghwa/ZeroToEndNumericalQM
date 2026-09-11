# Bell states: implementation plan

1. Keep the two-spin basis `(++, +-, -+, --)`, also written `(00, 01, 10, 11)`.
   Introduce pure-state projectors, statistical mixtures, and density-matrix checks.
2. Implement batched PyTorch partial traces with explicit subsystem axes. Use
   NumPy with index loops as an independent comparison. Include unequal subsystem
   dimensions in tests so that a swapped-axis bug cannot hide behind symmetry.
3. Calculate purity, von Neumann entropy in bits, and Pauli correlations. Compare
   all four Bell states, product states, and a classical mixture with identical
   local states. Define entanglement entropy only for a pure joint state.
4. Connect to the previous chapter through exact exchange-state entanglement.
   Study a Schmidt-angle family and tiny eigenvalues; these are parameter and
   floating-point studies, not grid or time-step convergence experiments.
5. Provide theory, numerical method, hints, exercise starters, and a reproducible
   Markdown report. Verify analytical values, local-unitary invariance, library
   agreement, invalid inputs, and optional CUDA behavior.
6. Update chapter links, run the project checks, and leave the work uncommitted
   for review. The next chapter remains two particles on a lattice.

No new dependencies or sparse solvers are needed for 4-by-4 density matrices.
