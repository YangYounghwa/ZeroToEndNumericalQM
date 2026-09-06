# PyTorch learning update

The roadmap now makes PyTorch the main learning target, with NumPy for
comparison and SciPy for selected sparse numerical methods. Foundations are
introduced within chapters; there is no Phase 0 folder.

The roadmap was committed first, as requested:
`066b5a7` — `docs: center roadmap on PyTorch and integrated foundations`.
The chapter updates were committed separately:

- `0e1b0cb` — Phase 1 stationary chapters, exercises, and validation.
- `8c64568` — Phase 2 evolution references and convergence studies.

## Existing chapters updated

| Area | Change and reason |
| --- | --- |
| All nine existing chapters | PyTorch-first reading order, commands, and foundation notes; matching starter updates where methods changed |
| Stationary Chapters 1–4 | Weighted eigenpair residuals in both implementations; PyTorch convergence experiments |
| Harmonic oscillator | Gaussian optimization with autograd/Adam, oscillator-basis expansion, NumPy comparison, and grid-versus-basis study |
| Double well | Domain study at fixed spacing, checking the low-energy tunneling pair |
| General 1D potential | Sparse SciPy eigensolver alongside the dense reference; PyTorch grid/domain studies and batched potential comparison |
| Radial hydrogen | PyTorch convergence experiments; retained the existing sparse reference and radial physics |
| 2D stationary | Small PyTorch versus sparse SciPy comparisons, dense-memory reporting, fixed-spacing domain study, and degenerate-subspace overlap exercise |
| Both time-evolution chapters | Native PyTorch matrix-exponential references and phase-aligned errors; SciPy sparse exponential action; independent time/grid/domain studies |
| Harmonic packet | Reject zero-norm states and invalid propagation shapes consistently |

SciPy was already a project dependency. No additional packages were needed.
The core PyTorch solvers stay in PyTorch; conversions occur at comparison and
reporting boundaries. Larger 2D convergence calculations retain sparse SciPy.

## Verification

- Existing baseline: 123 tests passed.
- Updated suite: **149 tests passed**.
- Strict mypy: **85 source files checked, no issues**.
- Ruff lint and formatting checks passed.
- All **nine published convergence scripts** and the **variational extension** ran successfully.
- PyTorch smoke test passed for CPU tensors, float64, eigendecomposition, and autograd.
- CUDA was unavailable in the installed CPU build, so GPU execution was not verified.

Detailed experiment output is saved locally in the ignored root `workbench/`
directory, in files named after each experiment.

## Results worth examining

1. The general 1D solver's PyTorch/SciPy ground-energy differences were below
   2e-14 in the grid study, while continuum energy error ranged from about
   4.8e-3 to 7.8e-5. Agreement between libraries does not prove grid accuracy.
2. Halving the time step reduced packet state error by approximately four,
   while norm error stayed around machine precision. Unitarity does not prove
   accurate dynamics.
3. The harmonic packet's domain study plateaued near 1.3e-2 state error at
   fixed dx = 0.1. Enlarging the box cannot remove the remaining grid error.
4. For the quartic oscillator with lambda = 0.1, the optimized Gaussian gave
   E ≈ 0.56030737. The 20-state oscillator basis gave E ≈ 0.55914633.
   The 320-point grid gave E ≈ 0.55903604 and still needs refinement.
   These are different approximations to the same continuum problem.

The advanced topics remain roadmap items. No new many-body, tensor-network,
Monte Carlo, scattering, or FFT chapters were created in this update.
