# Coding hints

Read [theory.md](theory.md), then [numerical_method.md](numerical_method.md).
Start with the PyTorch exercise; NumPy is the comparison.

## Shapes and dtypes

| Object | Shape | dtype |
| --- | --- | --- |
| x, k, potential, kinetic energy | `(N,)` | float64 |
| Packet columns | `(N, B)` | complex128 |
| Evolution phase columns | `(N, 1)` | complex128 |
| Saved batch history | `(S, N, B)` | complex128 |
| Single-packet result history | `(S, N)` | complex128 |
| Left/near/right probabilities | `(S, 3)` | float64 |

Create tensors on the selected device from the start. NumPy comparison data
must be on the CPU. Avoid conversions inside the propagation loop.

## Suggested implementation order

1. Build the endpoint-excluded grid with `arange`, then `2*pi*fftfreq`.
2. Sample the barrier and Gaussian packet. Normalize each state column with `dx`.
3. Precompute the half-potential and full-kinetic phases.
4. Apply potential half-step, FFT, kinetic step, inverse FFT, potential half-step.
5. Save requested snapshots, always including the first and last states.
6. Add weighted region sums, spectral energy, edge strips, and high-k probability.
7. Implement the small spectral matrix reference and phase-aligned state error.
8. Compare with NumPy FFT; then construct the periodic finite-difference CSR
   matrix for the SciPy comparison, including both corner links.

Use `dim=0` for `(N,B)` packet columns and `dim=1` for a single-packet `(S,N)`
history. Specify `norm="ortho"` on both FFT and inverse FFT. A complex quantum
state requires a full complex FFT, not `rfft`.

## Exercises

- Propagate an allowed plane wave. Its phase must be
  `exp(-1j*hbar*k**2*t/(2*mass))`; its density must remain uniform.
- Add a constant potential and check the additional phase.
- Batch two incident momenta and compare with two separate runs.
- Reverse a completed evolution with a negative time step.
- Reduce `dt` against the spectral matrix exponential; compare energy errors too.
- Make the box deliberately too small. Explain why norm stays one while the
  measured transmission becomes wrong.
- Increase `sigma`. Compute the incident high-energy tail before running the
  collision, and check that the initial packet still fits away from the barrier.

The starters contain signatures and TODOs, with no solution bodies. Copy them
to a subject folder under `workbench/` to preserve the supplied exercises.
The convergence starter imports the completed solution modules; point those
imports to your own implementation when testing your work.

Run the supplied tests to see the expected invariants and comparison tolerances.
Avoid demanding bit-for-bit NumPy/PyTorch agreement: FFT roundoff can accumulate.
