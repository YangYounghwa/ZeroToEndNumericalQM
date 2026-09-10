# Coding hints

## Shapes to keep visible

| Quantity | Shape |
| --- | --- |
| x, kx | `(Nx,)` |
| y, ky | `(Ny,)` |
| Potential and kinetic energy | `(Ny,Nx)` |
| Initial packet batch | `(Ny,Nx,B)` |
| Propagator history | `(S,Ny,Nx,B)` |
| Single-packet history | `(S,Ny,Nx)` |
| Centers | `(S,2)`, ordered x then y |
| Covariances | `(S,2,2)` |

Grid and potential arrays use float64; packet and phase arrays use complex128.
Keep them on one explicit device. Use unequal `Nx`/`Ny` and `dx`/`dy` in your
first checks, so shape assumptions cannot hide.

## Implementation order

1. Build the two periodic axes and wave numbers.
2. Construct the kinetic energy by broadcasting the squared wave numbers.
3. Normalize packet batches using `dx*dy` and a sum over `(0,1)`.
4. Generalize the 1D split step to `fft2`, explicitly naming spatial axes.
5. Verify exact plane-wave and free-Gaussian evolution before adding a potential.
6. Build `K`, check that it is symmetric and positive definite, and use
   `torch.linalg.eigh` to obtain normal-mode frequencies.
7. Construct the coupled Gaussian with `Omega = Q @ diag(nu) @ Q.T`.
8. Measure centers, covariance, spectral energy, and boundary-strip probability.
9. Compare with exact normal-mode motion, then NumPy.

`eigh` returns squared frequencies here. Take their square roots before using
oscillatory time factors or constructing the ground-state width. Eigenvector
signs are arbitrary; expressions such as `Q @ diag(nu) @ Q.T` remove that
ambiguity. Do not compare individual mode signs across libraries.

## Exercises

- Remove the coupling and recover the product of two 1D oscillator evolutions.
- Restore coupling and observe the nonzero xy covariance.
- Batch two displaced packets under the same potential; compare separate runs.
- Reverse propagation with negative `dt` and check the recovered state.
- At a fixed fine grid, halve `dt` and compare the entire state after removing
  its global phase. Check center and energy errors too.
- Make one box axis too short. Track its boundary tail and explain why norm
  conservation does not rule out a wrong result.
- Pass a smooth nonseparable potential such as `cos(pi*x/Lx)*sin(pi*y/Ly)` to
  the generic propagator. On a tiny grid, validate with the spectral matrix
  exponential; the coherent-state formula no longer applies.

The starters contain public signatures and TODOs. Copy them into a subject
folder under `workbench/`. Add imports as needed. The convergence starter's
solution imports should point to your own implementation when testing your work.
