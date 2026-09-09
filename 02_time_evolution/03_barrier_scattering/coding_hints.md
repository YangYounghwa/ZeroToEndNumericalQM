# Coding Hints

Implement the PyTorch starter before reading its solution. Use the earlier
packet chapters for LU factorization and complex arithmetic.

## Tensor layout

| Quantity | Shape | Type |
| --- | --- | --- |
| Grid and potential | (N,) | float64 |
| Dense Hamiltonian | (N, N) | float64 |
| Initial packet batch | (N, B) | complex128 |
| LU factors and right matrix | (N, N) | complex128 |
| Batch snapshots | (saved_times, N, B) | complex128 |
| Single-packet snapshots | (saved_times, N) | complex128 |
| Region probabilities | (saved_times,) | float64 |
| One state's interior-link currents | (N-1,) | float64 |

The barrier and region masks are Boolean tensors with shape (N,). Keep every
related tensor on the same device. Convert only at comparison/report boundaries.

## Suggested implementation order

1. Create the interior grid and confirm its missing zero endpoints.
2. Sample the barrier with `torch.where`. Construct the height tensor directly
   in float64; casting a rounded float32 constant afterwards loses precision.
3. Build H with `torch.diag` and verify Hermiticity.
4. Construct the Gaussian envelope and complex phase, then normalize columns.
5. Factor the CN left matrix once with `torch.linalg.lu_factor`.
6. Update all columns with `torch.linalg.lu_solve`; save selected snapshots.
7. Reduce density under disjoint region masks and calculate full energy.
8. Add current, analytical transmission, and the spectrum integral.
9. Run the four convergence studies and record a valid measurement window.

## Batch pseudocode

```text
construct one grid and one Hamiltonian
stack Gaussian packets with different k0 along dimension 1
normalize each column using dx
factor A once
for each time step:
    states = solve(A, B @ states)
    save only when requested, and always save the last step
stack saved states along a new time dimension
```

Here B denotes the CN right matrix, not the number of packet columns. Compare
each batch column with an individually propagated packet.

## Measurements

For a cut c, use `grid < -c`, `grid > c`, and the complement of their union.
Multiply density sums by dx. The near probability is part of the budget, while
edge-strip probability is an overlapping diagnostic. Initially the left region
contains the incident packet: its probability is not yet R.

Use `.conj()` before multiplying amplitudes in energy, overlaps, or current.
For a midpoint current test, save every integration step. For a phase-aligned
state comparison, compute the phase of dx*vdot(reference, numerical).

## Analytical transmission

Use the equations in theory.md. Separate masks avoid square roots of negative
real arguments and undefined threshold divisions. Include E=0, E=V0, no barrier,
and thick barriers in tests. Compare the formula with an independent solution
of the four continuity equations.

Increasing sigma narrows the momentum distribution but makes the packet wider
in position. Increase the initial separation/domain and recheck the measurement
time when changing sigma; an analytical spectrum comparison alone does not
validate a newly chosen time-dependent simulation.

## SciPy comparison

Keep NumPy grids and states as arrays. Build the three operator diagonals with
`scipy.sparse.diags`, factor the left matrix once in CSC form with `splu`, and
use the returned object's solve method. This is a numerical comparison, not a
separate SciPy learning track. Do not route the PyTorch solver through SciPy.

## Expected checks

Energy and norm conservation, complete probability budget, local continuity,
forward/reverse evolution, batch/individual agreement, PyTorch/SciPy agreement,
second-order time error, aligned grid convergence, and domain/observation-time
checks should all support the interpretation. A plausible packet shape alone
does not establish accurate transmission.
