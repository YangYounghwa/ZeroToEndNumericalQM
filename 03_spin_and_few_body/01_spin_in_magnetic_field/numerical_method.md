# Numerical methods for one spin

## 1. Shapes, types, and normalization

| Quantity | Shape |
| --- | --- |
| Pauli matrices, ordered x/y/z | `(3,2,2)` |
| Magnetic fields | `(B,3)` |
| One initial state for each field | `(B,2)` |
| Hamiltonians | `(B,2,2)` |
| Requested times | `(T,)` |
| Propagators | `(T,B,2,2)` |
| State history | `(T,B,2)` |
| Bloch vectors | `(T,B,3)` |
| Measurement probabilities, + then - | `(T,B,2)` |

Here `B` means batch size, not magnetic-field magnitude. A batch contains
independent spin experiments, not an interacting many-spin system.

Use float64 fields and times, complex128 operators and spinors. Normalize by
`sum(abs(state)**2, dim=-1)` without a spatial weight. Convert the field to
complex128 when contracting with the Pauli matrices; `sigma_y` contains
imaginary entries. All PyTorch tensors in an operation must share a device.

## 2. Native matrix exponential

For a constant Hamiltonian, evaluate `torch.linalg.matrix_exp(-1j*H*t/hbar)`
at each requested time. It is a matrix function, unlike elementwise `torch.exp`.
Broadcast the time and field axes to form `(T,B,2,2)` before applying it to
the initial spinors.

This route has floating-point error, but no integration time step. More output
samples give a denser trajectory; they do not improve the accuracy of a state
at a fixed time. Negative times are valid and reverse the evolution.

The NumPy comparison uses Hermitian eigendecomposition:

$$
H=V\operatorname{diag}(E)V^\dagger,\qquad
U(t)=V\operatorname{diag}(e^{-iEt/\hbar})V^\dagger.
$$

Use the conjugate transpose, not the plain transpose. Eigenvector phases and
the basis inside a degenerate eigenspace are arbitrary. Compare propagators,
states, or observables instead of individual eigenvectors. Zero field is an
explicit test case because its spectrum is degenerate.

## 3. Independent analytical references

Compare matrix-exponential states with the closed-form Pauli expression derived
in the theory document. The sinc form handles zero field without a `0/0`.
Compare Bloch vectors separately with Rodrigues' three-dimensional rotation.

Contractions such as

```python
torch.einsum("...i,aij,...j->...a", states.conj(), pauli, states).real
```

compute the three spin expectations for any leading history/batch axes.
Taking the real part is justified for Hermitian operators; test Hermiticity
separately so a construction error is not silently accepted.

For state comparisons, compute `overlap = sum(reference.conj()*state)` and
remove `exp(1j*angle(overlap))` from the relative phase. Norm errors are checked
independently. Global phase alignment does not remove a wrong relative phase
between the two spin amplitudes; that error changes spin observables.

## 4. Crank-Nicolson as an accuracy experiment

Reuse the earlier trapezoidal approximation to Schrödinger's equation:

$$
\left(I+\frac{i\,dt}{2\hbar}H\right)\psi_{n+1}
=\left(I-\frac{i\,dt}{2\hbar}H\right)\psi_n.
$$

The PyTorch implementation factors the left matrices once with batched
`torch.linalg.lu_factor`, then reuses `lu_solve`. NumPy solves once for the
complete 2-by-2 step matrix and repeatedly applies it. This is inexpensive for
one spin; it is not a reason to form large dense propagators in grid problems.
Neither implementation explicitly forms a matrix inverse.

For Hermitian `H`, the CN step is unitary and commutes with `H`. It preserves
norm and static-field energy up to roundoff, and a negative step reverses it.
Nevertheless, its phase is approximate. For energies `+/-hbar*omega/2`, its
eigenphases are `-/+2*atan(omega*dt/4)` per step. The Bloch vector rotates with
effective speed

$$
\omega_{CN}=\frac4{dt}\arctan\left(\frac{\omega dt}{4}\right)
=\omega-\frac{\omega^3dt^2}{48}+O(dt^4).
$$

At fixed final time, halving `dt` reduces the error by about four. The phase
lag grows with duration and field strength, even when norm and energy appear
perfect. The report measures both state and Bloch-vector errors.

## 5. Measurement weights and numerical limits

The measurement routine accepts a nonzero real direction and normalizes it.
It returns `(norm_squared +/- axis.dot(bloch))/2`. Thus the sum of weights
equals the input norm squared; normalization drift is exposed rather than
hidden by renormalizing output probabilities. Inputs should already be normalized.

Roundoff can produce tiny negative weights near a mathematically zero outcome.
Tests allow small absolute tolerances and do not clip values to conceal errors.
Check Pauli algebra, projector probabilities, zero-field identity, signed
precession, eigenstate phases, and the `2*pi`/`4*pi` distinction.

There is no spatial-grid or basis convergence study: dimension two is exact for
this model. The relevant numerical studies are finite precision and CN time
error. Sparse solvers would add complexity to a 2-by-2 problem without benefit.

## 6. Cost and model limits

Each independent field uses a 2-by-2 matrix. Storing `T` propagators for `B`
fields costs `O(T*B)` memory with a small constant. CN needs `O(B)` working
memory and `O(S*B)` for `S` snapshots, saving the initial and final states even
when `store_every` does not divide the number of steps.

Use `evolve_constant` for this chapter's physical predictions. CN is retained
to understand integration error. Do not use the constant-field exponential
formula unchanged when the direction of the field varies in time: Hamiltonians
at different times need not commute.
