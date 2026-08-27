# Numerical Method

## 1. Spatial discretization

Use `N` interior points on `[x_min, x_max]` with spacing

$$
h=\frac{x_{\max}-x_{\min}}{N+1}.
$$

The second-order kinetic Hamiltonian is tridiagonal:

$$
H_{ii}=\frac{\hbar^2}{mh^2},
\qquad
H_{i,i\pm1}=-\frac{\hbar^2}{2mh^2}.
$$

## 2. Crank-Nicolson derivation

Write the Schrödinger equation as

$$
\frac{d\psi}{dt}=-\frac{i}{\hbar}H\psi.
$$

Average the right-hand side between time levels `n` and `n+1`:

$$
\frac{\psi^{n+1}-\psi^n}{\Delta t}
=-\frac{i}{2\hbar}H(\psi^{n+1}+\psi^n).
$$

Rearranging gives

$$
\boxed{
\left(I+\frac{i\Delta t}{2\hbar}H\right)\psi^{n+1}
=\left(I-\frac{i\Delta t}{2\hbar}H\right)\psi^n
}.
$$

Define `A` as the left matrix and `B` as the right matrix. For a fixed
Hamiltonian and time step, factorize `A` once and solve

```text
A psi_next = B psi_current
```

at every step.

## 3. Accuracy and unitarity

The one-step propagator is

$$
U_{CN}=\left(I+\frac{i\Delta tH}{2\hbar}\right)^{-1}
\left(I-\frac{i\Delta tH}{2\hbar}\right).
$$

For Hermitian `H`, this Cayley transform is unitary up to linear-solver and
floating-point error. It is second-order accurate in time globally:

$$
\|\psi_{CN}(T)-\psi(T)\|=O(\Delta t^2).
$$

Unconditional stability means the method does not blow up for large `dt`; it
does not mean a large `dt` is accurate.

## 4. Matrix-exponential reference

For a small dense Hamiltonian, calculate

$$
U(T)=\exp(-iHT/\hbar)
$$

directly. This is expensive: dense storage costs `O(N^2)` and a dense matrix
exponential costs approximately `O(N^3)`. It is used only as a reference for
time-step convergence, not as the main large-grid algorithm.

## 5. Observables

For each stored state,

$$
P(t)\approx h\sum_i|\psi_i(t)|^2,
$$

$$
\langle x\rangle(t)\approx h\sum_i x_i|\psi_i(t)|^2,
$$

and

$$
\langle H\rangle(t)\approx
h\,\psi(t)^\dagger H\psi(t).
$$

Compare wavefunctions only after accounting for an irrelevant global phase.
The provided state error aligns that phase using the discrete overlap.

## 6. Error sources and checks

- Spatial finite differences contribute `O(h^2)` error.
- Crank-Nicolson contributes `O(dt^2)` global time error.
- A finite domain produces artificial reflections.
- Storing every state costs `O(num_steps * N)` memory.
- A conserved norm alone does not prove accuracy because Crank-Nicolson remains
  unitary even with an overly large time step.

Required checks are Hermiticity, norm and energy conservation, agreement with
analytical center and width, second-order time convergence, and recovery of the
initial state after forward and reverse propagation.
