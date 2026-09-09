# Numerical Method: Barrier Scattering

## 1. Interior grid and sampled barrier

For walls at ±L and N interior values,

$$h=\frac{2L}{N+1},\qquad x_i=-L+(i+1)h,\quad i=0,\ldots,N-1.$$

The omitted endpoint amplitudes are zero. They produce reflecting Dirichlet
walls, not absorbing boundaries. Sample V0 where |x|<a/2 and zero elsewhere.

The centered second difference gives

$$H_{ii}=\frac{\hbar^2}{mh^2}+V_i,\qquad
H_{i,i\pm1}=-\frac{\hbar^2}{2mh^2}.$$

PyTorch builds a small real float64 dense matrix using diagonal tensors. The
NumPy/SciPy comparison stores only the three diagonals in CSR form. Both solve
the same discrete problem; conversion to a dense array is unnecessary in the
sparse implementation.

## 2. Normalize and evolve columns

Normalize each state with h*sum(abs(psi)^2)=1. A batch of B incident momenta has
shape (N, B), with one shared Hamiltonian. Different barrier heights require
different Hamiltonians and factorizations; this API batches initial states.

Average the Schrödinger equation between two adjacent time levels:

$$
\frac{\psi^{n+1}-\psi^n}{\Delta t}
=-\frac{iH}{2\hbar}(\psi^{n+1}+\psi^n).
$$

Thus

$$A\psi^{n+1}=B\psi^n,\qquad
A=I+\frac{i\Delta t}{2\hbar}H,\quad
B=I-\frac{i\Delta t}{2\hbar}H.$$

Factor A once. Use `torch.linalg.lu_solve` for each right-hand side, or SciPy's
sparse LU solve in the comparison. Do not calculate an explicit inverse. Only
the initial state is normalized: renormalizing every step could hide errors.

For an eigenvalue E, the update factor is
(1-i*dt*E/(2*hbar))/(1+i*dt*E/(2*hbar)). Its magnitude is one, but its phase differs
from exp(-i*dt*E/hbar). CN is unitary for Hermitian H and globally second order
in time. High-energy components can require smaller time steps to reach the
second-order regime even when norm conservation looks perfect.

Store the initial state, each requested snapshot, and the final state. The
time array records actual step indices, including a final partial save interval.
Saving fewer snapshots reduces storage but does not change propagation accuracy.
It can miss short boundary or collision events in the diagnostics.

## 3. Probability diagnostics

Create disjoint masks for left, near, and right, with their union covering every
interior grid point. Multiply each density sum by h. Their sum must equal the
norm; the near probability must not be silently assigned to transmission.

Boundary-strip probability is a separate diagnostic that overlaps the left and
right regions. Never add it as a fourth region in the conservation sum.
Track its history and compare calculations on larger domains.

Energy uses the full kinetic-plus-barrier Hamiltonian:

$$E_h(t)=h\operatorname{Re}[\psi(t)^\dagger H\psi(t)].$$

## 4. Local current check

The finite-difference Hamiltonian has the link current

$$j_{i+1/2}=\frac{\hbar}{mh}\operatorname{Im}(\psi_i^*\psi_{i+1}).$$

At the two omitted zero-amplitude endpoints, the boundary link currents vanish.
For a single CN step, evaluate the current at the midpoint state
psi_mid=(psi_next+psi_previous)/2. The update then satisfies

$$
\frac{|\psi_i^{n+1}|^2-|\psi_i^n|^2}{\Delta t}
+\frac{j_{i+1/2}^{\rm mid}-j_{i-1/2}^{\rm mid}}h=0
$$

up to solve and floating-point error. This follows by applying
|a|^2-|b|^2=2*Re(conj((a+b)/2)*(a-b)) to the CN equation.
It checks spatial indexing and current direction, beyond total norm alone.
Use adjacent integration steps for this test; averaging two widely separated
saved snapshots is not the CN midpoint state.

## 5. Separate the errors

| Study | Hold fixed | Change | Reference |
| --- | --- | --- | --- |
| Time step | H and final time | dt | PyTorch matrix exponential, checked against SciPy exponential action |
| Grid | Walls, physical barrier, packet, final time, small dt | N and h | Packet-averaged continuum transmission |
| Measurement time | Grid, potential, packet, dt | Observation time | Plateau of outgoing probabilities and small near probability |
| Domain | h, physical barrier, packet, dt, final time | L and N together | Common-grid state in a larger box |
| Momentum quadrature | Physical packet and barrier | Number of k samples | Finer integration |

The continuum transmission comparison includes all remaining finite-grid,
finite-time, finite-domain, and time errors. It is not a pure grid-error
measurement unless the other contributions are already sufficiently small.

### Rectangular-edge alignment

A discontinuous barrier can change its represented width when the grid is
refined. The published sequence uses L=40, a=2 and N=439, 839, 1639. Then
h=2/11, 2/21, 2/41. Both physical edges ±1 stay halfway between grid points,
and the number of nonzero potential samples times h stays equal to a.

Do not demand monotonic or second-order convergence from arbitrary unaligned
grids. Even an aligned interface needs a measured convergence study because the
potential is discontinuous. Free finite-difference dispersion is also altered:

$$E_h(k)=\frac{\hbar^2}{mh^2}(1-\cos kh),\qquad
v_h(k)=\frac{\hbar}{mh}\sin kh.$$

These approach the continuum energy and velocity as h decreases. Both the
incident wave and its reflected/transmitted components need adequate resolution.

### Domain and observation window

The report compares walls ±20, ±30, ±40 at the same h against ±50. A packet can
hit the left wall while transmission on the right still appears stable.
Compare phase-aligned states on the common grid, not only right probability.
Do not renormalize the cropped larger-domain reference: missing tail probability
is part of the error being diagnosed.

## 6. Analytical-reference implementation

Evaluate T(E) with separate masks below and at/above V0. Above the threshold,
use the sinc form to avoid division by E-V0 at E=V0. PyTorch uses the normalized
sinc convention sin(pi*x)/(pi*x), so pass q*a/pi.

Below the threshold, compute log(sinh(kappa*a)) as
kappa*a + log(-expm1(-2*kappa*a)) - log(2). This avoids overflow for thick
barriers. Handle zero energy and zero height/width explicitly.

Integrate T(k)*f(k) on [0, k0+8*Delta_k] with `torch.trapezoid` and check
quadrature refinement. The omitted positive high-k Gaussian tail is tiny;
the negative-k weight is deliberately not normalized away.

## 7. Cost and limits

Dense PyTorch matrix storage and each one-column solve cost O(N^2); the initial
factorization costs O(N^3). More columns share the factorization. Sparse 1D
tridiagonal storage and LU work grow linearly with N for this fixed-bandwidth
operator. Snapshot storage costs O(saved_times*N*B).

CPU float64/complex128 is the tested baseline. CUDA is optional, not a speed
promise for these small matrices. No FFT, absorber, time-dependent barrier, or
non-Hermitian propagation is implemented here.
