# Numerical method

1. Build four complex128 annihilation matrices of shape `(16,16)`.
2. Form number operators and Hermitian hopping products.
3. Add the interaction and site bias.
4. Select rows and columns with masks `(3,9,6,12)`.
5. Use `torch.linalg.eigh` for energies and normalized eigenvectors.
6. Use `torch.linalg.matrix_exp` for evolution at each requested time.
7. Compare with the explicit NumPy matrix and spectral evolution.

Use ordinary vector normalization. There is no grid weight.
State arrays have shape `(...,4)`. Time arrays have shape `(T,)`.
Evolution returns `(T,...,4)`. The last axis always labels the sector basis.
PyTorch accepts a device argument for construction. Outputs remain on that
device. Use float64 times and complex128 states.

The full Fock construction exposes fermion signs. It is a teaching method,
not a scalable many-body solver. For dimension D, dense storage costs O(D^2)
and eigendecomposition costs O(D^3). Here D is 16 before projection and 4
after projection. A many-site version needs a directly constructed sector and
sparse methods. Do not silently extend this full-space method to long chains.

All four sector states are included. There is no basis truncation error.
Direct exponentials do not use a time-step integrator. Adding output times
does not improve their accuracy. The experiment instead measures the error
of the large-U spin approximation against the exact gap. Increasing U changes
the physical model and the approximation parameter t/U.

Check the full anticommutator algebra before projection. Check the symmetric
spectrum, the atomic limit t=0, and the noninteracting limit U=0. Compare biased
matrices between implementations. Check eigenpair residuals, double occupancy,
norm, energy, total particle count, and reversed evolution. Compare states
only in the same basis and phase convention.

At degeneracies, compare spectra or invariant subspaces. Do not require
individual eigenvectors to match. Reject nonfinite parameters, unnormalized
states, invalid shapes, and non-Hermitian evolution matrices.
