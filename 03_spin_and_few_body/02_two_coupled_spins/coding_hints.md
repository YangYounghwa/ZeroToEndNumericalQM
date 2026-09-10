# Coding hints

## Implementation order

1. Write the ordered product basis explicitly and build two spinor products.
2. Build `sigma_a tensor I` and `I tensor sigma_a`. Apply each to `|+->` to
   verify that the intended site changes.
3. Build the exchange matrix and compare with its explicit form in the theory.
4. Add each site's field, preserving the sign and units of the previous chapter.
5. Check the zero-field singlet/triplet energies and eigenpair residuals.
6. Evolve joint states with native matrix exponentials, then compare NumPy.
7. Calculate local means, all nine pair correlations, and joint probabilities.
8. Add the determinant product test and exact longitudinal-field reference.
9. Complete the CN time-error exercise and validation report.

Keep float64 parameters and complex128 operators. `sigma_y` is essential to
isotropic exchange: omitting it changes both the spectrum and swap dynamics.

## Useful checks

- `sigma1_x @ |+->` gives `|-->`; `sigma2_x @ |+->` gives `|++>`.
- Different-site operators commute; same-site Pauli operators generally do not.
- At `J=0`, a product of independent spin evolutions matches the joint evolution.
- At zero field and `J=1`, the spectrum is `[-0.75,0.25,0.25,0.25]`.
- Starting from `|+->`, probability reaches `|-+>` completely at `t=pi` for
  `J=hbar=1`. At `t=pi/2` the coefficient determinant is one half.
- Uniform fields preserve total spin squared; unequal fields need not.
- Halving a converged CN time step reduces state error by about four.

## Exercises

Change the exchange sign and identify the new lowest-energy sector. Add a
common longitudinal field, then a difference between the two fields. Compare
the resulting transfer peak with the analytical block solution.

Create a batch with different `J` values and field pairs. Compare individual
runs and verify that the batch axis is not accidentally included in a Kronecker
product. Each batch member needs four amplitudes.

Evaluate `C_zz` and its connected part throughout a swap. Explain how both local
Bloch vectors can vanish while a joint pure state still has strong correlations.
Try a transverse field at only one site and check which commutators cease to
vanish before asserting conservation laws.

Copy starters to a subject folder under `workbench/`; add imports for your own
implementations in the convergence exercise. The supplied starters preserve
public signatures and TODOs without copying the completed function bodies.
