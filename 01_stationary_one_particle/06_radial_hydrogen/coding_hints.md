# Coding Hints

## NumPy/SciPy implementation

Recommended function order:

1. `make_radial_grid`
2. `effective_potential`
3. `build_hamiltonian`
4. `normalize_wavefunctions`
5. `solve_radial_hydrogen`
6. analytical energies and observables

Construct the sparse operator with three arrays:

```text
main[i] = hbar**2 / (mass * spacing**2) + V_eff(r[i])
off[i]  = -hbar**2 / (2 * mass * spacing**2)
```

Pass `(off, main, off)` and offsets `(-1, 0, 1)` to `scipy.sparse.diags`.
Request only `num_states` eigenpairs with `eigsh(..., which="SA")`, then sort
them explicitly.

## PyTorch implementation

- Use `torch.float64` for agreement with NumPy.
- Create every tensor directly on the selected device.
- Build one-sector matrices with `torch.diag`.
- For batched sectors, broadcast `ell[:, None]` against `grid[None, :]` and use
  `torch.diag_embed` for the potential matrices.
- Normalize along the radial-grid dimension, which is `dim=-2` for both the
  single and batched wavefunction layouts.

## Expected shapes

```text
grid:                         (N,)
effective_potential:          (N,)
energies:                     (num_states,)
radial_wavefunctions:         (N, num_states)
batched effective potentials: (B, N)
batched wavefunctions:        (B, N, num_states)
```

## Boundary conditions

Do not put `0` or `r_max` in `grid`. Their wavefunction values are already
known to be zero. With `N` interior unknowns, use

```text
spacing = r_max / (N + 1)
grid = spacing * [1, 2, ..., N]
```

## Validation tests

- The first grid point is positive.
- The sparse Hamiltonian has exactly `3N - 2` nonzero entries.
- The Hamiltonian equals its transpose.
- `spacing * U.T @ U` is the identity.
- Energies approach `-Z**2 / (2*n**2)` in atomic units.
- Numerical `<r>` approaches the analytical value.
- The `2s` and `2p` energies approach the same value.
- NumPy and PyTorch agree on the same grid.

## Common mistakes

- Normalizing with `r**2 * |u|**2`; that factor belongs with `R`, not `u`.
- Using state index zero directly as the principal quantum number. For a fixed
  `ell`, the first state has `n = ell + 1`.
- Assuming sparse eigenpairs arrive sorted.
- Comparing only the ground state when testing `r_max`; excited states expose
  domain truncation more clearly.
