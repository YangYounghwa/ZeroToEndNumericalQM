# Chapter 6: Radial Hydrogen Equation

This final stationary-one-particle chapter solves the Coulomb problem after
separating its angular variables. It introduces the reduced radial
wavefunction, the singular origin, angular-momentum sectors, and sparse
eigenvalue calculations.

## Learning goals

- Derive the one-dimensional equation for the reduced radial function `u(r)`.
- Enforce the physical condition `u(0) = 0` without sampling the singularity.
- Include the angular-momentum-dependent centrifugal potential.
- Build a sparse tridiagonal Hamiltonian and request only low-energy states.
- Verify hydrogenic energies, radial normalization, and `<r>`.
- Separate radial-spacing error from outer-boundary error.
- Compare NumPy/SciPy sparse results with PyTorch dense and batched results.

## Reading order

1. [Quantum theory](theory.md)
2. [Numerical method](numerical_method.md)
3. [Coding hints](coding_hints.md)
4. `boilerplates/radial_hydrogen_numpy_starter.py`
5. `boilerplates/radial_hydrogen_torch_starter.py`
6. `boilerplates/radial_hydrogen_convergence_starter.py`
7. `solutions/radial_hydrogen_numpy_solution.py`
8. `solutions/radial_hydrogen_torch_solution.py`
9. `solutions/radial_hydrogen_convergence.py`

Editable copies of the starters are provided in `workbench/`. That directory
is ignored by Git, Ruff, and mypy.

## Commands

Run these from the project root:

```powershell
uv run python 01_stationary_one_particle/06_radial_hydrogen/solutions/radial_hydrogen_numpy_solution.py
uv run python 01_stationary_one_particle/06_radial_hydrogen/solutions/radial_hydrogen_torch_solution.py
uv run python 01_stationary_one_particle/06_radial_hydrogen/solutions/radial_hydrogen_convergence.py
uv run pytest 01_stationary_one_particle/06_radial_hydrogen
```

The examples use atomic units: `m = hbar = 1` and Coulomb potential
`V(r) = -Z/r`. Distances are measured in Bohr radii and energies in Hartree.

## Completion criteria

- The Hamiltonian is real symmetric and tridiagonal.
- The NumPy Hamiltonian uses sparse storage.
- Reduced radial functions are orthonormal under `dr` integration.
- Energies and `<r>` agree with hydrogenic analytical values.
- States with the same principal quantum number are degenerate across valid
  angular-momentum sectors.
- Grid and outer-domain convergence are demonstrated separately.
- NumPy and PyTorch agree within a defined tolerance.

## Next step

Phase 2 starts time evolution with a free Gaussian wave packet. Stationary
eigenstates are replaced by a complex state that changes in time, and norm
conservation becomes a central validation check.
