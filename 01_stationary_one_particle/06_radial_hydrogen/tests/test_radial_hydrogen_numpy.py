from typing import Any

import numpy as np
import pytest
import radial_hydrogen_numpy_solution as solution
from scipy.sparse import issparse


def test_radial_grid_excludes_singular_origin_and_outer_boundary() -> None:
    grid, spacing = solution.make_radial_grid(99, r_max=20.0)
    assert grid.shape == (99,)
    assert np.isclose(spacing, 0.2)
    assert np.isclose(grid[0], spacing)
    assert np.isclose(grid[-1], 20.0 - spacing)
    assert np.all(grid > 0.0)


def test_hamiltonian_is_sparse_tridiagonal_and_hermitian() -> None:
    _, _, _, hamiltonian = solution.build_hamiltonian(num_points=80)
    assert issparse(hamiltonian)
    assert hamiltonian.nnz == 3 * 80 - 2
    assert np.allclose(hamiltonian.toarray(), hamiltonian.toarray().T)


def test_s_state_energies_and_radii_match_hydrogen_reference() -> None:
    result = solution.solve_radial_hydrogen(num_points=800, num_states=3)
    assert np.allclose(
        result.energies,
        solution.analytical_energies(3),
        atol=3.5e-4,
    )
    assert np.allclose(
        solution.expectation_radius(result),
        solution.analytical_expectation_radius(3),
        atol=3.0e-2,
    )


def test_radial_states_are_orthonormal_and_have_small_residuals() -> None:
    result = solution.solve_radial_hydrogen(num_points=300, num_states=4)
    overlap = (
        result.spacing * result.radial_wavefunctions.T @ result.radial_wavefunctions
    )
    assert np.allclose(overlap, np.eye(4), atol=1e-12)
    assert np.all(solution.residual_norms(result) < 1e-10)


def test_coulomb_energy_depends_on_n_not_angular_momentum() -> None:
    s_sector = solution.solve_radial_hydrogen(
        num_points=600, num_states=2, angular_momentum=0
    )
    p_sector = solution.solve_radial_hydrogen(
        num_points=600, num_states=1, angular_momentum=1
    )
    assert np.isclose(s_sector.energies[1], -0.125, atol=1e-4)
    assert np.isclose(p_sector.energies[0], -0.125, atol=1e-4)
    assert np.isclose(s_sector.energies[1], p_sector.energies[0], atol=1e-4)


def test_nuclear_charge_scaling() -> None:
    hydrogen = solution.solve_radial_hydrogen(
        num_points=600, num_states=1, r_max=30.0, nuclear_charge=1.0
    )
    ion = solution.solve_radial_hydrogen(
        num_points=600, num_states=1, r_max=15.0, nuclear_charge=2.0
    )
    assert np.isclose(ion.energies[0] / hydrogen.energies[0], 4.0, atol=2e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 2}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"r_max": 0.0}, "r_max"),
        ({"angular_momentum": -1}, "angular_momentum"),
        ({"angular_momentum": 0.5}, "angular_momentum"),
        ({"nuclear_charge": 0.0}, "positive"),
    ],
)
def test_invalid_solver_inputs_raise_value_error(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_radial_hydrogen(**kwargs)
