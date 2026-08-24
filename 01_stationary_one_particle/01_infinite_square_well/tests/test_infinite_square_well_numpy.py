from typing import Any

import infinite_square_well_numpy_solution as solution
import numpy as np
import pytest


def test_hamiltonian_is_hermitian() -> None:
    _, _, hamiltonian = solution.build_hamiltonian(num_points=40)
    assert np.allclose(hamiltonian, hamiltonian.T.conj())


def test_lowest_energies_match_analytical_values() -> None:
    result = solution.solve_infinite_well(num_points=300, num_states=3)
    exact = solution.analytical_energies(3)
    assert np.allclose(result.energies, exact, rtol=1e-4)


def test_wavefunctions_are_discretely_orthonormal() -> None:
    result = solution.solve_infinite_well(num_points=100, num_states=4)
    overlap = result.spacing * result.wavefunctions.T @ result.wavefunctions
    assert np.allclose(overlap, np.eye(4), atol=1e-12)


def test_grid_refinement_reduces_ground_state_error() -> None:
    exact = solution.analytical_energies(1)[0]
    coarse = solution.solve_infinite_well(num_points=30, num_states=1)
    fine = solution.solve_infinite_well(num_points=60, num_states=1)
    coarse_error = abs(coarse.energies[0] - exact)
    fine_error = abs(fine.energies[0] - exact)
    assert fine_error < coarse_error / 3.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"length": 0.0}, "positive"),
        ({"mass": -1.0}, "positive"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_infinite_well(**kwargs)
