from typing import Any

import finite_square_well_numpy_solution as solution
import numpy as np
import pytest


def test_potential_and_hamiltonian_are_correctly_structured() -> None:
    grid, _, potential, hamiltonian = solution.build_hamiltonian(num_points=80)
    assert np.all(potential[np.abs(grid) < 1.0] == -20.0)
    assert np.all(potential[np.abs(grid) >= 1.0] == 0.0)
    assert np.allclose(hamiltonian, hamiltonian.T.conj())


def test_bound_energies_match_matching_equations() -> None:
    # N=599 places x=+/-1 halfway between grid points, representing the step
    # symmetrically instead of assigning a boundary point to either side.
    result = solution.solve_finite_square_well(num_points=599, num_states=8)
    numerical = result.energies[solution.bound_state_mask(result)]
    exact = solution.analytical_bound_energies()
    assert len(numerical) == len(exact)
    assert np.allclose(numerical, exact, atol=0.03)


def test_wavefunctions_are_discretely_orthonormal() -> None:
    result = solution.solve_finite_square_well(num_points=160, num_states=6)
    overlap = result.spacing * result.wavefunctions.T @ result.wavefunctions
    assert np.allclose(overlap, np.eye(6), atol=1e-12)


def test_states_have_alternating_parity() -> None:
    result = solution.solve_finite_square_well(num_points=160, num_states=5)
    for state in range(5):
        expected = (-1) ** state * result.wavefunctions[:, state]
        assert np.allclose(result.wavefunctions[::-1, state], expected, atol=1e-10)


def test_bound_states_are_more_localized_than_first_continuum_state() -> None:
    result = solution.solve_finite_square_well(num_points=300, num_states=8)
    probabilities = solution.probability_inside_well(result)
    bound = solution.bound_state_mask(result)
    first_continuum_probability = probabilities[np.flatnonzero(~bound)[0]]
    assert np.all(probabilities[bound] > first_continuum_probability)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"depth": 0.0}, "positive"),
        ({"half_width": 8.0}, "smaller"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_finite_square_well(**kwargs)
