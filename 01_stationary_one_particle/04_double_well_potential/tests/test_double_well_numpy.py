from typing import Any

import double_well_numpy_solution as solution
import numpy as np
import pytest


def test_potential_has_expected_landmarks_and_hamiltonian_is_hermitian() -> None:
    points = np.array([-1.5, 0.0, 1.5], dtype=np.float64)
    potential = solution.double_well_potential(points)
    assert np.allclose(potential, [0.0, 8.0, 0.0])
    _, _, _, hamiltonian = solution.build_hamiltonian(num_points=50)
    assert np.allclose(hamiltonian, hamiltonian.T.conj())


def test_wavefunctions_are_discretely_orthonormal() -> None:
    result = solution.solve_double_well(num_points=160, num_states=5)
    overlap = result.spacing * result.wavefunctions.T @ result.wavefunctions
    assert np.allclose(overlap, np.eye(5), atol=1e-12)


def test_states_have_alternating_parity() -> None:
    result = solution.solve_double_well(num_points=180, num_states=4)
    for state in range(4):
        expected = (-1) ** state * result.wavefunctions[:, state]
        assert np.allclose(result.wavefunctions[::-1, state], expected, atol=1e-9)


def test_stationary_states_have_balanced_left_probability() -> None:
    result = solution.solve_double_well(num_points=200, num_states=4)
    assert np.allclose(solution.probability_left(result), 0.5, atol=1e-12)


def test_tunneling_splitting_is_positive_and_decreases_with_barrier() -> None:
    low = solution.solve_double_well(num_points=220, num_states=2, barrier_height=4.0)
    high = solution.solve_double_well(num_points=220, num_states=2, barrier_height=12.0)
    assert 0.0 < solution.tunneling_splitting(high) < solution.tunneling_splitting(low)


def test_localized_pair_is_normalized_and_separated() -> None:
    result = solution.solve_double_well(num_points=220, num_states=2)
    left, right = solution.localized_pair(result)
    assert np.isclose(result.spacing * np.vdot(left, left), 1.0)
    assert np.isclose(result.spacing * np.vdot(right, right), 1.0)
    assert result.spacing * np.sum(np.abs(left[result.grid < 0.0]) ** 2) > 0.9


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"barrier_height": 0.0}, "positive"),
        ({"separation": 6.0}, "smaller"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_double_well(**kwargs)
