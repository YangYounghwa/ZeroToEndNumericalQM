from typing import Any

import harmonic_oscillator_numpy_solution as solution
import numpy as np
import pytest


def test_hamiltonian_is_hermitian() -> None:
    _, _, potential, hamiltonian = solution.build_hamiltonian(num_points=40)
    assert np.all(potential >= 0.0)
    assert np.allclose(hamiltonian, hamiltonian.T.conj())


def test_lowest_energies_match_analytical_values() -> None:
    result = solution.solve_harmonic_oscillator(num_points=300, num_states=4)
    exact = solution.analytical_energies(4)
    assert np.allclose(result.energies, exact, rtol=2e-3)


def test_wavefunctions_are_discretely_orthonormal() -> None:
    result = solution.solve_harmonic_oscillator(num_points=120, num_states=4)
    overlap = result.spacing * result.wavefunctions.T @ result.wavefunctions
    assert np.allclose(overlap, np.eye(4), atol=1e-12)


def test_states_have_alternating_parity() -> None:
    result = solution.solve_harmonic_oscillator(num_points=120, num_states=4)
    for state in range(4):
        expected = (-1) ** state * result.wavefunctions[:, state]
        assert np.allclose(result.wavefunctions[::-1, state], expected, atol=1e-11)


def test_position_expectation_values_match_exact_results() -> None:
    result = solution.solve_harmonic_oscillator(num_points=300, num_states=3)
    expected_x_squared = np.arange(3, dtype=np.float64) + 0.5
    assert np.allclose(solution.expectation_x_power(result, 1), 0.0, atol=1e-12)
    assert np.allclose(
        solution.expectation_x_power(result, 2), expected_x_squared, rtol=2e-3
    )


def test_grid_refinement_reduces_ground_state_error() -> None:
    exact = solution.analytical_energies(1)[0]
    coarse = solution.solve_harmonic_oscillator(num_points=60, num_states=1)
    fine = solution.solve_harmonic_oscillator(num_points=120, num_states=1)
    coarse_error = abs(coarse.energies[0] - exact)
    fine_error = abs(fine.energies[0] - exact)
    assert fine_error < coarse_error / 3.5


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"x_max": 0.0}, "positive"),
        ({"mass": -1.0}, "positive"),
        ({"omega": 0.0}, "positive"),
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_harmonic_oscillator(**kwargs)
