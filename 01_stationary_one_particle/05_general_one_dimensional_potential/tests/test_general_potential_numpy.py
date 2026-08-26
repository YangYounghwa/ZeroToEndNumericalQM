from collections.abc import Callable
from typing import Any, cast

import general_potential_numpy_solution as solution
import numpy as np
import pytest
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def test_grid_and_hamiltonian_have_expected_structure() -> None:
    zero_potential: Callable[[FloatArray], FloatArray] = np.zeros_like
    grid, spacing, potential, hamiltonian = solution.build_hamiltonian(
        zero_potential,
        num_points=40,
        x_min=-2.0,
        x_max=3.0,
    )
    assert grid.shape == (40,)
    assert grid[0] > -2.0
    assert grid[-1] < 3.0
    assert np.isclose(spacing, 5.0 / 41.0)
    assert np.all(potential == 0.0)
    assert np.allclose(hamiltonian, hamiltonian.T.conj())


def test_shifted_harmonic_oscillator_matches_exact_reference() -> None:
    result = solution.solve_stationary(
        solution.shifted_harmonic_potential,
        num_points=220,
        num_states=4,
    )
    exact = solution.analytical_harmonic_energies(4)
    assert np.allclose(result.energies, exact, atol=1.2e-2)
    assert np.allclose(solution.expectation_position(result), 0.75, atol=2e-4)


def test_wavefunctions_are_orthonormal_and_have_small_residuals() -> None:
    result = solution.solve_stationary(
        solution.shifted_harmonic_potential,
        num_points=140,
        num_states=5,
    )
    overlap = result.spacing * result.wavefunctions.T @ result.wavefunctions
    assert np.allclose(overlap, np.eye(5), atol=1e-12)
    assert np.all(solution.residual_norms(result) < 1e-11)


def test_adding_constant_to_potential_shifts_energies_only() -> None:
    def base_potential(grid: FloatArray) -> FloatArray:
        return 0.5 * grid**2 + 0.05 * grid**4

    def shifted_potential(grid: FloatArray) -> FloatArray:
        return base_potential(grid) + 2.75

    base = solution.solve_stationary(base_potential, num_points=120, num_states=4)
    shifted = solution.solve_stationary(
        shifted_potential,
        num_points=120,
        num_states=4,
    )
    assert np.allclose(shifted.energies - base.energies, 2.75, atol=1e-12)
    assert np.allclose(
        np.abs(shifted.wavefunctions),
        np.abs(base.wavefunctions),
        atol=1e-11,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 1}, "num_points"),
        ({"num_states": 0}, "num_states"),
        ({"x_min": 1.0, "x_max": 1.0}, "x_min"),
        ({"mass": 0.0}, "positive"),
    ],
)
def test_invalid_solver_inputs_raise_value_error(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        solution.solve_stationary(np.zeros_like, **kwargs)


def test_invalid_potential_output_is_rejected() -> None:
    grid = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    with pytest.raises(ValueError, match="one value"):
        solution.evaluate_potential(grid, lambda _: np.zeros(2))
    complex_potential = cast(
        solution.PotentialFunction,
        lambda x: 1j * x,
    )
    with pytest.raises(ValueError, match="real"):
        solution.evaluate_potential(grid, complex_potential)
    with pytest.raises(ValueError, match="finite"):
        solution.evaluate_potential(grid, lambda x: np.full_like(x, np.inf))
