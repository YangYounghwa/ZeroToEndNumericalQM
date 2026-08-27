from typing import Any

import free_gaussian_numpy_solution as solution
import numpy as np
import pytest
from scipy.sparse import issparse


def test_grid_and_free_hamiltonian_structure() -> None:
    grid, spacing = solution.make_grid(80, -10.0, 14.0)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    assert grid.shape == (80,)
    assert np.isclose(spacing, 24.0 / 81.0)
    assert issparse(hamiltonian)
    assert hamiltonian.nnz == 3 * 80 - 2
    assert np.allclose(hamiltonian.toarray(), hamiltonian.toarray().T)


def test_initial_gaussian_has_requested_center_and_width() -> None:
    grid, spacing = solution.make_grid(500, -20.0, 20.0)
    state = solution.normalize_wavefunction(
        solution.gaussian_wave_packet(grid, center=-3.0, width=0.7), spacing
    )
    density = np.abs(state) ** 2
    mean = spacing * np.sum(grid * density)
    variance = spacing * np.sum((grid - mean) ** 2 * density)
    assert np.isclose(mean, -3.0, atol=1e-10)
    assert np.isclose(np.sqrt(variance), 0.7, atol=1e-10)


def test_crank_nicolson_conserves_norm_and_energy() -> None:
    result = solution.solve_free_packet(num_points=180, num_steps=120)
    norms = solution.probability_norms(result)
    energies = solution.energy_expectations(result)
    assert np.max(np.abs(norms - 1.0)) < 2e-13
    assert np.max(np.abs(energies - energies[0])) < 2e-12


def test_packet_center_and_width_follow_analytical_motion() -> None:
    result = solution.solve_free_packet(num_points=400, num_steps=300)
    final_time = float(result.times[-1])
    expected_center = -6.0 + 2.0 * final_time
    expected_width = 0.8 * np.sqrt(1.0 + (final_time / (2.0 * 0.8**2)) ** 2)
    assert np.isclose(
        solution.expectation_position(result)[-1], expected_center, atol=0.04
    )
    assert np.isclose(solution.position_width(result)[-1], expected_width, atol=0.03)


def test_matrix_exponential_and_small_step_cn_agree() -> None:
    grid, spacing = solution.make_grid(80, -12.0, 12.0)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.gaussian_wave_packet(grid, center=-3.0, width=0.8)
    reference = solution.matrix_exponential_state(initial, hamiltonian, spacing, 0.2)
    _, states = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.002, 100
    )
    assert solution.state_l2_error(states[-1], reference, spacing) < 2e-5


def test_forward_then_reverse_recovers_initial_state() -> None:
    grid, spacing = solution.make_grid(100, -14.0, 14.0)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.normalize_wavefunction(
        solution.gaussian_wave_packet(grid), spacing
    )
    _, forward = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.01, 60
    )
    _, reverse = solution.propagate_crank_nicolson(
        forward[-1], hamiltonian, spacing, -0.01, 60
    )
    assert solution.state_l2_error(reverse[-1], initial, spacing) < 2e-13


def test_time_step_error_is_second_order() -> None:
    grid, spacing = solution.make_grid(70, -12.0, 12.0)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.gaussian_wave_packet(grid, center=-3.0, width=0.9)
    reference = solution.matrix_exponential_state(initial, hamiltonian, spacing, 0.4)
    _, coarse = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.04, 10
    )
    _, fine = solution.propagate_crank_nicolson(initial, hamiltonian, spacing, 0.02, 20)
    coarse_error = solution.state_l2_error(coarse[-1], reference, spacing)
    fine_error = solution.state_l2_error(fine[-1], reference, spacing)
    assert coarse_error / fine_error > 3.8


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_points": 2}, "num_points"),
        ({"num_points": 20, "x_min": 1.0, "x_max": 1.0}, "x_min"),
    ],
)
def test_invalid_grid_inputs_raise_value_error(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solution.make_grid(**kwargs)
