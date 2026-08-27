import harmonic_packet_numpy_solution as solution
import numpy as np
from scipy.sparse import issparse


def test_harmonic_hamiltonian_is_sparse_and_hermitian() -> None:
    grid, spacing = solution.make_grid(100)
    potential, hamiltonian = solution.build_hamiltonian(grid, spacing)
    assert potential.shape == grid.shape
    assert np.all(potential >= 0.0)
    assert issparse(hamiltonian)
    assert hamiltonian.nnz == 3 * 100 - 2
    assert np.allclose(hamiltonian.toarray(), hamiltonian.toarray().T)


def test_coherent_state_has_ground_state_width() -> None:
    grid, spacing = solution.make_grid(400)
    state = solution.normalize_wavefunction(solution.coherent_state(grid), spacing)
    density = np.abs(state) ** 2
    mean = spacing * np.sum(grid * density)
    variance = spacing * np.sum((grid - mean) ** 2 * density)
    assert np.isclose(mean, 2.0, atol=1e-12)
    assert np.isclose(np.sqrt(variance), 1.0 / np.sqrt(2.0), atol=1e-12)


def test_norm_and_energy_are_conserved() -> None:
    result = solution.solve_harmonic_packet(num_points=180, num_steps=160)
    norms = solution.probability_norms(result)
    energies = solution.energy_expectations(result)
    assert np.max(np.abs(norms - 1.0)) < 3e-13
    assert np.max(np.abs(energies - energies[0])) < 3e-12


def test_center_follows_classical_harmonic_trajectory() -> None:
    time_step = 2.0 * np.pi / 400.0
    result = solution.solve_harmonic_packet(
        num_points=400,
        time_step=time_step,
        num_steps=400,
    )
    exact = solution.analytical_center(result.times)
    assert np.max(np.abs(solution.expectation_position(result) - exact)) < 0.025


def test_coherent_packet_width_stays_nearly_constant() -> None:
    result = solution.solve_harmonic_packet(num_points=400, num_steps=300)
    widths = solution.position_width(result)
    assert np.max(widths) - np.min(widths) < 0.003


def test_packet_returns_after_one_period() -> None:
    time_step = 2.0 * np.pi / 500.0
    result = solution.solve_harmonic_packet(
        num_points=300,
        time_step=time_step,
        num_steps=500,
    )
    fidelity = solution.state_fidelity(
        result.wavefunctions[0], result.wavefunctions[-1], result.spacing
    )
    assert fidelity > 0.999


def test_matrix_exponential_and_small_step_cn_agree() -> None:
    grid, spacing = solution.make_grid(80, -8.0, 8.0)
    _, hamiltonian = solution.build_hamiltonian(grid, spacing)
    initial = solution.coherent_state(grid)
    reference = solution.matrix_exponential_state(initial, hamiltonian, spacing, 0.2)
    _, states = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.002, 100
    )
    assert solution.state_l2_error(states[-1], reference, spacing) < 4e-5


def test_forward_then_reverse_recovers_initial_state() -> None:
    grid, spacing = solution.make_grid(100, -8.0, 8.0)
    _, hamiltonian = solution.build_hamiltonian(grid, spacing)
    initial = solution.normalize_wavefunction(solution.coherent_state(grid), spacing)
    _, forward = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.01, 80
    )
    _, reverse = solution.propagate_crank_nicolson(
        forward[-1], hamiltonian, spacing, -0.01, 80
    )
    assert solution.state_l2_error(reverse[-1], initial, spacing) < 3e-13


def test_time_step_error_is_second_order() -> None:
    grid, spacing = solution.make_grid(70, -8.0, 8.0)
    _, hamiltonian = solution.build_hamiltonian(grid, spacing)
    initial = solution.coherent_state(grid)
    reference = solution.matrix_exponential_state(initial, hamiltonian, spacing, 0.4)
    _, coarse = solution.propagate_crank_nicolson(
        initial, hamiltonian, spacing, 0.04, 10
    )
    _, fine = solution.propagate_crank_nicolson(initial, hamiltonian, spacing, 0.02, 20)
    coarse_error = solution.state_l2_error(coarse[-1], reference, spacing)
    fine_error = solution.state_l2_error(fine[-1], reference, spacing)
    assert coarse_error / fine_error > 3.7
