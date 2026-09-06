import free_gaussian_numpy_solution as reference
import free_gaussian_torch_solution as solution
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


@pytest.mark.parametrize("time", [-0.4, 0.0, 0.4])
def test_torch_exponential_matches_sparse_scipy(time: float) -> None:
    grid, spacing = solution.make_grid(60, -8, 8)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.gaussian_wave_packet(grid, center=-2, width=0.7, wave_number=1)
    torch_state = solution.matrix_exponential_state(initial, hamiltonian, spacing, time)
    sparse_state = reference.sparse_exponential_state(
        initial.numpy(), csr_matrix(hamiltonian.numpy()), spacing, time
    )
    np.testing.assert_allclose(torch_state.numpy(), sparse_state, atol=2e-12, rtol=0)
    if time >= 0:
        dense_state = reference.matrix_exponential_state(
            initial.numpy(), csr_matrix(hamiltonian.numpy()), spacing, time
        )
        np.testing.assert_allclose(sparse_state, dense_state, atol=2e-12, rtol=0)


def test_torch_time_error_is_second_order_despite_conserved_norm() -> None:
    grid, spacing = solution.make_grid(70, -8, 8)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    initial = solution.gaussian_wave_packet(grid, center=-2, width=0.7, wave_number=1)
    exact = solution.matrix_exponential_state(initial, hamiltonian, spacing, 0.4)
    errors = []
    for dt, steps in ((0.04, 10), (0.02, 20)):
        _, states = solution.propagate_crank_nicolson(
            initial, hamiltonian, spacing, dt, steps
        )
        assert abs((spacing * torch.sum(torch.abs(states[-1]) ** 2)).item() - 1) < 1e-12
        errors.append(solution.state_l2_error(states[-1], exact, spacing).item())
    assert 3.6 < errors[0] / errors[1] < 4.1
    assert errors[0] > 1e-5


def test_torch_spatial_error_decreases_without_time_discretization() -> None:
    errors = []
    time = 0.8
    for points in (79, 159):
        grid, spacing = solution.make_grid(points, -8, 8)
        hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
        initial = solution.gaussian_wave_packet(
            grid, center=-2, width=0.7, wave_number=1
        )
        numerical = solution.matrix_exponential_state(
            initial, hamiltonian, spacing, time
        )
        exact = solution.analytical_wavefunction(
            grid, time, center=-2, width=0.7, wave_number=1
        )
        errors.append(solution.state_l2_error(numerical, exact, spacing).item())
    assert 3.6 < errors[0] / errors[1] < 4.3


def test_torch_domain_error_decreases_at_fixed_spacing() -> None:
    errors = []
    time = 2.0
    for half_width in (3, 8):
        grid, spacing = solution.make_grid(20 * half_width - 1, -half_width, half_width)
        hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
        initial = solution.gaussian_wave_packet(
            grid, center=-2, width=0.7, wave_number=1
        )
        numerical = solution.matrix_exponential_state(
            initial, hamiltonian, spacing, time
        )
        exact = solution.analytical_wavefunction(
            grid, time, center=-2, width=0.7, wave_number=1
        )
        errors.append(solution.state_l2_error(numerical, exact, spacing).item())
    assert errors[0] > 2 * errors[1]


def test_torch_zero_initial_state_is_rejected() -> None:
    grid, spacing = solution.make_grid(20, -8, 8)
    hamiltonian = solution.build_free_hamiltonian(len(grid), spacing)
    with pytest.raises(ValueError, match="nonzero"):
        solution.propagate_crank_nicolson(
            torch.zeros(20, dtype=torch.complex128), hamiltonian, spacing, 0.01, 2
        )
