import harmonic_packet_numpy_solution as numpy_solution
import harmonic_packet_torch_solution as torch_solution
import numpy as np
import torch


def test_torch_hamiltonian_and_states_use_expected_dtypes() -> None:
    result = torch_solution.solve_harmonic_packet(num_points=60, num_steps=4)
    assert result.grid.dtype == torch.float64
    assert result.potential.dtype == torch.float64
    assert result.hamiltonian.dtype == torch.complex128
    assert result.wavefunctions.dtype == torch.complex128
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mH)


def test_numpy_and_torch_harmonic_propagation_agree() -> None:
    numpy_result = numpy_solution.solve_harmonic_packet(num_points=80, num_steps=30)
    torch_result = torch_solution.solve_harmonic_packet(num_points=80, num_steps=30)
    assert np.allclose(
        numpy_result.wavefunctions,
        torch_result.wavefunctions.numpy(),
        atol=3e-11,
    )


def test_torch_batch_matches_individual_propagation() -> None:
    grid, spacing = torch_solution.make_grid(70, -8.0, 8.0)
    _, hamiltonian = torch_solution.build_hamiltonian(grid, spacing)
    initial_states = torch.stack(
        [
            torch_solution.coherent_state(grid, center=-2.0),
            torch_solution.coherent_state(grid, center=0.0),
            torch_solution.coherent_state(grid, center=2.0),
        ],
        dim=1,
    )
    _, batch = torch_solution.propagate_crank_nicolson_batch(
        initial_states, hamiltonian, spacing, 0.01, 20
    )
    for index in range(3):
        _, individual = torch_solution.propagate_crank_nicolson(
            initial_states[:, index], hamiltonian, spacing, 0.01, 20
        )
        assert torch.allclose(batch[:, :, index], individual, atol=1e-12)


def test_torch_center_norm_and_energy_checks() -> None:
    result = torch_solution.solve_harmonic_packet(num_points=100, num_steps=80)
    norms = torch_solution.probability_norms(result)
    energies = torch_solution.energy_expectations(result)
    centers = torch_solution.expectation_position(result)
    exact = torch_solution.analytical_center(result.times)
    assert torch.max(torch.abs(norms - 1.0)) < 3e-13
    assert torch.max(torch.abs(energies - energies[0])) < 3e-12
    assert torch.max(torch.abs(centers - exact)) < 0.08
