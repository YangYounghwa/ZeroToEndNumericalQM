import free_gaussian_numpy_solution as numpy_solution
import free_gaussian_torch_solution as torch_solution
import numpy as np
import torch


def test_torch_hamiltonian_and_states_use_complex128() -> None:
    result = torch_solution.solve_free_packet(num_points=60, num_steps=4)
    assert result.grid.dtype == torch.float64
    assert result.hamiltonian.dtype == torch.complex128
    assert result.wavefunctions.dtype == torch.complex128
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mH)


def test_numpy_and_torch_free_propagation_agree() -> None:
    numpy_result = numpy_solution.solve_free_packet(num_points=80, num_steps=30)
    torch_result = torch_solution.solve_free_packet(num_points=80, num_steps=30)
    assert np.allclose(
        numpy_result.wavefunctions,
        torch_result.wavefunctions.numpy(),
        atol=2e-11,
    )


def test_torch_batch_matches_individual_propagation() -> None:
    grid, spacing = torch_solution.make_grid(70, -12.0, 12.0)
    hamiltonian = torch_solution.build_free_hamiltonian(70, spacing)
    initial_states = torch.stack(
        [
            torch_solution.gaussian_wave_packet(grid, wave_number=1.0),
            torch_solution.gaussian_wave_packet(grid, wave_number=2.0),
            torch_solution.gaussian_wave_packet(grid, wave_number=3.0),
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


def test_torch_norm_and_energy_are_conserved() -> None:
    result = torch_solution.solve_free_packet(num_points=80, num_steps=40)
    norms = torch_solution.probability_norms(result)
    energies = torch_solution.energy_expectations(result)
    assert torch.max(torch.abs(norms - 1.0)) < 2e-13
    assert torch.max(torch.abs(energies - energies[0])) < 2e-12
