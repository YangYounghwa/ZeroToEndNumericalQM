import general_potential_numpy_solution as numpy_solution
import general_potential_torch_solution as torch_solution
import numpy as np
import torch


def test_torch_hamiltonian_is_hermitian_and_float64() -> None:
    result = torch_solution.solve_stationary(
        torch_solution.shifted_harmonic_potential,
        num_points=60,
        num_states=3,
    )
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mH)
    assert result.grid.dtype == torch.float64
    assert result.potential.dtype == torch.float64
    assert result.energies.dtype == torch.float64
    assert result.wavefunctions.dtype == torch.float64


def test_numpy_and_torch_general_solvers_agree() -> None:
    numpy_result = numpy_solution.solve_stationary(
        numpy_solution.shifted_harmonic_potential,
        num_points=100,
        num_states=4,
    )
    torch_result = torch_solution.solve_stationary(
        torch_solution.shifted_harmonic_potential,
        num_points=100,
        num_states=4,
    )
    assert np.allclose(
        numpy_result.energies,
        torch_result.energies.numpy(),
        atol=1e-11,
    )
    assert np.allclose(
        np.abs(numpy_result.wavefunctions),
        torch.abs(torch_result.wavefunctions).numpy(),
        atol=1e-10,
    )


def test_batched_solver_matches_individual_torch_solves() -> None:
    grid, spacing = torch_solution.make_grid(80, -7.0, 7.0)
    centers = torch.tensor([-0.8, 0.0, 0.9], dtype=torch.float64)[:, None]
    potentials = 0.5 * (grid[None, :] - centers) ** 2
    batch = torch_solution.solve_potential_batch(
        grid,
        spacing,
        potentials,
        num_states=3,
    )
    assert batch.energies.shape == (3, 3)
    assert batch.wavefunctions.shape == (3, 80, 3)
    for index, center in enumerate(centers[:, 0]):
        center_value = float(center.item())

        def potential(x: torch.Tensor, value: float = center_value) -> torch.Tensor:
            return 0.5 * (x - value) ** 2

        individual = torch_solution.solve_stationary(
            potential,
            num_points=80,
            num_states=3,
            x_min=-7.0,
            x_max=7.0,
        )
        assert torch.allclose(batch.energies[index], individual.energies, atol=1e-12)


def test_torch_wavefunctions_are_orthonormal_and_have_small_residuals() -> None:
    result = torch_solution.solve_stationary(
        torch_solution.shifted_harmonic_potential,
        num_points=100,
        num_states=4,
    )
    overlap = result.spacing * result.wavefunctions.mT @ result.wavefunctions
    assert torch.allclose(overlap, torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert torch.all(torch_solution.residual_norms(result) < 1e-11)
