import numpy as np
import stationary_2d_numpy_solution as numpy_solution
import stationary_2d_torch_solution as torch_solution
import torch


def test_torch_shapes_dtype_and_hermiticity() -> None:
    result = torch_solution.solve_stationary_2d(num_x=10, num_y=9, num_states=3)
    assert result.wavefunctions.shape == (9, 10, 3)
    assert result.hamiltonian.dtype == torch.float64
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mT)


def test_numpy_and_torch_two_dimensional_solvers_agree() -> None:
    numpy_result = numpy_solution.solve_stationary_2d(num_x=10, num_y=9, num_states=4)
    torch_result = torch_solution.solve_stationary_2d(num_x=10, num_y=9, num_states=4)
    assert np.allclose(numpy_result.energies, torch_result.energies.numpy(), atol=1e-11)


def test_torch_batched_potentials_match_individual_solves() -> None:
    x_grid, y_grid, dx, dy, base_potential, full_hamiltonian = (
        torch_solution.build_hamiltonian(
            torch_solution.anisotropic_oscillator_potential,
            num_x=8,
            num_y=7,
        )
    )
    kinetic = full_hamiltonian - torch.diag(base_potential.reshape(-1))
    y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
    potentials = torch.stack(
        [0.5 * (x_mesh**2 + factor * y_mesh**2) for factor in (1.0, 1.5, 2.0)]
    )
    batch = torch_solution.solve_potential_batch(kinetic, potentials, dx, dy, 3)
    for index, factor in enumerate((1.0, 1.5, 2.0)):

        def potential(
            x_values: torch.Tensor,
            y_values: torch.Tensor,
            coefficient: float = factor,
        ) -> torch.Tensor:
            return 0.5 * (x_values**2 + coefficient * y_values**2)

        individual = torch_solution.solve_stationary_2d(
            potential, num_x=8, num_y=7, num_states=3
        )
        assert torch.allclose(batch.energies[index], individual.energies, atol=1e-12)


def test_torch_orthonormality_and_residuals() -> None:
    result = torch_solution.solve_stationary_2d(num_x=10, num_y=9, num_states=4)
    vectors = result.wavefunctions.reshape(-1, 4)
    overlap = result.spacing_x * result.spacing_y * vectors.mT @ vectors
    assert torch.allclose(overlap, torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert torch.all(torch_solution.residual_norms(result) < 1e-11)
