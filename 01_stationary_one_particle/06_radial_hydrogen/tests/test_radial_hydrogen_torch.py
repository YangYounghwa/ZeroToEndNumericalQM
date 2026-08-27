import numpy as np
import radial_hydrogen_numpy_solution as numpy_solution
import radial_hydrogen_torch_solution as torch_solution
import torch


def test_torch_hamiltonian_is_hermitian_and_float64() -> None:
    result = torch_solution.solve_radial_hydrogen(num_points=80, num_states=3)
    assert torch.allclose(result.hamiltonian, result.hamiltonian.mH)
    assert result.grid.dtype == torch.float64
    assert result.effective_potential.dtype == torch.float64
    assert result.energies.dtype == torch.float64
    assert result.radial_wavefunctions.dtype == torch.float64


def test_numpy_and_torch_radial_solvers_agree() -> None:
    numpy_result = numpy_solution.solve_radial_hydrogen(
        num_points=160,
        num_states=3,
        angular_momentum=1,
    )
    torch_result = torch_solution.solve_radial_hydrogen(
        num_points=160,
        num_states=3,
        angular_momentum=1,
    )
    assert np.allclose(numpy_result.energies, torch_result.energies.numpy(), atol=1e-11)
    assert np.allclose(
        np.abs(numpy_result.radial_wavefunctions),
        torch.abs(torch_result.radial_wavefunctions).numpy(),
        atol=1e-9,
    )


def test_batched_angular_momentum_solver_matches_individual_solves() -> None:
    angular_momenta = torch.tensor([0, 1, 2], dtype=torch.int64)
    batch = torch_solution.solve_angular_momentum_batch(
        angular_momenta,
        num_points=100,
        num_states=2,
    )
    assert batch.energies.shape == (3, 2)
    assert batch.radial_wavefunctions.shape == (3, 100, 2)
    for index, angular_momentum in enumerate((0, 1, 2)):
        individual = torch_solution.solve_radial_hydrogen(
            num_points=100,
            num_states=2,
            angular_momentum=angular_momentum,
        )
        assert torch.allclose(batch.energies[index], individual.energies, atol=1e-12)


def test_torch_states_are_orthonormal_and_have_small_residuals() -> None:
    result = torch_solution.solve_radial_hydrogen(num_points=120, num_states=4)
    overlap = (
        result.spacing * result.radial_wavefunctions.mT @ result.radial_wavefunctions
    )
    assert torch.allclose(overlap, torch.eye(4, dtype=torch.float64), atol=1e-12)
    assert torch.all(torch_solution.residual_norms(result) < 1e-11)
