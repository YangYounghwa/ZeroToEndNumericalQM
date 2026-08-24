import finite_square_well_numpy_solution as numpy_solution
import finite_square_well_torch_solution as torch_solution
import numpy as np
import torch


def test_torch_hamiltonian_is_hermitian() -> None:
    _, _, _, hamiltonian = torch_solution.build_hamiltonian(num_points=40)
    assert torch.allclose(hamiltonian, hamiltonian.mH)


def test_torch_wavefunctions_are_discretely_orthonormal() -> None:
    result = torch_solution.solve_finite_square_well(num_points=100, num_states=5)
    overlap = result.spacing * result.wavefunctions.mT @ result.wavefunctions
    assert torch.allclose(overlap, torch.eye(5, dtype=torch.float64), atol=1e-12)


def test_numpy_and_torch_results_agree() -> None:
    numpy_result = numpy_solution.solve_finite_square_well(num_points=100, num_states=6)
    torch_result = torch_solution.solve_finite_square_well(num_points=100, num_states=6)
    assert np.allclose(numpy_result.energies, torch_result.energies.numpy(), atol=1e-11)
    numpy_probability = numpy_solution.probability_inside_well(numpy_result)
    torch_probability = torch_solution.probability_inside_well(torch_result)
    assert np.allclose(numpy_probability, torch_probability.numpy(), atol=1e-11)


def test_torch_uses_float64() -> None:
    result = torch_solution.solve_finite_square_well(num_points=30, num_states=2)
    assert result.grid.dtype == torch.float64
    assert result.energies.dtype == torch.float64
    assert result.wavefunctions.dtype == torch.float64
