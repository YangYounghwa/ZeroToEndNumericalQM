import harmonic_oscillator_numpy_solution
import harmonic_oscillator_torch_solution
import numpy as np
import torch


def test_torch_hamiltonian_is_hermitian() -> None:
    _, _, potential, hamiltonian = harmonic_oscillator_torch_solution.build_hamiltonian(
        num_points=40
    )
    assert torch.all(potential >= 0.0)
    assert torch.allclose(hamiltonian, hamiltonian.mH)


def test_torch_wavefunctions_are_discretely_orthonormal() -> None:
    result = harmonic_oscillator_torch_solution.solve_harmonic_oscillator(
        num_points=100, num_states=4
    )
    overlap = result.spacing * result.wavefunctions.mT @ result.wavefunctions
    assert torch.allclose(overlap, torch.eye(4, dtype=torch.float64), atol=1e-12)


def test_numpy_and_torch_results_agree() -> None:
    numpy_result = harmonic_oscillator_numpy_solution.solve_harmonic_oscillator(
        num_points=80, num_states=4
    )
    torch_result = harmonic_oscillator_torch_solution.solve_harmonic_oscillator(
        num_points=80, num_states=4
    )
    assert np.allclose(numpy_result.energies, torch_result.energies.numpy(), atol=1e-11)
    numpy_x2 = harmonic_oscillator_numpy_solution.expectation_x_power(numpy_result, 2)
    torch_x2 = harmonic_oscillator_torch_solution.expectation_x_power(torch_result, 2)
    assert np.allclose(numpy_x2, torch_x2.numpy(), atol=1e-11)


def test_torch_uses_float64() -> None:
    result = harmonic_oscillator_torch_solution.solve_harmonic_oscillator(
        num_points=20, num_states=2
    )
    assert result.grid.dtype == torch.float64
    assert result.potential.dtype == torch.float64
    assert result.energies.dtype == torch.float64
    assert result.wavefunctions.dtype == torch.float64
