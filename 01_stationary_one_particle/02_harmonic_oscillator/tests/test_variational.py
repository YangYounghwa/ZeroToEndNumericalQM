from itertools import pairwise

import harmonic_oscillator_variational as solution
import numpy as np
import torch


def test_autograd_matches_analytical_gaussian_derivative() -> None:
    log_alpha = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
    energy = solution.gaussian_energy(log_alpha, coupling=0.2)
    (gradient,) = torch.autograd.grad(energy, log_alpha)
    alpha = log_alpha.exp()
    expected = (alpha - alpha.reciprocal()) / 4 - 3 * 0.2 / (2 * alpha**2)
    assert torch.allclose(gradient, expected, atol=1e-14, rtol=0)


def test_gaussian_optimizer_recovers_exact_harmonic_ground_state() -> None:
    result = solution.optimize_gaussian()
    assert abs(result.alpha - 1) < 2e-6
    assert abs(result.energy - 0.5) < 1e-11


def test_basis_matrix_uses_continuum_quartic_matrix_elements_at_cutoff() -> None:
    # <n|x^4|n> = 3(2n² + 2n + 1)/4, including the last retained state.
    size, coupling = 5, 0.1
    hamiltonian = solution.oscillator_basis_hamiltonian(size, coupling)
    levels = torch.arange(size, dtype=torch.float64)
    expected = levels + 0.5 + coupling * 3 * (2 * levels**2 + 2 * levels + 1) / 4
    assert torch.allclose(hamiltonian.diag(), expected, atol=1e-14, rtol=0)


def test_basis_energies_converge_from_above_and_agree_with_numpy() -> None:
    energies = []
    for size in (4, 8, 12, 20):
        spectrum = torch.linalg.eigvalsh(solution.oscillator_basis_hamiltonian(size))
        np.testing.assert_allclose(
            spectrum.numpy(), solution.numpy_basis_energies(size), atol=1e-12, rtol=0
        )
        energies.append(spectrum[0].item())
    assert all(first > second for first, second in pairwise(energies))
    assert energies[-1] > 0.5
    assert abs(energies[-1] - energies[-2]) < 1e-6
    assert solution.optimize_gaussian(coupling=0.1).energy > energies[-1]
