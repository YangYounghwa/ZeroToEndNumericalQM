"""Cross-library agreement and independently checked finite-difference references."""

import numpy as np
import pytest
import torch
import tunnelling_numpy_solution as reference
import tunnelling_torch_solution as solution
from scipy.sparse.linalg import expm_multiply


def test_numpy_torch_scattering_agree() -> None:
    numpy_result = reference.solve_tunnelling()
    torch_result = solution.solve_tunnelling()
    np.testing.assert_allclose(
        numpy_result.grid, torch_result.grid.numpy(), atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        numpy_result.wavefunctions,
        torch_result.wavefunctions.numpy(),
        atol=2e-12,
        rtol=0,
    )


def test_periodic_fd_fourier_eigenvalue_and_corner_links() -> None:
    x, dx, _ = reference.make_grid(80, 10)
    mass, hbar, wave_number = 1.7, 0.8, 3 * np.pi / 10
    potential = np.full_like(x, 0.4)
    hamiltonian = reference.periodic_fd_hamiltonian(potential, dx, mass, hbar)
    mode = np.exp(1j * wave_number * x)
    eigenvalue = 2 * hbar**2 / (mass * dx**2) * np.sin(wave_number * dx / 2) ** 2 + 0.4
    np.testing.assert_allclose(
        hamiltonian @ mode, eigenvalue * mode, atol=1e-13, rtol=0
    )
    assert hamiltonian[0, -1] == pytest.approx(-(hbar**2) / (2 * mass * dx**2))
    assert hamiltonian.nnz == 3 * len(x)


def test_cn_second_order_and_conservation() -> None:
    x, dx, _ = reference.make_grid(96, 12)
    hamiltonian = reference.periodic_fd_hamiltonian(reference.gaussian_barrier(x), dx)
    initial = reference.normalize_columns(
        reference.gaussian_packet(x, -4, 1)[:, None], dx
    )
    exact = expm_multiply(-4j * hamiltonian, initial[:, 0])
    errors = []
    for dt in (0.04, 0.02, 0.01):
        _, history = reference.propagate_cn(
            initial, hamiltonian, dx, dt, round(4 / dt), 1000
        )
        final = history[-1, :, 0]
        errors.append(
            solution.phase_aligned_error(
                torch.from_numpy(final), torch.from_numpy(exact), dx
            )
        )
        assert dx * np.sum(np.abs(final) ** 2) == pytest.approx(1, abs=1e-12)
        assert (dx * np.vdot(final, hamiltonian @ final)).real == pytest.approx(
            (dx * np.vdot(initial[:, 0], hamiltonian @ initial[:, 0])).real, abs=1e-12
        )
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_fd_and_fft_approach_same_scattering() -> None:
    transmissions = []
    for points in (512, 1024, 2048):
        x, dx, _ = reference.make_grid(points, 64)
        potential = reference.gaussian_barrier(x)
        initial = reference.gaussian_packet(x)[:, None]
        _, history = reference.propagate_cn(
            initial,
            reference.periodic_fd_hamiltonian(potential, dx),
            dx,
            0.01,
            3000,
            3000,
        )
        transmissions.append(float(dx * np.sum(np.abs(history[-1, x > 4, 0]) ** 2)))
    fft = float(
        solution.region_probabilities(
            solution.solve_tunnelling(time_step=0.01, num_steps=3000)
        )[-1, 2]
    )
    errors = [abs(value - fft) for value in transmissions]
    assert errors[0] > 3 * errors[1] > 9 * errors[2]
    assert errors[-1] < 1e-4
