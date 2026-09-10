"""Analytical and same-grid checks for two-dimensional dynamics."""

from math import sqrt

import pytest
import torch
import wave_packet_2d_torch_solution as solution


def test_rectangular_modes_and_area_normalization() -> None:
    grid = solution.make_grid(30, 21, 9, 7)
    modes = ((2, -3), (-4, 1))
    initial = torch.stack(
        [
            torch.exp(
                1j
                * (
                    ix * torch.pi / 9 * grid.x[None, :]
                    + iy * torch.pi / 7 * grid.y[:, None]
                )
            )
            for ix, iy in modes
        ],
        dim=2,
    ) / sqrt(18 * 14)
    mass, hbar = 1.7, 0.8
    kinetic = solution.kinetic_energy(grid, mass, hbar)
    times, history = solution.propagate_batch(
        initial, kinetic, torch.full_like(kinetic, 0.3), grid.area, 0.13, 7, 3, hbar
    )
    energies = torch.tensor(
        [
            hbar**2 / (2 * mass) * ((ix * torch.pi / 9) ** 2 + (iy * torch.pi / 7) ** 2)
            + 0.3
            for ix, iy in modes
        ],
        dtype=torch.float64,
    )
    expected = initial * torch.exp(-1j * energies * times[-1] / hbar)[None, None, :]
    torch.testing.assert_close(history[-1], expected, atol=2e-13, rtol=0)
    assert history.shape == (4, 21, 30, 2)
    torch.testing.assert_close(
        grid.area * history.abs().square().sum((1, 2)),
        torch.ones((4, 2), dtype=torch.float64),
        atol=2e-13,
        rtol=0,
    )


def test_free_anisotropic_gaussian_state_and_moments() -> None:
    grid = solution.make_grid(128, 90, 20, 18)
    initial = solution.free_gaussian(grid)
    kinetic = solution.kinetic_energy(grid)
    times, history = solution.propagate_batch(
        initial[:, :, None], kinetic, torch.zeros_like(kinetic), grid.area, 0.1, 30, 10
    )
    result = solution.Evolution2D(
        grid, kinetic, torch.zeros_like(kinetic), times, history[:, :, :, 0]
    )
    centers, covariance = solution.position_statistics(result)
    expected_centers = torch.tensor([-3, 1], dtype=torch.float64) + times[
        :, None
    ] * torch.tensor([0.8, -0.4], dtype=torch.float64)
    expected_variances = torch.tensor([1, 1.4**2], dtype=torch.float64)[
        None, :
    ] + times[:, None] ** 2 / (4 * torch.tensor([1, 1.4**2], dtype=torch.float64))
    torch.testing.assert_close(centers, expected_centers, atol=1e-10, rtol=0)
    torch.testing.assert_close(
        covariance.diagonal(dim1=1, dim2=2), expected_variances, atol=1e-9, rtol=0
    )
    assert float(covariance[:, 0, 1].abs().max()) < 1e-10
    assert (
        solution.phase_aligned_error(
            result.wavefunctions[-1], solution.free_gaussian(grid, 3), grid.area
        )
        < 1e-9
    )


def test_coupled_motion_covariance_and_energy() -> None:
    result = solution.solve_coupled()
    centers, covariance = solution.position_statistics(result)
    expected, _, expected_covariance, expected_energy = solution.coherent_reference(
        result.times, solution.stiffness_matrix()
    )
    assert float((centers - expected).abs().max()) < 2e-4
    assert float((covariance - expected_covariance).abs().max()) < 7e-5
    assert abs(float(expected_covariance[0, 1])) > 0.05
    assert (
        float((solution.energy_expectations(result) - expected_energy).abs().max())
        < 2e-4
    )
    assert float(solution.boundary_probability(result).max()) < 1e-14


def test_normal_modes_match_uncoupled_limits_and_ehrenfest() -> None:
    stiffness = solution.stiffness_matrix(1, 1.3, 0)
    times = torch.tensor([0, 0.7, 1.4], dtype=torch.float64)
    center, _, covariance, _ = solution.coherent_reference(times, stiffness)
    expected_x = -2 * times.cos() + 0.4 * times.sin()
    expected_y = (1.3 * times).cos() - 0.6 / 1.3 * (1.3 * times).sin()
    torch.testing.assert_close(center, torch.stack([expected_x, expected_y], 1))
    torch.testing.assert_close(
        covariance, torch.diag(torch.tensor([0.5, 0.5 / 1.3], dtype=torch.float64))
    )
    coupled = solution.stiffness_matrix()
    epsilon = 1e-4
    stencil_times = torch.tensor(
        [0.7 - epsilon, 0.7, 0.7 + epsilon], dtype=torch.float64
    )
    positions, momenta, _, _ = solution.coherent_reference(stencil_times, coupled)
    torch.testing.assert_close(
        (momenta[2] - momenta[0]) / (2 * epsilon),
        -coupled @ positions[1],
        atol=1e-8,
        rtol=0,
    )


def test_second_order_against_small_spectral_exponential() -> None:
    grid = solution.make_grid(12, 10, 4, 3)
    # Smooth, periodic, nonseparable potential for a fixed-grid time-only check.
    potential = (
        0.7
        * torch.cos(torch.pi * grid.x[None, :] / 4)
        * torch.sin(torch.pi * grid.y[:, None] / 3)
    )
    kinetic = solution.kinetic_energy(grid)
    initial = solution.normalize_batch(
        solution.free_gaussian(grid, center=(-1, 0), sigma=(0.8, 0.7))[:, :, None],
        grid.area,
    )[:, :, 0]
    hamiltonian = solution.spectral_hamiltonian(kinetic, potential)
    torch.testing.assert_close(hamiltonian, hamiltonian.mH, atol=1e-12, rtol=0)
    exact: torch.Tensor = (
        torch.linalg.matrix_exp(-1j * hamiltonian) @ initial.reshape(-1)
    ).reshape(10, 12)
    errors = []
    for dt in (0.1, 0.05, 0.025):
        _, history = solution.propagate_batch(
            initial[:, :, None], kinetic, potential, grid.area, dt, round(1 / dt), 1000
        )
        errors.append(
            solution.phase_aligned_error(history[-1, :, :, 0], exact, grid.area)
        )
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_batch_single_reversal_and_no_mutation() -> None:
    grid = solution.make_grid(40, 32, 10, 8)
    stiffness = solution.stiffness_matrix()
    initial = torch.stack(
        [
            solution.coherent_packet(grid, stiffness, center=center)
            for center in ((-2, 1), (1, -1))
        ],
        dim=2,
    )
    original = initial.clone()
    kinetic, potential = (
        solution.kinetic_energy(grid),
        solution.coupled_potential(grid, stiffness),
    )
    _, batch = solution.propagate_batch(
        initial, kinetic, potential, grid.area, 0.02, 50, 17
    )
    for column in range(2):
        _, single = solution.propagate_batch(
            initial[:, :, column : column + 1],
            kinetic,
            potential,
            grid.area,
            0.02,
            50,
            17,
        )
        torch.testing.assert_close(
            single[:, :, :, 0], batch[:, :, :, column], atol=1e-13, rtol=0
        )
    _, reverse = solution.propagate_batch(
        batch[-1], kinetic, potential, grid.area, -0.02, 50, 50
    )
    torch.testing.assert_close(reverse[-1], batch[0], atol=1e-13, rtol=0)
    torch.testing.assert_close(initial, original)


def test_separable_product_matches_two_independent_axes() -> None:
    grid = solution.make_grid(40, 32, 10, 8)
    stiffness = solution.stiffness_matrix(coupling=0)
    initial = solution.coherent_packet(grid, stiffness)
    potential = solution.coupled_potential(grid, stiffness)
    _, history = solution.propagate_batch(
        initial[:, :, None],
        solution.kinetic_energy(grid),
        potential,
        grid.area,
        0.03,
        30,
        30,
    )
    # Construct normalized 1D Gaussian factors and evolve independently.
    factors = []
    for axis, k, spacing, frequency, center, p in zip(
        (grid.x, grid.y),
        (grid.kx, grid.ky),
        (grid.dx, grid.dy),
        (1, 1.3),
        (-2, 1),
        (0.4, -0.6),
        strict=True,
    ):
        state = torch.exp(
            -frequency * (axis - center) ** 2 / 2 + 1j * p * (axis - center)
        )
        state = state / torch.sqrt(spacing * state.abs().square().sum())
        half_v = torch.exp(-0.25j * 0.03 * frequency**2 * axis**2)
        kinetic_phase = torch.exp(-0.5j * 0.03 * k**2)
        for _ in range(30):
            state = half_v * torch.fft.ifft(
                kinetic_phase * torch.fft.fft(half_v * state, norm="ortho"),
                norm="ortho",
            )
        factors.append(state)
    torch.testing.assert_close(
        history[-1, :, :, 0],
        factors[1][:, None] * factors[0][None, :],
        atol=1e-13,
        rtol=0,
    )


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive-definite"):
        solution.stiffness_matrix(coupling=1.3)
    grid = solution.make_grid(12, 10, 4, 3)
    kinetic = solution.kinetic_energy(grid)
    with pytest.raises(ValueError, match="spatial shape"):
        solution.propagate_batch(torch.ones((10, 12, 1)), kinetic.T, kinetic, grid.area)
    with pytest.raises(ValueError, match="nonzero norms"):
        solution.propagate_batch(torch.zeros((10, 12, 1)), kinetic, kinetic, grid.area)
    with pytest.raises(ValueError, match="finite and real"):
        solution.propagate_batch(
            torch.ones((10, 12, 1)), kinetic, kinetic.to(torch.complex128), grid.area
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu() -> None:
    cpu = solution.solve_coupled(nx=32, ny=24, num_steps=10)
    gpu = solution.solve_coupled(nx=32, ny=24, num_steps=10, device="cuda")
    torch.testing.assert_close(
        gpu.wavefunctions.cpu(), cpu.wavefunctions, atol=1e-12, rtol=0
    )
