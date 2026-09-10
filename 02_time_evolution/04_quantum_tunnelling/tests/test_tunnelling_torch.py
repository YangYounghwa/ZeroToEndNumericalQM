"""Independent physics references and checks of FFT propagation."""

from math import pi, sqrt

import pytest
import torch
import tunnelling_torch_solution as solution


def test_periodic_grid_and_fft_order() -> None:
    x, dx, k = solution.make_grid(8, 4)
    torch.testing.assert_close(x, torch.arange(-4, 4, dtype=torch.float64))
    expected = (pi / 4) * torch.tensor(
        [0, 1, 2, 3, -4, -3, -2, -1], dtype=torch.float64
    )
    torch.testing.assert_close(k, expected)
    assert dx == 1


@pytest.mark.parametrize("points", [63, 64])
def test_free_fourier_modes_exact_in_batch(points: int) -> None:
    x, dx, k = solution.make_grid(points, 10)
    mass, hbar, dt, steps = 1.7, 0.8, 0.13, 7
    modes = torch.tensor([2, -3, 5], dtype=torch.float64) * pi / 10
    initial = torch.exp(1j * x[:, None] * modes[None, :]) / sqrt(20)
    kinetic = hbar**2 * k**2 / (2 * mass)
    times, history = solution.propagate_batch(
        initial, kinetic, torch.full_like(x, 0.4), dx, dt, steps, 3, hbar
    )
    expected = initial * torch.exp(
        -1j * (hbar**2 * modes**2 / (2 * mass) + 0.4) * (dt * steps) / hbar
    )
    torch.testing.assert_close(history[-1], expected, atol=2e-13, rtol=0)
    torch.testing.assert_close(
        times, dt * torch.tensor([0, 3, 6, 7], dtype=torch.float64)
    )


def test_second_order_against_spectral_exponential() -> None:
    x, dx, k = solution.make_grid(96, 12)
    potential = solution.gaussian_barrier(x)
    initial = solution.normalize_columns(
        solution.gaussian_packet(x, -4, 0.8)[:, None], dx
    )
    hamiltonian = solution.spectral_hamiltonian(k**2 / 2, potential)
    torch.testing.assert_close(hamiltonian, hamiltonian.mH, atol=1e-12, rtol=0)
    exact: torch.Tensor = torch.linalg.matrix_exp(-4j * hamiltonian) @ initial[:, 0]
    errors = []
    for dt in (0.1, 0.05, 0.025):
        _, history = solution.propagate_batch(
            initial, k**2 / 2, potential, dx, dt, round(4 / dt), 1000
        )
        errors.append(solution.phase_aligned_error(history[-1, :, 0], exact, dx))
    assert 3.8 < errors[0] / errors[1] < 4.2
    assert 3.8 < errors[1] / errors[2] < 4.2


def test_batch_single_norm_and_reversal() -> None:
    x, dx, k = solution.make_grid(128, 16)
    initial = torch.stack(
        [solution.gaussian_packet(x, -5, 1, momentum) for momentum in (1.2, 1.8)], dim=1
    )
    potential = solution.gaussian_barrier(x)
    _, batch = solution.propagate_batch(initial, k**2 / 2, potential, dx, 0.03, 80, 17)
    torch.testing.assert_close(
        dx * batch.abs().square().sum(1),
        torch.ones((6, 2), dtype=torch.float64),
        atol=2e-13,
        rtol=0,
    )
    for column in range(2):
        _, single = solution.propagate_batch(
            initial[:, column : column + 1], k**2 / 2, potential, dx, 0.03, 80, 17
        )
        torch.testing.assert_close(
            batch[:, :, column], single[:, :, 0], atol=2e-13, rtol=0
        )
    _, backward = solution.propagate_batch(
        batch[-1], k**2 / 2, potential, dx, -0.03, 80, 80
    )
    torch.testing.assert_close(backward[-1], batch[0], atol=2e-13, rtol=0)


def test_incident_spectrum_matches_gaussian_tail() -> None:
    # A large box gives closely spaced k bins for independent numerical quadrature.
    x, dx, k = solution.make_grid(8192, 512)
    initial = solution.normalize_columns(
        solution.gaussian_packet(x, -20, 2)[:, None], dx
    )[:, 0]
    spectrum = torch.fft.fft(initial, norm="ortho")
    weights = dx * spectrum.abs().square()
    negative, above = solution.incident_tail_probabilities(sigma=2)
    assert float(weights[k < 0].sum()) == pytest.approx(negative, abs=1e-9)
    assert float(weights[k.square() / 2 > 2.5].sum()) == pytest.approx(above, rel=0.03)
    assert float((weights * k.square() / 2).sum()) == pytest.approx(
        1.5**2 / 2 + 1 / (8 * 2**2), abs=1e-12
    )
    assert solution.incident_tail_probabilities(sigma=3)[1] < above


def test_default_tunnelling_has_small_tail_and_boundary_error() -> None:
    result = solution.solve_tunnelling()
    budget = solution.region_probabilities(result)
    negative, above = solution.incident_tail_probabilities()
    assert negative < 1e-18
    assert above < 6e-6
    assert 0.015 < float(budget[-1, 2]) < 0.017
    assert float(budget[-1, 1]) < 1e-4
    assert float(budget[-1, 2]) > 1000 * above
    assert float(solution.boundary_probability(result).max()) < 1e-8
    assert float(solution.high_frequency_probability(result).max()) < 1e-12
    torch.testing.assert_close(
        budget.sum(1), torch.ones_like(result.times), atol=2e-12, rtol=0
    )
    energies = solution.energy_expectations(result)
    assert float((energies - energies[0]).abs().max()) < 1e-5


def test_free_packet_reappears_across_periodic_boundary() -> None:
    # This intentionally demonstrates wraparound, despite perfectly conserved norm.
    x, dx, k = solution.make_grid(512, 20)
    initial = solution.gaussian_packet(x, center=14, sigma=1, wave_number=2)
    _, history = solution.propagate_batch(
        initial[:, None], k**2 / 2, torch.zeros_like(x), dx, 0.1, 40, 40
    )
    final = history[-1, :, 0]
    assert float(dx * final[x < -10].abs().square().sum()) > 0.7
    assert float(dx * final.abs().square().sum()) == pytest.approx(1, abs=1e-12)


def test_reject_invalid_inputs() -> None:
    x, dx, k = solution.make_grid(32, 8)
    initial = solution.gaussian_packet(x, -4, 1)[:, None]
    with pytest.raises(ValueError, match="finite and real"):
        solution.propagate_batch(initial, k**2 / 2, x.to(torch.complex128), dx)
    with pytest.raises(ValueError, match="nonzero norms"):
        solution.propagate_batch(torch.zeros_like(initial), k**2 / 2, x, dx)
    with pytest.raises(ValueError, match="same size"):
        solution.propagate_batch(initial, k**2 / 2, x[:-1], dx)
    with pytest.raises(ValueError, match="step counts"):
        solution.propagate_batch(initial, k**2 / 2, x, dx, time_step=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu() -> None:
    cpu = solution.solve_tunnelling(
        num_points=128, half_extent=32, center=-10, num_steps=10
    )
    gpu = solution.solve_tunnelling(
        num_points=128, half_extent=32, center=-10, num_steps=10, device="cuda"
    )
    torch.testing.assert_close(
        gpu.wavefunctions.cpu(), cpu.wavefunctions, atol=1e-12, rtol=0
    )
