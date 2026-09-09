import barrier_scattering_numpy_solution as reference
import barrier_scattering_torch_solution as solution
import numpy as np
import pytest
import torch


def test_barrier_and_hamiltonian_match_sparse_reference() -> None:
    grid, spacing, potential, hamiltonian = solution.build_hamiltonian(
        159, 12, height=1.2
    )
    np_grid, np_spacing, np_potential, sparse = reference.build_hamiltonian(
        159, 12, height=1.2
    )
    assert grid.dtype == potential.dtype == hamiltonian.dtype == torch.float64
    assert spacing == np_spacing
    np.testing.assert_allclose(grid.numpy(), np_grid, atol=1e-14, rtol=0)
    np.testing.assert_allclose(potential.numpy(), np_potential, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        hamiltonian.numpy(), sparse.toarray(), atol=1e-12, rtol=0
    )
    assert torch.allclose(hamiltonian, hamiltonian.mH, atol=1e-14, rtol=0)


def test_torch_and_scipy_agree_through_an_actual_collision() -> None:
    numpy_result = reference.solve_scattering()
    result = solution.solve_scattering()
    np.testing.assert_allclose(
        result.times.numpy(), numpy_result.times, atol=1e-14, rtol=0
    )
    np.testing.assert_allclose(
        result.wavefunctions.numpy(), numpy_result.wavefunctions, atol=2e-12, rtol=0
    )
    regions = solution.region_probabilities(result)
    norm = regions.left + regions.near + regions.right
    assert torch.max(torch.abs(norm - 1)) < 2e-12
    energy = solution.energy_expectations(result)
    assert torch.max(torch.abs(energy - energy[0])) < 2e-11
    assert regions.near[-1] < 2e-4
    assert 0.01 < regions.right[-1] < 0.8


def test_batch_matches_individual_packets_and_saves_final_step() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(79, 10)
    initial = torch.stack(
        [
            solution.gaussian_packet(grid, center=-4, sigma=0.7, wave_number=k)
            for k in (1.4, 2.0, 2.6)
        ],
        dim=1,
    )
    times, batch = solution.propagate_batch(
        initial, hamiltonian, spacing, num_steps=23, store_every=10
    )
    assert batch.shape == (4, 79, 3)
    torch.testing.assert_close(
        times,
        torch.tensor([0, 0.2, 0.4, 0.46], dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )
    for index in range(3):
        _, single = solution.propagate_batch(
            initial[:, index : index + 1],
            hamiltonian,
            spacing,
            num_steps=23,
            store_every=10,
        )
        torch.testing.assert_close(
            batch[:, :, index], single[:, :, 0], atol=1e-12, rtol=0
        )


def test_local_continuity_equation_for_cn_midpoint_current() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(
        99, 10, mass=1.3, hbar=0.8
    )
    initial = solution.gaussian_packet(grid, center=-2, sigma=0.8)
    dt = 0.02
    _, states = solution.propagate_batch(
        initial[:, None], hamiltonian, spacing, dt, 1, 1, hbar=0.8
    )
    before, after = states[0, :, 0], states[1, :, 0]
    midpoint = (before + after) / 2
    current = solution.link_current(midpoint, spacing, mass=1.3, hbar=0.8)
    full_current = torch.cat(
        (
            torch.zeros(1, dtype=torch.float64),
            current,
            torch.zeros(1, dtype=torch.float64),
        )
    )
    density_rate = (torch.abs(after) ** 2 - torch.abs(before) ** 2) / dt
    divergence = (full_current[1:] - full_current[:-1]) / spacing
    assert torch.max(torch.abs(density_rate + divergence)) < 1e-12


def test_forward_and_reverse_recover_initial_state() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(79, 10)
    initial = solution.gaussian_packet(grid, center=-4, sigma=0.8)[:, None]
    _, forward = solution.propagate_batch(initial, hamiltonian, spacing, 0.02, 60, 60)
    _, backward = solution.propagate_batch(
        forward[-1], hamiltonian, spacing, -0.02, 60, 60
    )
    torch.testing.assert_close(backward[-1], forward[0], atol=1e-12, rtol=0)


def test_time_error_is_second_order_on_a_fixed_grid() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(99, 12)
    initial = solution.normalize_columns(
        solution.gaussian_packet(grid, center=-6, sigma=1)[:, None], spacing
    )
    exact: torch.Tensor = torch.linalg.matrix_exp(-4j * hamiltonian) @ initial[:, 0]
    errors = []
    for dt in (0.02, 0.01):
        steps = round(4 / dt)
        _, states = solution.propagate_batch(
            initial, hamiltonian, spacing, dt, steps, steps
        )
        final = states[-1, :, 0]
        overlap = spacing * torch.vdot(exact, final)
        aligned = final * torch.exp(-1j * torch.angle(overlap))
        errors.append(
            float(torch.sqrt(spacing * torch.sum(torch.abs(aligned - exact) ** 2)))
        )
    assert 3.7 < errors[0] / errors[1] < 4.1


def test_plane_wave_transmission_limits_and_threshold() -> None:
    energies = torch.tensor([0, 1, 2.5, 10], dtype=torch.float64)
    torch.testing.assert_close(
        solution.plane_wave_transmission(energies, height=0), torch.ones_like(energies)
    )
    torch.testing.assert_close(
        solution.plane_wave_transmission(energies, width=0), torch.ones_like(energies)
    )
    transmission = solution.plane_wave_transmission(energies)
    assert transmission[0] == 0
    assert 0 < transmission[1] < 1
    assert abs(transmission[2].item() - 1 / (1 + 2.5 * 2**2 / 2)) < 1e-14
    resonance_energy = 2.5 + (torch.pi / 2) ** 2 / 2
    assert (
        abs(
            solution.plane_wave_transmission(
                torch.tensor(resonance_energy, dtype=torch.float64)
            ).item()
            - 1
        )
        < 1e-14
    )
    near_threshold = solution.plane_wave_transmission(
        torch.tensor([2.5 - 1e-9, 2.5 + 1e-9], dtype=torch.float64)
    )
    torch.testing.assert_close(
        near_threshold, transmission[2].expand(2), atol=1e-9, rtol=0
    )
    thick = solution.plane_wave_transmission(energies[:2], width=1000)
    assert bool(torch.isfinite(thick).all())
    assert bool(torch.all(thick < 1e-100))


def test_plane_wave_formula_matches_independent_boundary_matching() -> None:
    for energy in (0.2, 1.0, 2.5, 4.0, 10.0):
        k = np.sqrt(2 * energy)
        q = np.sqrt(complex(2 * (energy - 2.5)))
        cosine = np.cos(2 * q)
        sine_over_q = 2.0 if q == 0 else np.sin(2 * q) / q
        phase = np.exp(2j * k)
        # Unknowns: reflected amplitude, interior value, interior derivative,
        # transmitted amplitude. Enforce value/derivative at x=0 and x=2.
        matrix = np.array(
            [
                [-1, 1, 0, 0],
                [1j * k, 0, 1, 0],
                [0, cosine, sine_over_q, -phase],
                [0, -q * np.sin(2 * q), cosine, -1j * k * phase],
            ],
            dtype=np.complex128,
        )
        coefficients = np.linalg.solve(matrix, np.array([1, 1j * k, 0, 0]))
        expected = abs(coefficients[-1]) ** 2
        actual = solution.plane_wave_transmission(
            torch.tensor(energy, dtype=torch.float64)
        ).item()
        assert abs(actual - expected) < 1e-12
        assert abs(abs(coefficients[0]) ** 2 + expected - 1) < 1e-12


def test_packet_reference_resolves_spectrum_and_quadrature() -> None:
    packet = solution.packet_transmission_reference()
    finer = solution.packet_transmission_reference(num_samples=8001)
    central = solution.plane_wave_transmission(torch.tensor(2.0, dtype=torch.float64))
    assert abs(packet.item() - finer.item()) < 1e-8
    assert abs(packet.item() - central.item()) > 0.03
    narrow = solution.packet_transmission_reference(sigma=30)
    assert abs(narrow.item() - central.item()) < 0.001


def test_invalid_propagation_inputs_are_rejected() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(39, 10)
    with pytest.raises(ValueError, match="nonzero"):
        solution.propagate_batch(
            torch.zeros((39, 1), dtype=torch.complex128), hamiltonian, spacing
        )
    with pytest.raises(ValueError, match="positive"):
        solution.propagate_batch(
            solution.gaussian_packet(grid)[:, None], hamiltonian, spacing, store_every=0
        )
    bad = hamiltonian.clone()
    bad[0, 1] += 0.1
    with pytest.raises(ValueError, match="Hermitian"):
        solution.propagate_batch(solution.gaussian_packet(grid)[:, None], bad, spacing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_on_a_small_batch() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(39, 10)
    initial = solution.gaussian_packet(grid, center=-4, sigma=1)[:, None]
    _, cpu = solution.propagate_batch(initial, hamiltonian, spacing, num_steps=10)
    _, cuda = solution.propagate_batch(
        initial.cuda(), hamiltonian.cuda(), spacing, num_steps=10
    )
    torch.testing.assert_close(cuda.cpu(), cpu, atol=1e-11, rtol=0)
