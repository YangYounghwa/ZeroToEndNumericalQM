"""Spin algebra, independent analytical predictions, and time-step checks."""

from itertools import pairwise
from math import pi

import pytest
import spin_field_torch_solution as solution
import torch


def test_pauli_algebra_and_total_spin() -> None:
    sigma = solution.pauli_matrices()
    identity = torch.eye(2, dtype=torch.complex128)
    torch.testing.assert_close(sigma, sigma.mH)
    for component in sigma:
        torch.testing.assert_close(component @ component, identity)
        torch.testing.assert_close(
            torch.linalg.eigvalsh(component), torch.tensor([-1, 1], dtype=torch.float64)
        )
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        torch.testing.assert_close(
            sigma[a] @ sigma[b] - sigma[b] @ sigma[a], 2j * sigma[c]
        )
        torch.testing.assert_close(
            sigma[a] @ sigma[b] + sigma[b] @ sigma[a], torch.zeros_like(identity)
        )
    spin = 0.7 * sigma / 2
    torch.testing.assert_close((spin @ spin).sum(0), 3 * 0.7**2 / 4 * identity)


def test_spinor_bloch_coordinates_and_measurement_projectors() -> None:
    theta, phi = pi / 3, pi / 5
    state = solution.spinor(theta, phi)
    # Compute the reference angles in float64, independently of the spinor code.
    angle = torch.tensor([theta, phi], dtype=torch.float64)
    expected = torch.stack(
        [
            angle[0].sin() * angle[1].cos(),
            angle[0].sin() * angle[1].sin(),
            angle[0].cos(),
        ]
    )
    torch.testing.assert_close(
        solution.bloch_vectors(state), expected, atol=2e-15, rtol=0
    )
    axis = torch.tensor([1, -2, 3], dtype=torch.float64)
    direction = axis / torch.linalg.vector_norm(axis)
    observable = torch.einsum(
        "a,aij->ij", direction.to(torch.complex128), solution.pauli_matrices()
    )
    projectors = torch.stack(
        [(torch.eye(2) + observable) / 2, (torch.eye(2) - observable) / 2]
    )
    direct = torch.einsum("i,aij,j->a", state.conj(), projectors, state).real
    torch.testing.assert_close(
        solution.measurement_probabilities(state, axis), direct, atol=2e-15, rtol=0
    )
    torch.testing.assert_close(
        solution.measurement_probabilities(state, expected),
        torch.tensor([1, 0], dtype=torch.float64),
        atol=2e-15,
        rtol=0,
    )
    # Measurement weights expose an unnormalized input instead of normalizing it.
    assert float(
        solution.measurement_probabilities(2 * state, axis).sum()
    ) == pytest.approx(4)


@pytest.mark.parametrize("gamma", [1.0, -1.7, 0.0])
def test_matrix_exponential_closed_form_rodrigues_and_energy(gamma: float) -> None:
    fields = torch.tensor(
        [[0, 0, 0], [0.3, -0.4, 1], [-1, 2, 0.5]], dtype=torch.float64
    )
    initial = torch.stack(
        [solution.spinor(), solution.spinor(0), solution.spinor(pi / 3, pi / 5)]
    )
    times = torch.tensor([-1.1, 0, 0.2, 1.3, 6.7], dtype=torch.float64)
    result = solution.evolve_constant(initial, fields, times, gamma, hbar=0.7)
    propagators = solution.matrix_propagators(fields, times, gamma, hbar=0.7)
    torch.testing.assert_close(
        propagators,
        solution.closed_form_propagators(fields, times, gamma, hbar=0.7),
        atol=2e-14,
        rtol=0,
    )
    identity = torch.eye(2, dtype=torch.complex128).expand_as(propagators)
    torch.testing.assert_close(
        propagators.mH @ propagators, identity, atol=2e-14, rtol=0
    )
    expected = solution.rodrigues_bloch(
        solution.bloch_vectors(initial), fields, times, gamma
    )
    torch.testing.assert_close(
        solution.bloch_vectors(result.states), expected, atol=3e-14, rtol=0
    )
    energy = solution.energy_expectations(result)
    torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=3e-14, rtol=0)
    torch.testing.assert_close(
        result.states[:, 0],
        initial[0].expand_as(result.states[:, 0]),
        atol=1e-14,
        rtol=0,
    )


def test_signed_larmor_rotation_and_transverse_spin_flip() -> None:
    times = torch.tensor([0, pi / 2, pi, 2 * pi], dtype=torch.float64)
    fields = torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float64)
    initial = torch.stack([solution.spinor(), solution.spinor(0)])
    result = solution.evolve_constant(initial, fields, times)
    expected = torch.stack([times.cos(), -times.sin(), torch.zeros_like(times)], 1)
    torch.testing.assert_close(
        solution.bloch_vectors(result.states[:, 0]), expected, atol=3e-15, rtol=0
    )
    down = solution.measurement_probabilities(result.states[:, 1], fields[0])[:, 1]
    torch.testing.assert_close(down, torch.sin(times / 2) ** 2, atol=3e-15, rtol=0)


def test_two_pi_and_four_pi_spinor_rotations() -> None:
    field = torch.tensor([[0, 0, 1]], dtype=torch.float64)
    initial = solution.spinor()[None, :]
    result = solution.evolve_constant(
        initial, field, torch.tensor([0, 2 * pi, 4 * pi], dtype=torch.float64)
    )
    torch.testing.assert_close(result.states[1], -initial, atol=5e-15, rtol=0)
    torch.testing.assert_close(result.states[2], initial, atol=5e-15, rtol=0)
    torch.testing.assert_close(
        solution.bloch_vectors(result.states),
        solution.bloch_vectors(initial).expand(3, -1, -1),
        atol=1e-14,
        rtol=0,
    )


def test_energy_eigenstate_only_accumulates_phase() -> None:
    fields = torch.tensor([[0.3, -0.4, 1]], dtype=torch.float64)
    energies, vectors = torch.linalg.eigh(solution.hamiltonians(fields))
    initial = vectors[:, :, 0]
    times = torch.linspace(0, 10, 21, dtype=torch.float64)
    result = solution.evolve_constant(initial, fields, times)
    expected = (
        torch.exp(-1j * times[:, None, None] * energies[:, 0][None, :, None]) * initial
    )
    torch.testing.assert_close(result.states, expected, atol=1e-14, rtol=0)
    assert float(solution.phase_aligned_errors(result.states, initial).max()) < 1e-14


def test_cn_second_order_and_wrong_phase_despite_conserved_norm() -> None:
    fields = torch.tensor([[0, 0, 1]], dtype=torch.float64)
    initial = solution.spinor()[None, :]
    errors = []
    for dt in (0.4, 0.2, 0.1, 0.05):
        result = solution.propagate_cn(
            initial, fields, dt, round(10 / dt), store_every=1000
        )
        exact = solution.evolve_constant(initial, fields, result.times)
        errors.append(
            float(solution.phase_aligned_errors(result.states[-1], exact.states[-1])[0])
        )
        assert float((result.states.abs().square().sum(-1) - 1).abs().max()) < 1e-13
    assert errors[0] > 0.01
    for previous, current in pairwise(errors):
        assert 3.9 < previous / current < 4.1
    coarse = solution.propagate_cn(initial, fields, 0.4, 25, store_every=25)
    angle = 25 * 4 * torch.atan(torch.tensor(0.4 / 4, dtype=torch.float64))
    effective = torch.stack([angle.cos(), -angle.sin(), torch.zeros_like(angle)])
    torch.testing.assert_close(
        solution.bloch_vectors(coarse.states[-1, 0]), effective, atol=3e-14, rtol=0
    )


def test_cn_batch_energy_reversal_and_snapshots() -> None:
    fields = torch.tensor([[0.3, -0.4, 1], [1, 0, 0]], dtype=torch.float64)
    initial = torch.stack([solution.spinor(), solution.spinor(0)])
    result = solution.propagate_cn(initial, fields, 0.07, 20, 7)
    torch.testing.assert_close(
        result.times, 0.07 * torch.tensor([0, 7, 14, 20], dtype=torch.float64)
    )
    for index in range(2):
        single = solution.propagate_cn(
            initial[index : index + 1], fields[index : index + 1], 0.07, 20, 7
        )
        torch.testing.assert_close(
            result.states[:, index], single.states[:, 0], atol=1e-14, rtol=0
        )
    energy = solution.energy_expectations(result)
    torch.testing.assert_close(energy, energy[0].expand_as(energy), atol=1e-14, rtol=0)
    reverse = solution.propagate_cn(result.states[-1], fields, -0.07, 20, 20)
    torch.testing.assert_close(reverse.states[-1], initial, atol=1e-14, rtol=0)


def test_hbar_changes_energy_not_rotation_at_fixed_gamma() -> None:
    fields = torch.tensor([[0.3, -0.4, 1]], dtype=torch.float64)
    initial = solution.spinor()[None, :]
    times = torch.tensor([0, 2], dtype=torch.float64)
    first = solution.evolve_constant(initial, fields, times, hbar=1)
    second = solution.evolve_constant(initial, fields, times, hbar=0.3)
    torch.testing.assert_close(first.states, second.states, atol=1e-14, rtol=0)
    torch.testing.assert_close(0.3 * first.hamiltonians, second.hamiltonians)


def test_invalid_inputs() -> None:
    fields = torch.tensor([[0, 0, 1]], dtype=torch.float64)
    times = torch.tensor([0, 1], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonzero norms"):
        solution.evolve_constant(torch.zeros((1, 2)), fields, times)
    with pytest.raises(ValueError, match="finite and real"):
        solution.hamiltonians(fields.to(torch.complex128))
    with pytest.raises(ValueError, match="nonempty finite real"):
        solution.matrix_propagators(fields, torch.tensor([]))
    with pytest.raises(ValueError, match="nonzero"):
        solution.measurement_probabilities(solution.spinor(), torch.zeros(3))
    with pytest.raises(ValueError, match="step counts"):
        solution.propagate_cn(solution.spinor()[None, :], fields, time_step=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu() -> None:
    fields = torch.tensor([[0.3, -0.4, 1]], dtype=torch.float64)
    times = torch.linspace(0, 2, 11, dtype=torch.float64)
    initial = solution.spinor()[None, :]
    cpu = solution.evolve_constant(initial, fields, times)
    gpu = solution.evolve_constant(initial.cuda(), fields.cuda(), times.cuda())
    torch.testing.assert_close(gpu.states.cpu(), cpu.states, atol=1e-13, rtol=0)
