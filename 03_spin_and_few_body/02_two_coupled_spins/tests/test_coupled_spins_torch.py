"""Tensor-product conventions, exact exchange dynamics, and conditional symmetries."""

from itertools import pairwise
from math import pi

import coupled_spins_torch_solution as solution
import pytest
import torch


def test_local_actions_and_cross_site_commutation() -> None:
    first, second = solution.local_operators()
    basis = torch.eye(4, dtype=torch.complex128)
    torch.testing.assert_close(first[0] @ basis[:, 1], basis[:, 3])
    torch.testing.assert_close(second[0] @ basis[:, 1], basis[:, 0])
    torch.testing.assert_close(first[2] @ basis[:, 1], basis[:, 1])
    torch.testing.assert_close(second[2] @ basis[:, 1], -basis[:, 1])
    torch.testing.assert_close(first, first.mH)
    for a in range(3):
        for b in range(3):
            torch.testing.assert_close(
                first[a] @ second[b], second[b] @ first[a], atol=0, rtol=0
            )


def test_product_state_order_and_batch_pairing() -> None:
    first = torch.tensor([[1, 0], [1, 1j]], dtype=torch.complex128)
    second = torch.tensor([[0, 1], [1, -1]], dtype=torch.complex128)
    joint = solution.product_states(first, second)
    expected = torch.tensor([[0, 1, 0, 0], [1, -1, 1j, -1j]], dtype=torch.complex128)
    expected[1] /= 2
    torch.testing.assert_close(joint, expected)
    torch.testing.assert_close(
        solution.product_states(first[1], second[1]), expected[1]
    )
    torch.testing.assert_close(
        solution.product_determinant(joint), torch.zeros(2, dtype=torch.float64)
    )


@pytest.mark.parametrize("exchange", [-1.0, 0.0, 1.0, 2.3])
def test_singlet_triplet_spectrum(exchange: float) -> None:
    operators = solution.hamiltonians(
        torch.tensor([exchange], dtype=torch.float64),
        torch.zeros((1, 2, 3), dtype=torch.float64),
    )[0]
    basis = solution.singlet_triplet_basis()
    energies = torch.tensor(
        [exchange / 4] * 3 + [-3 * exchange / 4], dtype=torch.float64
    )
    torch.testing.assert_close(
        operators @ basis, basis * energies[None, :], atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        basis.mH @ basis, torch.eye(4, dtype=torch.complex128), atol=1e-14, rtol=0
    )
    first, second = solution.local_operators()
    total_spin_squared = (((first + second) / 2) @ ((first + second) / 2)).sum(0)
    torch.testing.assert_close(
        total_spin_squared @ basis,
        basis * torch.tensor([2, 2, 2, 0], dtype=torch.float64),
        atol=1e-14,
        rtol=0,
    )


def test_noninteracting_product_of_single_spin_propagators() -> None:
    fields = torch.tensor([[[0.3, -0.4, 1], [-0.2, 0.7, 0.1]]], dtype=torch.float64)
    gamma, hbar, time = -1.7, 0.6, 2.3
    initial_first = torch.tensor([1, 1j], dtype=torch.complex128) / 2**0.5
    initial_second = torch.tensor([1, 0], dtype=torch.complex128)
    initial = solution.product_states(initial_first, initial_second)[None, :]
    operators = solution.hamiltonians(torch.zeros(1), fields, gamma, hbar)
    result = solution.evolve_constant(
        initial, operators, torch.tensor([0, time], dtype=torch.float64), hbar
    )
    individual = []
    for field, state in zip(fields[0], (initial_first, initial_second), strict=True):
        operator = (
            -gamma
            * hbar
            / 2
            * torch.einsum(
                "a,aij->ij", field.to(torch.complex128), solution.pauli_matrices()
            )
        )
        individual.append(torch.linalg.matrix_exp(-1j * time * operator / hbar) @ state)
    expected = torch.kron(individual[0], individual[1])
    torch.testing.assert_close(result.states[-1, 0], expected, atol=1e-14, rtol=0)
    assert float(solution.product_determinant(result.states).max()) < 1e-14
    assert (
        float(solution.correlations(result.states, connected=True).abs().max()) < 1e-14
    )


@pytest.mark.parametrize("exchange", [-1.0, 0.0, 1.0])
def test_exact_zero_field_exchange(exchange: float) -> None:
    hbar = 0.7
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    times = torch.linspace(-1, 8, 31, dtype=torch.float64)
    operators = solution.hamiltonians(torch.tensor([exchange]), torch.zeros((1, 2, 3)))
    result = solution.evolve_constant(initial, operators, times, hbar)
    expected = solution.exchange_reference(times, exchange, hbar=hbar)
    torch.testing.assert_close(result.states[:, 0], expected, atol=2e-14, rtol=0)
    torch.testing.assert_close(
        solution.joint_probabilities(result.states)[:, 0, 2],
        torch.sin(exchange * times / (2 * hbar)) ** 2,
        atol=2e-14,
        rtol=0,
    )


def test_detuned_longitudinal_block_and_common_field_cancellation() -> None:
    gamma, hbar, exchange = -0.8, 0.7, 1.2
    fields = torch.tensor([[[0, 0, 1.1], [0, 0, -0.4]]], dtype=torch.float64)
    times = torch.linspace(0, 9, 37, dtype=torch.float64)
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    operators = solution.hamiltonians(
        torch.tensor([exchange], dtype=torch.float64), fields, gamma, hbar
    )
    result = solution.evolve_constant(initial, operators, times, hbar)
    expected = solution.exchange_reference(times, exchange, gamma * hbar * 1.5, hbar)
    torch.testing.assert_close(result.states[:, 0], expected, atol=2e-14, rtol=0)
    shifted = fields.clone()
    shifted[:, :, 2] += 0.9
    shifted_result = solution.evolve_constant(
        initial,
        solution.hamiltonians(
            torch.tensor([exchange], dtype=torch.float64), shifted, gamma, hbar
        ),
        times,
        hbar,
    )
    torch.testing.assert_close(result.states, shifted_result.states, atol=2e-14, rtol=0)


def test_mid_exchange_correlations_and_nonfactorization() -> None:
    state = solution.exchange_reference(torch.tensor([pi / 2], dtype=torch.float64))[0]
    torch.testing.assert_close(
        solution.local_bloch_vectors(state),
        torch.zeros((2, 3), dtype=torch.float64),
        atol=2e-15,
        rtol=0,
    )
    expected = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=torch.float64)
    torch.testing.assert_close(
        solution.correlations(state), expected, atol=2e-15, rtol=0
    )
    torch.testing.assert_close(
        solution.correlations(state, connected=True), expected, atol=2e-15, rtol=0
    )
    assert float(solution.product_determinant(state)) == pytest.approx(0.5, abs=1e-15)
    torch.testing.assert_close(
        solution.joint_probabilities(state),
        torch.tensor([0, 0.5, 0.5, 0], dtype=torch.float64),
        atol=1e-15,
        rtol=0,
    )


def test_only_appropriate_spin_quantities_are_conserved() -> None:
    first, second = solution.local_operators()
    total_z = (first[2] + second[2]) / 2
    total_squared = (((first + second) / 2) @ ((first + second) / 2)).sum(0)
    coupling = torch.tensor([1.0])
    uniform = torch.tensor([[[0.3, -0.4, 1], [0.3, -0.4, 1]]], dtype=torch.float64)
    common_h = solution.hamiltonians(coupling, uniform)[0]
    torch.testing.assert_close(
        common_h @ total_squared, total_squared @ common_h, atol=1e-14, rtol=0
    )
    longitudinal = torch.tensor([[[0, 0, 1], [0, 0, -0.4]]], dtype=torch.float64)
    unequal_h = solution.hamiltonians(coupling, longitudinal)[0]
    torch.testing.assert_close(unequal_h @ total_z, total_z @ unequal_h)
    assert (
        float((unequal_h @ total_squared - total_squared @ unequal_h).abs().max()) > 0.1
    )
    transverse = torch.tensor([[[1, 0, 0], [0, 0, 0]]], dtype=torch.float64)
    general_h = solution.hamiltonians(coupling, transverse)[0]
    assert float((general_h @ total_z - total_z @ general_h).abs().max()) > 0.1


def test_cn_batch_reversal_energy_and_second_order() -> None:
    coupling = torch.tensor([1.0, -0.6], dtype=torch.float64)
    fields = torch.tensor(
        [[[0.3, -0.4, 1], [-0.2, 0, 0.7]], [[0, 0, 0], [0, 0, 0]]], dtype=torch.float64
    )
    operators = solution.hamiltonians(coupling, fields)
    initial = solution.normalize_states(
        torch.tensor([[1, 1j, 0.3, -0.7], [0, 1, 0, 0]], dtype=torch.complex128)
    )
    errors = []
    for dt in (0.2, 0.1, 0.05):
        result = solution.propagate_cn(initial, operators, dt, round(4 / dt), 7)
        exact = solution.evolve_constant(initial, operators, result.times)
        errors.append(
            float(
                solution.phase_aligned_errors(result.states[-1], exact.states[-1]).max()
            )
        )
        energy = solution.energy_expectations(result)
        torch.testing.assert_close(
            energy, energy[0].expand_as(energy), atol=4e-14, rtol=0
        )
        assert float((result.states.abs().square().sum(-1) - 1).abs().max()) < 5e-14
        reverse = solution.propagate_cn(
            result.states[-1], operators, -dt, round(4 / dt), 1000
        )
        torch.testing.assert_close(reverse.states[-1], initial, atol=3e-14, rtol=0)
    for previous, current in pairwise(errors):
        assert 3.8 < previous / current < 4.2
    for index in range(2):
        single = solution.propagate_cn(
            initial[index : index + 1], operators[index : index + 1], 0.05, 80, 7
        )
        torch.testing.assert_close(
            single.states[:, 0], result.states[:, index], atol=2e-14, rtol=0
        )


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="fields"):
        solution.hamiltonians(torch.ones(2), torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="nonzero norms"):
        solution.product_states(torch.zeros(2), torch.ones(2))
    matrix = torch.eye(4, dtype=torch.complex128)[None, :, :]
    matrix[0, 0, 1] = torch.tensor(1j, dtype=torch.complex128)
    with pytest.raises(ValueError, match="Hermitian"):
        solution.evolve_constant(torch.ones((1, 4)), matrix, torch.tensor([0.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu() -> None:
    coupling = torch.tensor([1.0], dtype=torch.float64)
    fields = torch.zeros((1, 2, 3), dtype=torch.float64)
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    times = torch.linspace(0, 3, 11, dtype=torch.float64)
    cpu = solution.evolve_constant(
        initial, solution.hamiltonians(coupling, fields), times
    )
    gpu = solution.evolve_constant(
        initial.cuda(),
        solution.hamiltonians(coupling.cuda(), fields.cuda()),
        times.cuda(),
    )
    torch.testing.assert_close(cpu.states, gpu.states.cpu(), atol=1e-13, rtol=0)
