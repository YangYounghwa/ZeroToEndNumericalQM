"""NumPy comparisons for general fields, correlations, and time stepping."""

import coupled_spins_numpy_solution as reference
import coupled_spins_torch_solution as solution
import numpy as np
import torch


def test_general_batch_numpy_matches_torch() -> None:
    coupling = np.array([0.0, -0.7, 1.3])
    fields = np.array(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0.3, -0.4, 1], [-0.2, 0.7, 0.1]],
            [[0, 0, 1], [0, 0, -0.4]],
        ],
        dtype=np.float64,
    )
    initial = reference.normalize_states(
        np.array([[0, 1, 0, 0], [1, 1j, -0.4, 0.7], [1, 0, 0, 1]], dtype=np.complex128)
    )
    times = np.linspace(-1, 9, 41)
    operators = reference.hamiltonians(coupling, fields, gamma=-0.8, hbar=0.7)
    torch_h = solution.hamiltonians(
        torch.from_numpy(coupling), torch.from_numpy(fields), gamma=-0.8, hbar=0.7
    )
    np.testing.assert_allclose(operators, torch_h.numpy(), atol=1e-15, rtol=0)
    result = reference.evolve_constant(initial, operators, times, hbar=0.7)
    torch_result = solution.evolve_constant(
        torch.from_numpy(initial), torch_h, torch.from_numpy(times), hbar=0.7
    )
    np.testing.assert_allclose(
        result.states, torch_result.states.numpy(), atol=3e-14, rtol=0
    )
    np.testing.assert_allclose(
        reference.correlations(result.states, connected=True),
        solution.correlations(torch_result.states, connected=True).numpy(),
        atol=4e-14,
        rtol=0,
    )
    np.testing.assert_allclose(
        reference.product_determinant(result.states),
        solution.product_determinant(torch_result.states).numpy(),
        atol=3e-14,
        rtol=0,
    )
    numpy_cn = reference.propagate_cn(initial, operators, 0.07, 90, 11, hbar=0.7)
    torch_cn = solution.propagate_cn(
        torch.from_numpy(initial), torch_h, 0.07, 90, 11, hbar=0.7
    )
    np.testing.assert_allclose(
        numpy_cn.states, torch_cn.states.numpy(), atol=4e-14, rtol=0
    )


def test_numpy_product_states_and_factorization() -> None:
    first = np.array([[1, 0], [1, 1j]], dtype=np.complex128)
    second = np.array([[0, 1], [1, -1]], dtype=np.complex128)
    states = reference.product_states(first, second)
    np.testing.assert_allclose(
        np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        reference.product_determinant(states), 0, atol=1e-15, rtol=0
    )
    np.testing.assert_allclose(
        reference.correlations(states, connected=True), 0, atol=1e-15, rtol=0
    )
