from math import log2

import bell_states_numpy_solution as solution
import numpy as np
import pytest
from numpy.typing import NDArray


def test_numpy_bell_and_classical_mixture() -> None:
    states = solution.bell_states()
    np.testing.assert_allclose(states @ states.conj().T, np.eye(4), atol=1e-14)
    rho = solution.pure_density(states)
    np.testing.assert_allclose(solution.entanglement_entropy(states), 1, atol=1e-14)
    np.testing.assert_allclose(solution.purity(rho), 1, atol=1e-14)
    np.testing.assert_allclose(solution.entropy(rho), 0, atol=1e-13)
    expected = np.array(
        [np.diag(row) for row in [[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]]]
    )
    np.testing.assert_allclose(solution.pauli_correlations(rho), expected, atol=1e-14)
    mixture = solution.mixed_density(
        np.eye(4, dtype=np.complex128)[[0, 3]], np.array([0.5, 0.5])
    )
    assert float(solution.entropy(mixture)) == pytest.approx(1)
    assert float(solution.purity(mixture)) == pytest.approx(0.5)
    for keep in (0, 1):
        np.testing.assert_allclose(
            solution.partial_trace(mixture, keep=keep), np.eye(2) / 2
        )


def test_numpy_partial_trace_with_complex_off_diagonal_entries() -> None:
    a = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    b = np.array([1, 2j, -2], dtype=np.complex128) / 3
    rho = solution.pure_density(np.kron(a, b))
    np.testing.assert_allclose(
        solution.partial_trace(rho, (2, 3)), solution.pure_density(a), atol=1e-14
    )
    np.testing.assert_allclose(
        solution.partial_trace(rho, (2, 3), keep=1),
        solution.pure_density(b),
        atol=1e-14,
    )
    assert abs(float(solution.entanglement_entropy(np.kron(a, b), (2, 3)))) < 1e-13


def test_numpy_small_eigenvalue_entropy() -> None:
    for small in (0, 1e-14, 0.5):
        rho = np.diag(np.array([1 - small, small], dtype=np.complex128))
        expected = -sum(p * log2(p) for p in (small, 1 - small) if p > 0)
        assert float(solution.entropy(rho)) == pytest.approx(expected, abs=1e-18)


@pytest.mark.parametrize(
    "rho",
    [
        np.eye(2, dtype=np.complex128),
        np.diag(np.array([1.1, -0.1], dtype=np.complex128)),
        np.array([[1, 1j], [0, 0]], dtype=np.complex128),
    ],
)
def test_numpy_rejects_invalid_density(rho: NDArray[np.complex128]) -> None:
    with pytest.raises(ValueError, match="rho must"):
        solution.entropy(rho)
