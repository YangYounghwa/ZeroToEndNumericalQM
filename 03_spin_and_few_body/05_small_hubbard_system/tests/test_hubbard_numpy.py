from math import sqrt

import hubbard_numpy_solution as solution
import numpy as np
import pytest


@pytest.mark.parametrize("u", [-8.0, 0.0, 1.0, 4.0, 20.0])
def test_exact_spectrum_and_double_occupancy(u: float) -> None:
    h = solution.hamiltonian(interaction=u)
    energies, vectors = np.linalg.eigh(h)
    root = sqrt(u * u + 16)
    np.testing.assert_allclose(
        energies, sorted([(u - root) / 2, 0, u, (u + root) / 2]), atol=1e-13
    )
    assert float(solution.double_occupancy(vectors[:, 0])) == pytest.approx(
        (1 - u / root) / 2
    )
    np.testing.assert_allclose(h @ vectors, vectors * energies, atol=1e-13)
    np.testing.assert_allclose(vectors.conj().T @ vectors, np.eye(4), atol=1e-13)


def test_atomic_and_noninteracting_limits() -> None:
    np.testing.assert_array_equal(
        solution.hamiltonian(hopping=0, interaction=3, bias=2), np.diag([5, 0, 0, 1])
    )
    one = np.linalg.eigvalsh(np.array([[0.7, -1.2], [-1.2, -0.7]]))
    expected = sorted(float(a + b) for a in one for b in one)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(solution.hamiltonian(hopping=1.2, bias=1.4)),
        expected,
        atol=1e-13,
    )


def test_singlet_dynamics() -> None:
    times = np.linspace(0, 4, 21)
    singlet = np.array([0, 1, -1, 0], dtype=np.complex128) / sqrt(2)
    history = solution.evolve(solution.hamiltonian(interaction=3), singlet, times)
    np.testing.assert_allclose(
        solution.double_occupancy(history),
        16 / 25 * np.sin(5 * times / 2) ** 2,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        solution.site_occupations(history), np.ones((21, 2)), atol=1e-13
    )


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        solution.hamiltonian(interaction=float("nan"))
    with pytest.raises(ValueError, match="normalized"):
        solution.double_occupancy(np.zeros(4, dtype=np.complex128))
    h = solution.hamiltonian()
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    with pytest.raises(ValueError, match="nonempty"):
        solution.evolve(h, state, np.array([], dtype=np.float64))
    h[0, 1] = 7
    with pytest.raises(ValueError, match="Hermitian"):
        solution.evolve(h, state, np.array([0.0]))
