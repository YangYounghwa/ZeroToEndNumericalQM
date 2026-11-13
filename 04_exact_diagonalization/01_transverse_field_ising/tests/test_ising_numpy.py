from math import pi, sin

import ising_numpy_solution as solution
import numpy as np
import pytest
from scipy import sparse


@pytest.mark.parametrize("sites", [1, 2, 4])
def test_exact_limits(sites: int) -> None:
    field = 0.7
    expected = sorted(
        -field * (sites - 2 * mask.bit_count()) for mask in range(2**sites)
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(solution.hamiltonian(sites, coupling=0, field=field)),
        expected,
        atol=1e-13,
    )
    expected_diagonal = []
    for mask in range(2**sites):
        walls = sum(
            ((mask >> i) & 1) != ((mask >> (i + 1)) & 1) for i in range(sites - 1)
        )
        expected_diagonal.append(-(sites - 1) + 2 * walls)
    np.testing.assert_array_equal(
        solution.hamiltonian(sites, field=0), np.diag(expected_diagonal)
    )


def test_local_bit_order() -> None:
    state = np.zeros(8)
    state[0] = 1
    for site in range(3):
        flipped = solution.local_operator(3, site, "x") @ state
        assert int(np.argmax(flipped)) == 1 << site
    np.testing.assert_array_equal(
        solution.local_operator(3, 0, "z").diagonal(), [1, -1, 1, -1, 1, -1, 1, -1]
    )


@pytest.mark.parametrize("sites", [2, 5, 8])
def test_sparse_critical_spectrum(sites: int) -> None:
    energies, vectors, residuals = solution.low_energy(sites)
    angle = pi / (4 * sites + 2)
    assert energies[0] == pytest.approx(1 - 1 / sin(angle), abs=2e-12)
    assert energies[1] - energies[0] == pytest.approx(4 * sin(angle), abs=2e-12)
    assert np.max(residuals) < 1e-10
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(2), atol=1e-13)
    if sites <= 5:
        np.testing.assert_allclose(
            energies, np.linalg.eigvalsh(solution.hamiltonian(sites))[:2], atol=1e-12
        )


def test_sparse_storage() -> None:
    h = solution.sparse_hamiltonian(6)
    assert sparse.isspmatrix_csr(h)
    assert h.nnz <= 7 * 64
    assert (h - h.T).nnz == 0


def test_invalid_inputs() -> None:
    for sites in (0, 15, True):
        with pytest.raises(ValueError, match="num_sites"):
            solution.sparse_hamiltonian(sites)
    with pytest.raises(ValueError, match="finite"):
        solution.hamiltonian(2, coupling=float("inf"))
    with pytest.raises(ValueError, match="count"):
        solution.low_energy(2, count=4)
    with pytest.raises(ValueError, match="tolerance"):
        solution.low_energy(2, tolerance=0)
    with pytest.raises(ValueError, match="normalized"):
        solution.expectation(
            np.zeros(2, dtype=np.complex128), solution.local_operator(1, 0, "z")
        )
    with pytest.raises(ValueError, match="Hermitian"):
        solution.expectation(
            np.array([1, 0], dtype=np.complex128),
            sparse.csr_matrix([[0.0, 1.0], [0.0, 0.0]]),
        )
