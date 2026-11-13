"""Independent Kronecker-product reference, with sparse low-energy solves."""

from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import eigsh

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def _dimension(num_sites: int, limit: int = 14) -> int:
    if (
        isinstance(num_sites, bool)
        or not isinstance(num_sites, int)
        or not 1 <= num_sites <= limit
    ):
        raise ValueError(f"num_sites must be an integer from 1 to {limit}")
    return 1 << num_sites


def local_operator(num_sites: int, site: int, axis: str) -> sparse.csr_matrix:
    """Site zero is the rightmost Kronecker factor (least significant bit)."""
    _dimension(num_sites)
    if isinstance(site, bool) or not isinstance(site, int) or not 0 <= site < num_sites:
        raise ValueError("site must be a valid integer index")
    if axis not in ("x", "z"):
        raise ValueError("axis must be x or z")
    pauli = np.array([[0.0, 1.0], [1.0, 0.0]]) if axis == "x" else np.diag([1.0, -1.0])
    operator = sparse.csr_matrix([[1.0]])
    for index in reversed(range(num_sites)):
        factor = (
            sparse.csr_matrix(pauli) if index == site else sparse.eye(2, format="csr")
        )
        operator = sparse.csr_matrix(sparse.kron(operator, factor, format="csr"))
    return operator


def sparse_hamiltonian(
    num_sites: int, *, coupling: float = 1.0, field: float = 1.0
) -> sparse.csr_matrix:
    size = _dimension(num_sites)
    if not all(isfinite(x) for x in (coupling, field)):
        raise ValueError("coupling and field must be finite")
    h = sparse.csr_matrix((size, size), dtype=np.float64)
    z = [local_operator(num_sites, i, "z") for i in range(num_sites)]
    for i in range(num_sites):
        h = h - field * local_operator(num_sites, i, "x")
    for i in range(num_sites - 1):
        h = h - coupling * (z[i] @ z[i + 1])
    h = sparse.csr_matrix(h)
    h.eliminate_zeros()
    return h


def hamiltonian(
    num_sites: int, *, coupling: float = 1.0, field: float = 1.0
) -> RealArray:
    _dimension(num_sites, limit=10)
    return np.asarray(
        sparse_hamiltonian(num_sites, coupling=coupling, field=field).toarray(),
        dtype=np.float64,
    )


def low_energy(
    num_sites: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    count: int = 2,
    tolerance: float = 1e-12,
) -> tuple[RealArray, RealArray, RealArray]:
    """Sorted Ritz pairs and absolute residual norms; errors are not suppressed."""
    size = _dimension(num_sites)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count < size:
        raise ValueError("count must be an integer in [1, dimension)")
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    h = sparse_hamiltonian(num_sites, coupling=coupling, field=field)
    # Random deterministic start has components in both parity sectors.
    start = np.random.default_rng(731).normal(size=size)
    energies, vectors = eigsh(h, k=count, which="SA", tol=tolerance, v0=start)
    order = np.argsort(energies)
    energies, vectors = energies[order], vectors[:, order]
    residuals = np.linalg.norm(h @ vectors - vectors * energies, axis=0)
    return (
        np.asarray(energies, dtype=np.float64),
        np.asarray(vectors, dtype=np.float64),
        np.asarray(residuals, dtype=np.float64),
    )


def expectation(states: ComplexArray, operator: sparse.csr_matrix) -> RealArray:
    """Independent sparse-operator expectation for (..., D) states."""
    size = operator.shape[0]
    if (
        operator.shape != (size, size)
        or size == 0
        or not np.isfinite(operator.data).all()
    ):
        raise ValueError("operator must be finite, nonempty, and square")
    difference = operator - operator.conj().T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError("operator must be Hermitian")
    if states.ndim < 1 or states.shape[-1] != size or states.size == 0:
        raise ValueError("states must have nonempty shape (..., dimension)")
    if not np.isfinite(states).all() or not np.allclose(
        np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-10, rtol=0
    ):
        raise ValueError("states must be finite and normalized")
    flat = states.reshape(-1, size)
    values = np.sum(flat.conj() * (operator @ flat.T).T, axis=-1).real
    return np.asarray(values.reshape(states.shape[:-1]), dtype=np.float64)


def main() -> None:
    energies, vectors, residuals = low_energy(10)
    print(f"L=10 E0={energies[0]:.12f} gap={energies[1] - energies[0]:.12f}")
    print("Residual norms:", residuals.tolist())
    print(
        "Z0:",
        float(
            expectation(vectors[:, 0].astype(np.complex128), local_operator(10, 0, "z"))
        ),
    )


if __name__ == "__main__":
    main()
