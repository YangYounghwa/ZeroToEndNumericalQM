"""Independent Kronecker-product reference, with sparse low-energy solves."""

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

type RealArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


def _dimension(num_sites: int, limit: int = 14) -> int:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def local_operator(num_sites: int, site: int, axis: str) -> sparse.csr_matrix:
    """Site zero is the rightmost Kronecker factor (least significant bit)."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def sparse_hamiltonian(
    num_sites: int, *, coupling: float = 1.0, field: float = 1.0
) -> sparse.csr_matrix:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def hamiltonian(
    num_sites: int, *, coupling: float = 1.0, field: float = 1.0
) -> RealArray:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def low_energy(
    num_sites: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    count: int = 2,
    tolerance: float = 1e-12,
) -> tuple[RealArray, RealArray, RealArray]:
    """Sorted Ritz pairs and absolute residual norms; errors are not suppressed."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def expectation(states: ComplexArray, operator: sparse.csr_matrix) -> RealArray:
    """Independent sparse-operator expectation for (..., D) states."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def main() -> None:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


if __name__ == "__main__":
    main()
