"""Starter: complete a copy in workbench/. See the chapter coding hints."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

type Statistics = Literal["distinguishable", "boson", "fermion"]
type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def basis_pairs(num_sites: int, statistics: Statistics) -> list[tuple[int, int]]:
    """TODO: Implement basis_pairs using the chapter's derivation."""
    raise NotImplementedError


def single_particle_hamiltonian(
    num_sites: int, *, hopping: float = 1.0, onsite: RealArray | None = None
) -> ComplexArray:
    """TODO: Nearest-neighbor open chain with a real site potential."""
    raise NotImplementedError


def occupation_basis(num_sites: int, statistics: Statistics) -> list[tuple[int, ...]]:
    """TODO: Site counts for each basis pair; labeled pairs can have equal counts."""
    raise NotImplementedError


def two_particle_hamiltonian(h: ComplexArray, statistics: Statistics) -> ComplexArray:
    """TODO: Build matrix elements by labeled moves or creation/annihilation rules."""
    raise NotImplementedError


def localized_state(
    num_sites: int, statistics: Statistics, pair: tuple[int, int]
) -> ComplexArray:
    """TODO: Implement localized_state using the chapter's derivation."""
    raise NotImplementedError


def evolve(
    hamiltonian: ComplexArray,
    states: ComplexArray,
    times: RealArray,
    *,
    hbar: float = 1.0,
) -> ComplexArray:
    """TODO: Independent spectral evolution; output has shape (T, ..., D)."""
    raise NotImplementedError


def site_occupations(
    states: ComplexArray, num_sites: int, statistics: Statistics
) -> RealArray:
    """TODO: Implement site_occupations using the chapter's derivation."""
    raise NotImplementedError


def coincidence_probability(
    states: ComplexArray, num_sites: int, statistics: Statistics
) -> RealArray:
    """TODO: Implement coincidence_probability using the chapter's derivation."""
    raise NotImplementedError


def main() -> None:
    """TODO: Implement main using the chapter's derivation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
