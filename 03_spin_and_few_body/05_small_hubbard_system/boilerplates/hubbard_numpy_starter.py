"""Independent explicit matrix and spectral evolution for the Hubbard dimer."""

import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def hamiltonian(
    *, hopping: float = 1.0, interaction: float = 0.0, bias: float = 0.0
) -> ComplexArray:
    """Basis D0, up/down, down/up, D1; ascending creation-operator order."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def _states(states: ComplexArray) -> None:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def evolve(h: ComplexArray, states: ComplexArray, times: RealArray) -> ComplexArray:
    """Return (T, ..., 4) states using spectral phases; hbar=1."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def double_occupancy(states: ComplexArray) -> RealArray:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def site_occupations(states: ComplexArray) -> RealArray:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def main() -> None:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


if __name__ == "__main__":
    main()
