"""Starter template for a sparse 2D stationary solver."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

FloatArray = NDArray[np.float64]
Potential2D = Callable[[FloatArray, FloatArray], FloatArray]


def kinetic_1d(
    num_points: int, spacing: float, mass: float = 1.0, hbar: float = 1.0
) -> csr_matrix:
    # TODO: Return one sparse tridiagonal kinetic matrix.
    raise NotImplementedError


def build_hamiltonian(
    potential_function: Potential2D,
    num_x: int = 32,
    num_y: int = 30,
) -> tuple[FloatArray, FloatArray, float, float, FloatArray, csr_matrix]:
    # TODO: Build a C-order Kronecker-sum Hamiltonian.
    raise NotImplementedError


def solve_stationary_2d() -> object:
    # TODO: Request low eigenpairs, sort, normalize, and reshape them.
    raise NotImplementedError


def main() -> None:
    # TODO: Compare anisotropic-oscillator energies with exact sums.
    raise NotImplementedError


if __name__ == "__main__":
    main()
