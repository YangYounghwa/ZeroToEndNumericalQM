"""Starter template for a sparse 3D stationary solver."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Potential3D = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]


def build_hamiltonian(
    potential_function: Potential3D,
    num_x: int = 14,
    num_y: int = 13,
    num_z: int = 12,
) -> tuple[object, ...]:
    # TODO: Build the three-term sparse Kronecker sum and sampled potential.
    raise NotImplementedError


def solve_stationary_3d() -> object:
    # TODO: Request low eigenpairs, normalize, and reshape to (Nz,Ny,Nx,state).
    raise NotImplementedError


def dense_hamiltonian_bytes(num_x: int, num_y: int, num_z: int) -> int:
    # TODO: Calculate float64 dense storage before attempting allocation.
    raise NotImplementedError


def main() -> None:
    # TODO: Compare oscillator energies and report a memory estimate.
    raise NotImplementedError


if __name__ == "__main__":
    main()
