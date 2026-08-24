"""Starter template for the NumPy symmetric-double-well solver."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DoubleWellResult:
    grid: FloatArray
    spacing: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def make_grid(num_points: int, x_max: float = 6.0) -> tuple[FloatArray, float]:
    # TODO: Return N interior points and spacing.
    raise NotImplementedError


def double_well_potential(
    grid: FloatArray,
    separation: float = 1.5,
    barrier_height: float = 8.0,
) -> FloatArray:
    # TODO: Implement V_b * ((x / d)**2 - 1)**2.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    # TODO: Construct T + diag(V).
    raise NotImplementedError


def solve_double_well(
    num_points: int = 300,
    num_states: int = 6,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> DoubleWellResult:
    # TODO: Validate, diagonalize, select states, and normalize columns.
    raise NotImplementedError


def tunneling_splitting(result: DoubleWellResult) -> float:
    # TODO: Return the difference between the first two energies.
    raise NotImplementedError


def localized_pair(result: DoubleWellResult) -> tuple[FloatArray, FloatArray]:
    # TODO: Combine the two lowest parity eigenstates and determine left/right.
    raise NotImplementedError


def main() -> None:
    # TODO: Print energies, parity, left probability, and splitting.
    raise NotImplementedError


if __name__ == "__main__":
    main()
