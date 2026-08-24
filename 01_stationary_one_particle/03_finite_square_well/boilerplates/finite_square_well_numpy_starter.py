"""Starter template for the NumPy finite-square-well solver."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FiniteSquareWellResult:
    grid: FloatArray
    spacing: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def make_grid(num_points: int, x_max: float = 8.0) -> tuple[FloatArray, float]:
    # TODO: Return N interior points; endpoints are known zero values.
    raise NotImplementedError


def finite_square_well_potential(
    grid: FloatArray, half_width: float = 1.0, depth: float = 20.0
) -> FloatArray:
    # TODO: Return -depth inside and zero outside.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    # TODO: Construct T + diag(V).
    raise NotImplementedError


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    # TODO: Normalize every eigenvector column using the grid spacing.
    raise NotImplementedError


def solve_finite_square_well(
    num_points: int = 400,
    num_states: int = 8,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FiniteSquareWellResult:
    # TODO: Validate, diagonalize with np.linalg.eigh, select, and normalize.
    raise NotImplementedError


def bound_state_mask(result: FiniteSquareWellResult) -> NDArray[np.bool_]:
    # TODO: The outside potential is zero. Which energies are bound?
    raise NotImplementedError


def probability_inside_well(
    result: FiniteSquareWellResult, half_width: float = 1.0
) -> FloatArray:
    # TODO: Integrate density only over |x| < half_width.
    raise NotImplementedError


def main() -> None:
    # TODO: Print energies, bound classification, and inside probabilities.
    raise NotImplementedError


if __name__ == "__main__":
    main()
