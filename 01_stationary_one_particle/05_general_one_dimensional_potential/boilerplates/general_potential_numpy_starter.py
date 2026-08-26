"""Starter template for the reusable NumPy stationary solver."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PotentialFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class StationaryResult:
    grid: FloatArray
    spacing: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def make_grid(
    num_points: int,
    x_min: float = -8.0,
    x_max: float = 8.0,
) -> tuple[FloatArray, float]:
    # TODO: Validate the domain and return only the unknown interior points.
    raise NotImplementedError


def kinetic_energy_matrix(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    # TODO: Construct -hbar^2 D2 / (2m).
    raise NotImplementedError


def evaluate_potential(
    grid: FloatArray,
    potential_function: PotentialFunction,
) -> FloatArray:
    # TODO: Evaluate the callable and validate shape, reality, and finiteness.
    raise NotImplementedError


def solve_stationary(
    potential_function: PotentialFunction,
    num_points: int = 300,
    num_states: int = 6,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> StationaryResult:
    # TODO: Build H, diagonalize, select states, and normalize the columns.
    raise NotImplementedError


def expectation_position(result: StationaryResult) -> FloatArray:
    # TODO: Calculate <x> for every stored state.
    raise NotImplementedError


def residual_norms(result: StationaryResult) -> FloatArray:
    # TODO: Calculate the discrete norm of H psi - E psi for each state.
    raise NotImplementedError


def main() -> None:
    # TODO: Define a shifted harmonic potential and compare with exact results.
    raise NotImplementedError


if __name__ == "__main__":
    main()
