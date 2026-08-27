"""Starter template for the sparse reduced radial hydrogen solver."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RadialResult:
    grid: FloatArray
    spacing: float
    effective_potential: FloatArray
    energies: FloatArray
    radial_wavefunctions: FloatArray
    hamiltonian: csr_matrix
    angular_momentum: int
    nuclear_charge: float


def make_radial_grid(
    num_points: int,
    r_max: float = 40.0,
) -> tuple[FloatArray, float]:
    # TODO: Return the interior points between the two Dirichlet boundaries.
    raise NotImplementedError


def effective_potential(
    grid: FloatArray,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    # TODO: Add the Coulomb and centrifugal terms.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int = 800,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, csr_matrix]:
    # TODO: Construct the sparse tridiagonal Hamiltonian.
    raise NotImplementedError


def solve_radial_hydrogen(
    num_points: int = 800,
    num_states: int = 3,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> RadialResult:
    # TODO: Request the lowest sparse eigenpairs, sort, and normalize them.
    raise NotImplementedError


def analytical_energies(
    num_states: int,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    # TODO: Return exact hydrogenic energies for n = ell + 1, ell + 2, ...
    raise NotImplementedError


def expectation_radius(result: RadialResult) -> FloatArray:
    # TODO: Calculate <r> from the reduced radial probability |u|^2 dr.
    raise NotImplementedError


def main() -> None:
    # TODO: Compare numerical energies and radii with analytical values.
    raise NotImplementedError


if __name__ == "__main__":
    main()
