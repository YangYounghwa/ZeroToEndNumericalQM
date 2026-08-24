"""Starter template for the NumPy infinite-square-well solver.

Complete the TODO sections in a copy of this file. Keep this reusable starter
unchanged so another learner can start from the same point.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class InfiniteWellResult:
    """Numerical eigenstates on the interior spatial grid."""

    grid: FloatArray
    spacing: float
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def make_grid(num_points: int, length: float = 1.0) -> tuple[FloatArray, float]:
    """Return the interior grid and its spacing."""
    # TODO: Validate num_points and length.
    # TODO: Compute h = L / (N + 1).
    # TODO: Return the N interior points and h.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray]:
    """Build the finite-difference Hamiltonian."""
    # TODO: Validate the physical parameters.
    # TODO: Create the interior grid.
    # TODO: Construct the tridiagonal second-derivative matrix.
    # TODO: Multiply it by -hbar**2 / (2 * mass).
    raise NotImplementedError


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns with a discrete spatial integral."""
    # TODO: Compute one norm per column, including the grid spacing.
    raise NotImplementedError


def solve_infinite_well(
    num_points: int = 200,
    num_states: int = 4,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> InfiniteWellResult:
    """Calculate the lowest-energy eigenstates."""
    # TODO: Validate num_states.
    # TODO: Build H and solve it with a Hermitian eigensolver.
    # TODO: Keep and normalize the lowest num_states eigenvectors.
    # TODO: Return an InfiniteWellResult.
    raise NotImplementedError


def analytical_energies(
    num_states: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return the analytical energies for n = 1, ..., num_states."""
    # TODO: Implement E_n = n**2 * pi**2 * hbar**2 / (2 * mass * length**2).
    # Remember that np.arange excludes its stop value.
    raise NotImplementedError


def main() -> None:
    """Compare the numerical and analytical energies."""
    # TODO: Solve the problem and print a compact comparison table.
    raise NotImplementedError


if __name__ == "__main__":
    main()
