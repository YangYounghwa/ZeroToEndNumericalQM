"""Starter template for the NumPy harmonic-oscillator solver.

Complete the TODO sections in a copy of this file. Keep this reusable starter
unchanged so another learner can start from the same point.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HarmonicOscillatorResult:
    """Numerical eigenstates on a finite interior spatial grid."""

    grid: FloatArray
    spacing: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def make_grid(num_points: int, x_max: float = 8.0) -> tuple[FloatArray, float]:
    """Return interior points of [-x_max, x_max] and their spacing."""
    # TODO: Validate num_points and x_max.
    # TODO: Compute h = 2 * x_max / (num_points + 1).
    # TODO: Return only the num_points interior positions.
    raise NotImplementedError


def harmonic_potential(
    grid: FloatArray,
    mass: float = 1.0,
    omega: float = 1.0,
) -> FloatArray:
    """Evaluate the potential on the grid."""
    # TODO: Return 0.5 * mass * omega**2 * grid**2.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Build the finite-difference harmonic-oscillator Hamiltonian."""
    # TODO: Reuse the centered second derivative from Chapter 1.
    # TODO: Construct the kinetic-energy matrix.
    # TODO: Add the potential only to the main diagonal.
    raise NotImplementedError


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns with a discrete spatial integral."""
    # TODO: Compute one grid-weighted norm per column.
    raise NotImplementedError


def solve_harmonic_oscillator(
    num_points: int = 200,
    num_states: int = 4,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> HarmonicOscillatorResult:
    """Calculate the lowest-energy eigenstates."""
    # TODO: Validate inputs and build H.
    # TODO: Use a Hermitian eigensolver.
    # TODO: Keep and normalize exactly num_states columns.
    raise NotImplementedError


def analytical_energies(
    num_states: int,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return exact energies for n = 0, ..., num_states - 1."""
    # TODO: Implement hbar * omega * (n + 0.5).
    # The quantum number starts at zero for this problem.
    raise NotImplementedError


def expectation_x_power(
    result: HarmonicOscillatorResult,
    power: int,
) -> FloatArray:
    """Return <x^power> for all stored states."""
    # TODO: Multiply the density by grid**power and integrate over axis 0.
    raise NotImplementedError


def main() -> None:
    """Compare numerical and analytical energies."""
    # TODO: Solve the problem and print a compact comparison table.
    raise NotImplementedError


if __name__ == "__main__":
    main()
