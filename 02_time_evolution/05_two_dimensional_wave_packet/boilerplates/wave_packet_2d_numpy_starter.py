"""Starter: two-dimensional wave packets. Complete a copy in workbench/."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class Grid2D:
    x: FloatArray
    y: FloatArray
    kx: FloatArray
    ky: FloatArray
    dx: float
    dy: float

    @property
    def area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class Evolution2D:
    grid: Grid2D
    kinetic: FloatArray
    potential: FloatArray
    times: FloatArray
    wavefunctions: ComplexArray


def make_grid(
    nx: int = 96, ny: int = 80, half_x: float = 10.0, half_y: float = 8.0
) -> Grid2D:
    """TODO: Build independent endpoint-excluded periodic axes and unshifted angular wave numbers."""
    raise NotImplementedError


def kinetic_energy(grid: Grid2D, mass: float = 1.0, hbar: float = 1.0) -> FloatArray:
    """TODO: Broadcast squared ky and kx, with hbar squared over twice the mass."""
    raise NotImplementedError


def stiffness_matrix(
    omega_x: float = 1.0, omega_y: float = 1.3, coupling: float = 0.35
) -> FloatArray:
    """TODO: Construct the symmetric matrix and enforce a stable positive-definite oscillator."""
    raise NotImplementedError


def normal_modes(stiffness: FloatArray) -> tuple[FloatArray, FloatArray]:
    """TODO: Diagonalize the positive symmetric matrix; take square roots of its eigenvalues."""
    raise NotImplementedError


def coupled_potential(
    grid: Grid2D, stiffness: FloatArray, mass: float = 1.0
) -> FloatArray:
    """TODO: Broadcast x and y to construct m*r.T*K*r/2 with the xy term included."""
    raise NotImplementedError


def coherent_packet(
    grid: Grid2D,
    stiffness: FloatArray,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    """TODO: Construct the Gaussian quadratic form from the positive matrix square root, then add the momentum phase."""
    raise NotImplementedError


def free_gaussian(
    grid: Grid2D,
    time: float = 0.0,
    center: tuple[float, float] = (-3.0, 1.0),
    sigma: tuple[float, float] = (1.0, 1.4),
    momentum: tuple[float, float] = (0.8, -0.4),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    """TODO: Multiply the two analytical Gaussian factors, including spreading and phase."""
    raise NotImplementedError


def normalize_batch(states: ComplexArray, area: float) -> ComplexArray:
    """TODO: Sum density over spatial axes only, including dx*dy; reject zero or nonfinite norms."""
    raise NotImplementedError


def propagate_batch(
    initial_states: ComplexArray,
    kinetic: FloatArray,
    potential: FloatArray,
    area: float,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """TODO: Apply symmetric FFT2 steps on axes (0,1), keeping the batch axis unchanged and saving requested times."""
    raise NotImplementedError


def solve_coupled(
    nx: int = 96,
    ny: int = 80,
    half_x: float = 10.0,
    half_y: float = 8.0,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
) -> Evolution2D:
    """TODO: Build the default coupled problem, propagate its initial packet, and package a single-packet result."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise and write long experiment output to Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
