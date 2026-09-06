"""Starter template for NumPy/SciPy free Gaussian propagation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EvolutionResult:
    grid: FloatArray
    spacing: float
    times: FloatArray
    wavefunctions: ComplexArray
    hamiltonian: csr_matrix


def make_grid(
    num_points: int,
    x_min: float = -20.0,
    x_max: float = 20.0,
) -> tuple[FloatArray, float]:
    # TODO: Return the unknown interior points and spacing.
    raise NotImplementedError


def gaussian_wave_packet(
    grid: FloatArray,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
) -> ComplexArray:
    # TODO: Construct a complex Gaussian with probability width sigma.
    raise NotImplementedError


def propagate_crank_nicolson(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    # TODO: Factorize the left matrix once and propagate all steps.
    raise NotImplementedError


def solve_free_packet() -> EvolutionResult:
    # TODO: Assemble and run the default free-packet problem.
    raise NotImplementedError


def sparse_exponential_state(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> ComplexArray:
    # TODO: Normalize and apply scipy.sparse.linalg.expm_multiply.
    # Do not convert the Hamiltonian or propagator to a dense array.
    raise NotImplementedError


def main() -> None:
    # TODO: Report norm, energy, center, and width changes.
    raise NotImplementedError


if __name__ == "__main__":
    main()
