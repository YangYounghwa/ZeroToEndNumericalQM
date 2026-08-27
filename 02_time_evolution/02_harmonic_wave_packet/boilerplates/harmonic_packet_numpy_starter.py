"""Starter template for NumPy/SciPy harmonic-packet evolution."""

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
    potential: FloatArray
    hamiltonian: csr_matrix
    angular_frequency: float


def coherent_state(
    grid: FloatArray,
    center: float = 2.0,
    wave_number: float = 0.7,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    # TODO: Use the oscillator ground-state width.
    raise NotImplementedError


def propagate_crank_nicolson(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    # TODO: Reuse one factorization for all time steps.
    raise NotImplementedError


def solve_harmonic_packet() -> EvolutionResult:
    # TODO: Build H = T + V and evolve one period.
    raise NotImplementedError


def main() -> None:
    # TODO: Report conservation, center error, width, and fidelity.
    raise NotImplementedError


if __name__ == "__main__":
    main()
