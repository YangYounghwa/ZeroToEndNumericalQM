"""NumPy reference solution of the one-dimensional harmonic oscillator."""

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


def _validate_inputs(
    num_points: int,
    num_states: int,
    x_max: float,
    mass: float,
    omega: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if x_max <= 0 or mass <= 0 or omega <= 0 or hbar <= 0:
        raise ValueError("x_max, mass, omega, and hbar must be positive")


def make_grid(num_points: int, x_max: float = 8.0) -> tuple[FloatArray, float]:
    """Return interior points of [-x_max, x_max] and their spacing."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if x_max <= 0:
        raise ValueError("x_max must be positive")

    spacing = 2.0 * x_max / (num_points + 1)
    grid = -x_max + spacing * np.arange(1, num_points + 1, dtype=np.float64)
    return grid, spacing


def harmonic_potential(
    grid: FloatArray,
    mass: float = 1.0,
    omega: float = 1.0,
) -> FloatArray:
    """Evaluate V(x) = m omega^2 x^2 / 2 on the grid."""
    if mass <= 0 or omega <= 0:
        raise ValueError("mass and omega must be positive")
    return 0.5 * mass * omega**2 * grid**2


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Construct the finite-difference harmonic-oscillator Hamiltonian."""
    _validate_inputs(num_points, 1, x_max, mass, omega, hbar)
    grid, spacing = make_grid(num_points, x_max)

    main_diagonal = np.full(num_points, -2.0, dtype=np.float64)
    off_diagonal = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = (
        np.diag(main_diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    ) / spacing**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = harmonic_potential(grid, mass, omega)
    hamiltonian = kinetic + np.diag(potential)
    return grid, spacing, potential, hamiltonian


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = np.sqrt(spacing * np.sum(np.abs(wavefunctions) ** 2, axis=0))
    return wavefunctions / norms


def solve_harmonic_oscillator(
    num_points: int = 200,
    num_states: int = 4,
    x_max: float = 8.0,
    mass: float = 1.0,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> HarmonicOscillatorResult:
    """Solve for the lowest-energy harmonic-oscillator states."""
    _validate_inputs(num_points, num_states, x_max, mass, omega, hbar)
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, x_max, mass, omega, hbar
    )
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return HarmonicOscillatorResult(
        grid=grid,
        spacing=spacing,
        potential=potential,
        energies=energies[:num_states],
        wavefunctions=wavefunctions,
        hamiltonian=hamiltonian,
    )


def analytical_energies(
    num_states: int,
    omega: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return exact energies for quantum numbers 0 through num_states - 1."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if omega <= 0 or hbar <= 0:
        raise ValueError("omega and hbar must be positive")
    quantum_numbers = np.arange(num_states, dtype=np.float64)
    return hbar * omega * (quantum_numbers + 0.5)


def expectation_x_power(
    result: HarmonicOscillatorResult,
    power: int,
) -> FloatArray:
    """Return <x^power> for every eigenstate stored in a result."""
    if power < 0:
        raise ValueError("power must be nonnegative")
    density = np.abs(result.wavefunctions) ** 2
    return result.spacing * np.sum(result.grid[:, None] ** power * density, axis=0)


def main() -> None:
    result = solve_harmonic_oscillator()
    exact = analytical_energies(len(result.energies))

    print("state  numerical energy  analytical energy  relative error")
    for state, (numerical, analytical) in enumerate(
        zip(result.energies, exact, strict=True)
    ):
        relative_error = abs(numerical - analytical) / analytical
        print(
            f"{state:>5}  {numerical:>16.8f}  {analytical:>17.8f}  "
            f"{relative_error:>14.3e}"
        )


if __name__ == "__main__":
    main()
