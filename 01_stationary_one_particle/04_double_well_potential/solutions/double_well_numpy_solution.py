"""NumPy reference solution of a symmetric quartic double well."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DoubleWellResult:
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
    separation: float,
    barrier_height: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if x_max <= 0 or separation <= 0 or barrier_height <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("lengths, barrier_height, mass, and hbar must be positive")
    if separation >= x_max:
        raise ValueError("separation must be smaller than x_max")


def make_grid(num_points: int, x_max: float = 6.0) -> tuple[FloatArray, float]:
    """Return interior points of [-x_max, x_max] and their spacing."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    spacing = 2.0 * x_max / (num_points + 1)
    grid = -x_max + spacing * np.arange(1, num_points + 1, dtype=np.float64)
    return grid, spacing


def double_well_potential(
    grid: FloatArray,
    separation: float = 1.5,
    barrier_height: float = 8.0,
) -> FloatArray:
    """Evaluate V_b[(x/d)^2 - 1]^2 with minima at x = +/-d."""
    if separation <= 0 or barrier_height <= 0:
        raise ValueError("separation and barrier_height must be positive")
    return barrier_height * ((grid / separation) ** 2 - 1.0) ** 2


def build_hamiltonian(
    num_points: int,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Construct the finite-difference double-well Hamiltonian."""
    _validate_inputs(num_points, 1, x_max, separation, barrier_height, mass, hbar)
    grid, spacing = make_grid(num_points, x_max)
    main = np.full(num_points, -2.0, dtype=np.float64)
    off = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = (
        np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
    ) / spacing**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = double_well_potential(grid, separation, barrier_height)
    return grid, spacing, potential, kinetic + np.diag(potential)


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = np.sqrt(spacing * np.sum(np.abs(wavefunctions) ** 2, axis=0))
    return wavefunctions / norms


def solve_double_well(
    num_points: int = 300,
    num_states: int = 6,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> DoubleWellResult:
    """Solve for the lowest-energy stationary states."""
    _validate_inputs(
        num_points,
        num_states,
        x_max,
        separation,
        barrier_height,
        mass,
        hbar,
    )
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, x_max, separation, barrier_height, mass, hbar
    )
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return DoubleWellResult(
        grid, spacing, potential, energies[:num_states], wavefunctions, hamiltonian
    )


def tunneling_splitting(result: DoubleWellResult) -> float:
    """Return E_1 - E_0 for the lowest even-odd pair."""
    if len(result.energies) < 2:
        raise ValueError("result must contain at least two states")
    return float(result.energies[1] - result.energies[0])


def probability_left(result: DoubleWellResult) -> FloatArray:
    """Return the probability at x < 0 for every stored stationary state."""
    density = np.abs(result.wavefunctions[result.grid < 0.0, :]) ** 2
    return result.spacing * np.sum(density, axis=0)


def localized_pair(result: DoubleWellResult) -> tuple[FloatArray, FloatArray]:
    """Combine the two lowest parity states into left/right localized states."""
    if result.wavefunctions.shape[1] < 2:
        raise ValueError("result must contain at least two states")
    even = result.wavefunctions[:, 0]
    odd = result.wavefunctions[:, 1]
    left = (even + odd) / np.sqrt(2.0)
    right = (even - odd) / np.sqrt(2.0)
    if np.sum(np.abs(left[result.grid < 0.0]) ** 2) < np.sum(
        np.abs(right[result.grid < 0.0]) ** 2
    ):
        left, right = right, left
    return left, right


def main() -> None:
    result = solve_double_well()
    print("state  energy          parity  probability left")
    left_probabilities = probability_left(result)
    for state, energy in enumerate(result.energies):
        parity = "even" if state % 2 == 0 else "odd"
        print(
            f"{state:>5}  {energy:>14.8f}  {parity:>6}  "
            f"{left_probabilities[state]:>16.8f}"
        )
    print(f"\nground-pair tunneling splitting: {tunneling_splitting(result):.8e}")


if __name__ == "__main__":
    main()
