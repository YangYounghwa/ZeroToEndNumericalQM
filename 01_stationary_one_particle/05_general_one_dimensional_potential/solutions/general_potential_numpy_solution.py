"""Reusable NumPy solver for real one-dimensional stationary potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
PotentialFunction = Callable[[FloatArray], FloatArray]


@dataclass(frozen=True)
class StationaryResult:
    """Low-energy eigenstates on a finite interior spatial grid."""

    grid: FloatArray
    spacing: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: FloatArray


def _validate_inputs(
    num_points: int,
    num_states: int,
    x_min: float,
    x_max: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if not np.isfinite([x_min, x_max, mass, hbar]).all():
        raise ValueError("domain, mass, and hbar must be finite")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max")
    if mass <= 0.0 or hbar <= 0.0:
        raise ValueError("mass and hbar must be positive")


def make_grid(
    num_points: int,
    x_min: float = -8.0,
    x_max: float = 8.0,
) -> tuple[FloatArray, float]:
    """Return interior points of [x_min, x_max] and their uniform spacing."""
    _validate_inputs(num_points, 1, x_min, x_max, 1.0, 1.0)
    spacing = (x_max - x_min) / (num_points + 1)
    indices = np.arange(1, num_points + 1, dtype=np.float64)
    return x_min + spacing * indices, spacing


def kinetic_energy_matrix(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Construct the second-order finite-difference kinetic operator."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not np.isfinite([spacing, mass, hbar]).all():
        raise ValueError("spacing, mass, and hbar must be finite")
    if spacing <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("spacing, mass, and hbar must be positive")
    diagonal = np.full(num_points, -2.0, dtype=np.float64)
    off_diagonal = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = (
        np.diag(diagonal) + np.diag(off_diagonal, k=1) + np.diag(off_diagonal, k=-1)
    ) / spacing**2
    return -(hbar**2 / (2.0 * mass)) * second_derivative


def evaluate_potential(
    grid: FloatArray,
    potential_function: PotentialFunction,
) -> FloatArray:
    """Evaluate and validate a real, finite potential on the spatial grid."""
    raw_potential = np.asarray(potential_function(grid))
    if raw_potential.shape != grid.shape:
        raise ValueError("potential_function must return one value per grid point")
    if np.iscomplexobj(raw_potential):
        raise ValueError("potential_function must return real values")
    potential = raw_potential.astype(np.float64, copy=False)
    if not np.isfinite(potential).all():
        raise ValueError("potential values must be finite")
    return potential


def build_hamiltonian(
    potential_function: PotentialFunction,
    num_points: int = 300,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Construct H = T + diag(V) for a supplied potential function."""
    _validate_inputs(num_points, 1, x_min, x_max, mass, hbar)
    grid, spacing = make_grid(num_points, x_min, x_max)
    kinetic = kinetic_energy_matrix(num_points, spacing, mass, hbar)
    potential = evaluate_potential(grid, potential_function)
    return grid, spacing, potential, kinetic + np.diag(potential)


def normalize_wavefunctions(
    wavefunctions: FloatArray,
    spacing: float,
) -> FloatArray:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = np.sqrt(spacing * np.sum(np.abs(wavefunctions) ** 2, axis=0))
    return wavefunctions / norms


def solve_stationary(
    potential_function: PotentialFunction,
    num_points: int = 300,
    num_states: int = 6,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> StationaryResult:
    """Solve for the lowest states of an arbitrary real potential."""
    _validate_inputs(num_points, num_states, x_min, x_max, mass, hbar)
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        potential_function, num_points, x_min, x_max, mass, hbar
    )
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    selected = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return StationaryResult(
        grid,
        spacing,
        potential,
        energies[:num_states],
        selected,
        hamiltonian,
    )


def expectation_position(result: StationaryResult) -> FloatArray:
    """Return <x> for every stored eigenstate."""
    density = np.abs(result.wavefunctions) ** 2
    return result.spacing * np.sum(result.grid[:, np.newaxis] * density, axis=0)


def residual_norms(result: StationaryResult) -> FloatArray:
    """Return discrete L2 norms of H psi_n - E_n psi_n."""
    residuals = (
        result.hamiltonian @ result.wavefunctions
        - result.wavefunctions * result.energies[np.newaxis, :]
    )
    return np.sqrt(result.spacing * np.sum(np.abs(residuals) ** 2, axis=0))


def shifted_harmonic_potential(
    grid: FloatArray,
    center: float = 0.75,
    angular_frequency: float = 1.25,
    mass: float = 1.0,
    offset: float = 0.4,
) -> FloatArray:
    """Return a shifted harmonic potential used as an analytical reference."""
    if angular_frequency <= 0.0 or mass <= 0.0:
        raise ValueError("angular_frequency and mass must be positive")
    return offset + 0.5 * mass * angular_frequency**2 * (grid - center) ** 2


def analytical_harmonic_energies(
    num_states: int,
    angular_frequency: float = 1.25,
    hbar: float = 1.0,
    offset: float = 0.4,
) -> FloatArray:
    """Return exact energies for the shifted harmonic reference potential."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if angular_frequency <= 0.0 or hbar <= 0.0:
        raise ValueError("angular_frequency and hbar must be positive")
    quantum_numbers = np.arange(num_states, dtype=np.float64)
    return offset + hbar * angular_frequency * (quantum_numbers + 0.5)


def main() -> None:
    potential: PotentialFunction = shifted_harmonic_potential
    result = solve_stationary(potential)
    exact = analytical_harmonic_energies(len(result.energies))
    positions = expectation_position(result)
    print("state  numerical energy  exact energy     abs error       <x>")
    for state, energy in enumerate(result.energies):
        error = abs(energy - exact[state])
        print(
            f"{state:>5}  {energy:>16.8f}  {exact[state]:>12.8f}  "
            f"{error:>12.3e}  {positions[state]:>9.5f}"
        )


if __name__ == "__main__":
    main()
