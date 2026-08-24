"""NumPy reference solution of the one-dimensional infinite square well."""

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


def _validate_inputs(
    num_points: int,
    num_states: int,
    length: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if length <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("length, mass, and hbar must be positive")


def make_grid(num_points: int, length: float = 1.0) -> tuple[FloatArray, float]:
    """Return the interior grid and uniform grid spacing."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if length <= 0:
        raise ValueError("length must be positive")

    spacing = length / (num_points + 1)
    grid = spacing * np.arange(1, num_points + 1, dtype=np.float64)
    return grid, spacing


def build_hamiltonian(
    num_points: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray]:
    """Construct the finite-difference Hamiltonian on the interior grid."""
    _validate_inputs(num_points, 1, length, mass, hbar)
    grid, spacing = make_grid(num_points, length)

    main_diagonal = np.full(num_points, -2.0, dtype=np.float64)
    off_diagonal = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = (
        np.diag(main_diagonal)
        + np.diag(off_diagonal, k=1)
        + np.diag(off_diagonal, k=-1)
    ) / spacing**2
    hamiltonian = -(hbar**2 / (2.0 * mass)) * second_derivative
    return grid, spacing, hamiltonian


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = np.sqrt(spacing * np.sum(np.abs(wavefunctions) ** 2, axis=0))
    return wavefunctions / norms


def solve_infinite_well(
    num_points: int = 200,
    num_states: int = 4,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> InfiniteWellResult:
    """Solve for the lowest-energy states of an infinite square well."""
    _validate_inputs(num_points, num_states, length, mass, hbar)
    grid, spacing, hamiltonian = build_hamiltonian(num_points, length, mass, hbar)
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)

    return InfiniteWellResult(
        grid=grid,
        spacing=spacing,
        energies=energies[:num_states],
        wavefunctions=wavefunctions,
        hamiltonian=hamiltonian,
    )


def analytical_energies(
    num_states: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return exact energies for quantum numbers 1 through num_states."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if length <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("length, mass, and hbar must be positive")
    quantum_numbers = np.arange(1, num_states + 1, dtype=np.float64)
    return quantum_numbers**2 * np.pi**2 * hbar**2 / (2.0 * mass * length**2)


def main() -> None:
    result = solve_infinite_well()
    exact = analytical_energies(len(result.energies))

    print("state  numerical energy  analytical energy  relative error")
    for state, (numerical, analytical) in enumerate(
        zip(result.energies, exact, strict=True), start=1
    ):
        relative_error = abs(numerical - analytical) / analytical
        print(
            f"{state:>5}  {numerical:>16.8f}  {analytical:>17.8f}  "
            f"{relative_error:>14.3e}"
        )


if __name__ == "__main__":
    main()
