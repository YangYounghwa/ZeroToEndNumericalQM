"""NumPy reference solution of the one-dimensional finite square well."""

from collections.abc import Callable
from dataclasses import dataclass
from math import pi, sqrt, tan

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FiniteSquareWellResult:
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
    half_width: float,
    depth: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if x_max <= 0 or half_width <= 0 or depth <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("lengths, depth, mass, and hbar must be positive")
    if half_width >= x_max:
        raise ValueError("half_width must be smaller than x_max")


def make_grid(num_points: int, x_max: float = 8.0) -> tuple[FloatArray, float]:
    """Return interior points of [-x_max, x_max] and their spacing."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    spacing = 2.0 * x_max / (num_points + 1)
    grid = -x_max + spacing * np.arange(1, num_points + 1, dtype=np.float64)
    return grid, spacing


def finite_square_well_potential(
    grid: FloatArray,
    half_width: float = 1.0,
    depth: float = 20.0,
) -> FloatArray:
    """Return -depth inside |x| < half_width and zero outside."""
    if half_width <= 0 or depth <= 0:
        raise ValueError("half_width and depth must be positive")
    return np.where(np.abs(grid) < half_width, -depth, 0.0).astype(np.float64)


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, FloatArray]:
    """Construct the finite-difference finite-well Hamiltonian."""
    _validate_inputs(num_points, 1, x_max, half_width, depth, mass, hbar)
    grid, spacing = make_grid(num_points, x_max)
    main = np.full(num_points, -2.0, dtype=np.float64)
    off = np.ones(num_points - 1, dtype=np.float64)
    second_derivative = (
        np.diag(main) + np.diag(off, k=1) + np.diag(off, k=-1)
    ) / spacing**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = finite_square_well_potential(grid, half_width, depth)
    return grid, spacing, potential, kinetic + np.diag(potential)


def normalize_wavefunctions(wavefunctions: FloatArray, spacing: float) -> FloatArray:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = np.sqrt(spacing * np.sum(np.abs(wavefunctions) ** 2, axis=0))
    return wavefunctions / norms


def solve_finite_square_well(
    num_points: int = 400,
    num_states: int = 8,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FiniteSquareWellResult:
    """Solve for low-energy states, including any box continuum states."""
    _validate_inputs(num_points, num_states, x_max, half_width, depth, mass, hbar)
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, x_max, half_width, depth, mass, hbar
    )
    energies, wavefunctions = np.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return FiniteSquareWellResult(
        grid, spacing, potential, energies[:num_states], wavefunctions, hamiltonian
    )


def bound_state_mask(result: FiniteSquareWellResult) -> NDArray[np.bool_]:
    """Identify states below the outside potential, whose value is zero."""
    return result.energies < 0.0


def probability_inside_well(
    result: FiniteSquareWellResult,
    half_width: float = 1.0,
) -> FloatArray:
    """Return the probability in |x| < half_width for every stored state."""
    inside = np.abs(result.grid) < half_width
    density = np.abs(result.wavefunctions[inside, :]) ** 2
    return result.spacing * np.sum(density, axis=0)


def _bisect(function: Callable[[float], float], left: float, right: float) -> float:
    """Find a bracketed root without adding a SciPy dependency."""
    left_value = function(left)
    right_value = function(right)
    if left_value * right_value >= 0:
        raise ValueError("root is not bracketed")
    for _ in range(100):
        middle = 0.5 * (left + right)
        middle_value = function(middle)
        if left_value * middle_value <= 0:
            right = middle
        else:
            left = middle
            left_value = middle_value
    return 0.5 * (left + right)


def analytical_bound_energies(
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Solve the exact even and odd matching equations for bound energies."""
    if half_width <= 0 or depth <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("half_width, depth, mass, and hbar must be positive")
    z_max = half_width * sqrt(2.0 * mass * depth) / hbar
    epsilon = 1e-10
    roots: list[float] = []

    def tail(z: float) -> float:
        return sqrt(max(z_max**2 - z**2, 0.0))

    def even_equation(z: float) -> float:
        return z * tan(z) - tail(z)

    def odd_equation(z: float) -> float:
        return -z / tan(z) - tail(z)

    interval = 0
    while interval * pi < z_max:
        even_left = interval * pi + epsilon
        even_right = min(interval * pi + pi / 2.0 - epsilon, z_max - epsilon)
        if (
            even_left < even_right
            and even_equation(even_left) * even_equation(even_right) < 0
        ):
            roots.append(_bisect(even_equation, even_left, even_right))

        odd_left = interval * pi + pi / 2.0 + epsilon
        odd_right = min((interval + 1) * pi - epsilon, z_max - epsilon)
        if (
            odd_left < odd_right
            and odd_equation(odd_left) * odd_equation(odd_right) < 0
        ):
            roots.append(_bisect(odd_equation, odd_left, odd_right))
        interval += 1

    energies = [-depth + hbar**2 * z**2 / (2.0 * mass * half_width**2) for z in roots]
    return np.array(sorted(energies), dtype=np.float64)


def main() -> None:
    result = solve_finite_square_well()
    exact = analytical_bound_energies()
    numerical = result.energies[bound_state_mask(result)]
    count = min(len(numerical), len(exact))
    print(f"numerical bound states: {len(numerical)}")
    print("state  numerical energy  matching energy   abs error")
    for state in range(count):
        error = abs(numerical[state] - exact[state])
        print(
            f"{state:>5}  {numerical[state]:>16.8f}  {exact[state]:>15.8f}  "
            f"{error:>10.3e}"
        )


if __name__ == "__main__":
    main()
