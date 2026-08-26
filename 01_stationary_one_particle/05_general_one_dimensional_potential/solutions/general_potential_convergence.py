"""Convergence studies for the reusable one-dimensional solver."""

from collections.abc import Callable

import numpy as np
from general_potential_numpy_solution import (
    analytical_harmonic_energies,
    shifted_harmonic_potential,
    solve_stationary,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def grid_convergence() -> None:
    """Measure second-order energy convergence for a smooth potential."""
    exact_ground = float(analytical_harmonic_energies(1)[0])
    print("Grid convergence for the shifted harmonic reference")
    print("points  spacing       ground energy   absolute error")
    for num_points in (50, 100, 200, 400):
        result = solve_stationary(
            shifted_harmonic_potential,
            num_points=num_points,
            num_states=1,
        )
        error = abs(float(result.energies[0]) - exact_ground)
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>14.3e}"
        )


def potential_reuse_study() -> None:
    """Run the same solver with several distinct potential callables."""

    def free_particle(grid: FloatArray) -> FloatArray:
        return np.zeros_like(grid)

    def harmonic(grid: FloatArray) -> FloatArray:
        return 0.5 * grid**2

    def quartic(grid: FloatArray) -> FloatArray:
        return 0.25 * grid**4

    potentials: tuple[tuple[str, Callable[[FloatArray], FloatArray]], ...] = (
        ("free box", free_particle),
        ("harmonic", harmonic),
        ("quartic", quartic),
    )
    print("\nOne interface, several potentials")
    print("potential     E_0             E_1")
    for name, potential in potentials:
        result = solve_stationary(potential, num_points=200, num_states=2)
        print(f"{name:<12}  {result.energies[0]:>12.8f}  {result.energies[1]:>12.8f}")


def main() -> None:
    grid_convergence()
    potential_reuse_study()


if __name__ == "__main__":
    main()
