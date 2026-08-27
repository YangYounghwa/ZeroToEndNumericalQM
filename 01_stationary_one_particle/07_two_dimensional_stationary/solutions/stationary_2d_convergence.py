"""Grid convergence for the two-dimensional stationary solver."""

from stationary_2d_numpy_solution import (
    analytical_oscillator_energies,
    anisotropic_oscillator_potential,
    coupled_quartic_potential,
    solve_stationary_2d,
)


def separable_validation_convergence() -> None:
    """Validate grid convergence against an analytical spectrum."""
    exact_ground = float(analytical_oscillator_energies(1)[0])
    print("Separable oscillator validation")
    print("Nx=Ny   spacing x    numerical E0    absolute error")
    for num_points in (12, 18, 26, 36):
        result = solve_stationary_2d(
            anisotropic_oscillator_potential,
            num_x=num_points,
            num_y=num_points,
            num_states=1,
            x_min=-7.0,
            x_max=7.0,
            y_min=-7.0,
            y_max=7.0,
        )
        error = abs(float(result.energies[0]) - exact_ground)
        print(
            f"{num_points:>5}   {result.spacing_x:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>14.3e}"
        )


def nonseparable_grid_convergence() -> None:
    """Compare the coupled problem with a refined numerical reference."""
    reference = solve_stationary_2d(
        coupled_quartic_potential,
        num_x=64,
        num_y=58,
        num_states=1,
    )
    reference_energy = float(reference.energies[0])
    print("\nNonseparable coupled-quartic convergence")
    print("Nx   Ny    numerical E0    refined-reference error")
    for num_x, num_y in ((14, 12), (20, 18), (28, 26), (40, 36)):
        result = solve_stationary_2d(
            coupled_quartic_potential,
            num_x=num_x,
            num_y=num_y,
            num_states=1,
        )
        error = abs(float(result.energies[0]) - reference_energy)
        print(f"{num_x:>2}  {num_y:>3}  {result.energies[0]:>14.8f}  {error:>23.3e}")


def main() -> None:
    separable_validation_convergence()
    nonseparable_grid_convergence()


if __name__ == "__main__":
    main()
