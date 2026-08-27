"""Grid convergence for the two-dimensional stationary solver."""

from stationary_2d_numpy_solution import (
    analytical_oscillator_energies,
    solve_stationary_2d,
)


def grid_convergence() -> None:
    exact_ground = float(analytical_oscillator_energies(1)[0])
    print("2D anisotropic-oscillator ground-state convergence")
    print("Nx=Ny   spacing x    numerical E0    absolute error")
    for num_points in (12, 18, 26, 36):
        result = solve_stationary_2d(
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


def main() -> None:
    grid_convergence()


if __name__ == "__main__":
    main()
