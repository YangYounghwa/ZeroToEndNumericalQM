"""Grid convergence for the sparse 3D stationary solver."""

from stationary_3d_numpy_solution import (
    analytical_oscillator_energies,
    solve_stationary_3d,
)


def grid_convergence() -> None:
    exact_ground = float(analytical_oscillator_energies(1)[0])
    print("3D anisotropic-oscillator ground-state convergence")
    print("Nx=Ny=Nz   spacing    numerical E0    absolute error")
    for num_points in (10, 14, 20, 28):
        result = solve_stationary_3d(
            num_x=num_points,
            num_y=num_points,
            num_z=num_points,
            num_states=1,
            x_min=-5.0,
            x_max=5.0,
            y_min=-5.0,
            y_max=5.0,
            z_min=-5.0,
            z_max=5.0,
        )
        error = abs(float(result.energies[0]) - exact_ground)
        print(
            f"{num_points:>8}   {result.spacing_x:>9.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>14.3e}"
        )


def main() -> None:
    grid_convergence()


if __name__ == "__main__":
    main()
