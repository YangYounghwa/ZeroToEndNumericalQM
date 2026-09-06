"""Convergence studies for the reduced radial hydrogen equation."""

from radial_hydrogen_torch_solution import (
    analytical_energies,
    solve_radial_hydrogen,
)


def grid_convergence() -> None:
    """Measure energy convergence as the radial spacing decreases."""
    exact_ground = float(analytical_energies(1)[0])
    print("Grid convergence at fixed r_max = 40")
    print("points  spacing       ground energy   absolute error")
    for num_points in (100, 200, 400, 800):
        result = solve_radial_hydrogen(num_points=num_points, num_states=1)
        error = abs(float(result.energies[0]) - exact_ground)
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>14.3e}"
        )


def domain_convergence() -> None:
    """Show the outer-boundary effect on a spatially extended excited state."""
    exact = float(analytical_energies(3)[2])
    spacing_target = 0.05
    print("\nDomain convergence for the 3s state at nearly fixed spacing")
    print("r_max   points   numerical energy   absolute error")
    for r_max in (12.0, 20.0, 30.0, 40.0):
        num_points = round(r_max / spacing_target) - 1
        result = solve_radial_hydrogen(num_points=num_points, num_states=3, r_max=r_max)
        error = abs(float(result.energies[2]) - exact)
        print(
            f"{r_max:>5.1f}  {num_points:>7}  {result.energies[2]:>16.10f}  "
            f"{error:>14.3e}"
        )


def angular_momentum_study() -> None:
    """Verify Coulomb degeneracy across different angular-momentum sectors."""
    print("\nLowest state in each angular-momentum sector")
    print("ell   n   numerical energy      exact energy")
    for angular_momentum in (0, 1, 2):
        result = solve_radial_hydrogen(
            num_points=800,
            num_states=1,
            angular_momentum=angular_momentum,
        )
        exact = analytical_energies(1, angular_momentum)[0]
        print(
            f"{angular_momentum:>3}  {angular_momentum + 1:>2}  "
            f"{result.energies[0]:>16.10f}  {exact:>16.10f}"
        )


def main() -> None:
    grid_convergence()
    domain_convergence()
    angular_momentum_study()


if __name__ == "__main__":
    main()
