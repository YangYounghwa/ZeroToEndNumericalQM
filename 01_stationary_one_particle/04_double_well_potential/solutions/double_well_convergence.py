"""Convergence and barrier studies for the symmetric double well."""

from double_well_numpy_solution import solve_double_well, tunneling_splitting


def grid_convergence() -> None:
    reference = solve_double_well(num_points=800, num_states=2)
    reference_energy = float(reference.energies[0])
    print("Grid convergence at fixed x_max = 6")
    print("points  spacing       ground energy   reference error  splitting")
    for num_points in (75, 150, 300, 600):
        result = solve_double_well(num_points=num_points, num_states=2)
        error = abs(float(result.energies[0]) - reference_energy)
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>15.3e}  "
            f"{tunneling_splitting(result):>10.3e}"
        )


def barrier_study() -> None:
    print("\nTunneling splitting versus barrier height")
    print("barrier  ground energy   splitting")
    for barrier_height in (2.0, 4.0, 8.0, 12.0, 16.0):
        result = solve_double_well(
            num_points=300, num_states=2, barrier_height=barrier_height
        )
        print(
            f"{barrier_height:>7.1f}  {result.energies[0]:>14.8f}  "
            f"{tunneling_splitting(result):>10.3e}"
        )


def main() -> None:
    grid_convergence()
    barrier_study()


if __name__ == "__main__":
    main()
