"""Convergence experiment for the infinite square well."""

from infinite_square_well_torch_solution import (
    analytical_energies,
    solve_infinite_well,
)


def main() -> None:
    point_counts = (20, 40, 80, 160, 320)
    exact_ground_energy = float(analytical_energies(1)[0])
    previous_error: float | None = None

    print("points  spacing       ground energy   abs error      error ratio")
    for num_points in point_counts:
        result = solve_infinite_well(num_points=num_points, num_states=1)
        error = abs(float(result.energies[0]) - exact_ground_energy)
        ratio = "-" if previous_error is None else f"{previous_error / error:.3f}"
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>12.3e}  {ratio:>11}"
        )
        previous_error = error

    print("\nFor second-order convergence, the error ratio approaches 4.")


if __name__ == "__main__":
    main()
