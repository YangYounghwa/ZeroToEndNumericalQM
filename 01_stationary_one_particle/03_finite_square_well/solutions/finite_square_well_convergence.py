"""Grid and domain convergence studies for the finite square well."""

from finite_square_well_numpy_solution import analytical_bound_energies
from finite_square_well_torch_solution import (
    bound_state_mask,
    solve_finite_square_well,
)


def grid_convergence() -> None:
    exact = float(analytical_bound_energies()[0])
    print("Grid convergence at fixed x_max = 8")
    print("points  spacing       ground energy   abs error")
    for num_points in (100, 200, 400, 800):
        result = solve_finite_square_well(num_points=num_points, num_states=1)
        error = abs(float(result.energies[0]) - exact)
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  "
            f"{result.energies[0]:>14.8f}  {error:>12.3e}"
        )


def domain_convergence() -> None:
    # h=2/51 and even integer domains keep x=+/-1 halfway between grid
    # points, so changing x_max does not also change the represented well width.
    target_spacing = 2.0 / 51.0
    print("\nDomain convergence at fixed, step-aligned spacing")
    print("x_max  points  bound states  fifth energy    fifth edge density")
    for x_max in (2.0, 4.0, 6.0, 8.0, 10.0):
        num_points = round(2.0 * x_max / target_spacing) - 1
        result = solve_finite_square_well(
            num_points=num_points, num_states=6, x_max=x_max
        )
        fifth_state = 4
        edge_density = max(
            float(abs(result.wavefunctions[0, fifth_state]) ** 2),
            float(abs(result.wavefunctions[-1, fifth_state]) ** 2),
        )
        print(
            f"{x_max:>5.1f}  {num_points:>6}  "
            f"{int(bound_state_mask(result).sum()):>12}  "
            f"{result.energies[fifth_state]:>12.8f}  {edge_density:>18.3e}"
        )


def main() -> None:
    grid_convergence()
    domain_convergence()
    print("\nThe discontinuous potential can make grid error non-monotone.")


if __name__ == "__main__":
    main()
