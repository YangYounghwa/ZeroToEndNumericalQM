"""Grid and domain convergence experiments for the harmonic oscillator."""

from harmonic_oscillator_torch_solution import (
    analytical_energies,
    solve_harmonic_oscillator,
)


def grid_convergence() -> None:
    """Refine the grid while keeping the domain fixed."""
    exact = float(analytical_energies(1)[0])
    previous_error: float | None = None

    print("Grid convergence at fixed x_max = 8")
    print("points  spacing       ground energy   abs error      error ratio")
    for num_points in (50, 100, 200, 400):
        result = solve_harmonic_oscillator(
            num_points=num_points, num_states=1, x_max=8.0
        )
        energy = float(result.energies[0])
        error = abs(energy - exact)
        ratio = "-" if previous_error is None else f"{previous_error / error:.3f}"
        print(
            f"{num_points:>6}  {result.spacing:>10.3e}  {energy:>14.8f}  "
            f"{error:>12.3e}  {ratio:>11}"
        )
        previous_error = error


def domain_convergence() -> None:
    """Increase the domain while keeping the spacing approximately fixed."""
    target_spacing = 0.05
    exact = float(analytical_energies(1)[0])

    print("\nDomain convergence at spacing approximately 0.05")
    print("x_max  points  spacing       abs error      edge density")
    for x_max in (2.0, 3.0, 4.0, 6.0, 8.0):
        num_points = round(2.0 * x_max / target_spacing) - 1
        result = solve_harmonic_oscillator(
            num_points=num_points, num_states=1, x_max=x_max
        )
        error = abs(float(result.energies[0]) - exact)
        edge_density = max(
            float(abs(result.wavefunctions[0, 0]) ** 2),
            float(abs(result.wavefunctions[-1, 0]) ** 2),
        )
        print(
            f"{x_max:>5.1f}  {num_points:>6}  {result.spacing:>10.3e}  "
            f"{error:>12.3e}  {edge_density:>12.3e}"
        )


def main() -> None:
    grid_convergence()
    domain_convergence()
    print("\nSecond-order grid convergence gives an error ratio near 4.")


if __name__ == "__main__":
    main()
