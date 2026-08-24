"""Starter for the infinite-square-well grid-convergence experiment."""

# TODO: Import analytical_energies and solve_infinite_well from your solution.


def main() -> None:
    point_counts = (20, 40, 80, 160, 320)

    # TODO: Calculate the exact ground-state energy once.
    # TODO: Solve the problem for every point count.
    # TODO: Record spacing, numerical energy, and absolute error.
    # TODO: Print the ratio previous_error / current_error.
    # A second-order method should approach a ratio of 4 when h is halved.
    print(f"point counts: {point_counts}")
    raise NotImplementedError


if __name__ == "__main__":
    main()
