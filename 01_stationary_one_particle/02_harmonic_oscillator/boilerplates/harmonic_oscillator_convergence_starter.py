"""Starter for harmonic-oscillator grid and domain convergence studies."""

# TODO: Import the analytical energy and solver from your NumPy file.


def grid_convergence() -> None:
    """Refine the grid at fixed domain size."""
    point_counts = (50, 100, 200, 400)
    print(f"point counts: {point_counts}")
    # TODO: Hold x_max fixed and record spacing and ground-energy error.
    # TODO: Print previous_error / current_error. It should approach 4.
    raise NotImplementedError


def domain_convergence() -> None:
    """Increase domain size at approximately fixed spacing."""
    domain_sizes = (2.0, 3.0, 4.0, 6.0, 8.0)
    print(f"domain sizes: {domain_sizes}")
    # TODO: Adjust num_points so that spacing stays near a fixed target.
    # TODO: Record energy error and endpoint probability density.
    raise NotImplementedError


def main() -> None:
    grid_convergence()
    domain_convergence()


if __name__ == "__main__":
    main()
