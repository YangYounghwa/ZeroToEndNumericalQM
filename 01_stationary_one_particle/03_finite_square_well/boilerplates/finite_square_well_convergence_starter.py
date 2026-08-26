"""Starter for finite-square-well grid and domain convergence studies."""

# After copying the starters into workbench, import:
# from finite_square_well_numpy import (
#     analytical_bound_energies,
#     solve_finite_square_well,
# )


def grid_convergence() -> None:
    # TODO: Get the exact ground energy from analytical_bound_energies().
    # TODO: Compare finite-difference ground energies with that reference.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Increase x_max at approximately fixed spacing.
    # TODO: Track bound count, ground energy, and edge density.
    raise NotImplementedError


def main() -> None:
    grid_convergence()
    domain_convergence()


if __name__ == "__main__":
    main()
