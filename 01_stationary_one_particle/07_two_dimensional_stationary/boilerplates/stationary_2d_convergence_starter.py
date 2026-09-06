"""Starter grid-convergence study for the 2D stationary solver."""


def separable_validation_convergence() -> None:
    # TODO: Refine both axes and compare with an analytical energy.
    raise NotImplementedError


def nonseparable_grid_convergence() -> None:
    # TODO: Compare a coupled potential with a refined numerical reference.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Increase the domain at fixed spacing, keeping the potential fixed.
    # Report energy changes separately from grid-refinement error.
    raise NotImplementedError


def main() -> None:
    domain_convergence()
    separable_validation_convergence()
    nonseparable_grid_convergence()


if __name__ == "__main__":
    main()
