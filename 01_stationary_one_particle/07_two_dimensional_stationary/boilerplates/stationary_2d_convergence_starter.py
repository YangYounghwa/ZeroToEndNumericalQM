"""Starter grid-convergence study for the 2D stationary solver."""


def separable_validation_convergence() -> None:
    # TODO: Refine both axes and compare with an analytical energy.
    raise NotImplementedError


def nonseparable_grid_convergence() -> None:
    # TODO: Compare a coupled potential with a refined numerical reference.
    raise NotImplementedError


def main() -> None:
    separable_validation_convergence()
    nonseparable_grid_convergence()


if __name__ == "__main__":
    main()
