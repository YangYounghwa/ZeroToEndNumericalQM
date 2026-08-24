"""Starter for finite-square-well grid and domain convergence studies."""


def grid_convergence() -> None:
    # TODO: Compare ground energies with the matching-equation reference.
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
