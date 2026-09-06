"""Starter for double-well convergence and barrier studies."""


def grid_convergence() -> None:
    # TODO: Compare coarse grids with a high-resolution numerical reference.
    # TODO: Track both E_0 and the smaller difference E_1 - E_0.
    raise NotImplementedError


def barrier_study() -> None:
    # TODO: Vary barrier_height while keeping other parameters fixed.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Increase the domain at fixed spacing, keeping the potential fixed.
    # Report energy changes separately from grid-refinement error.
    raise NotImplementedError


def main() -> None:
    domain_convergence()
    grid_convergence()
    barrier_study()


if __name__ == "__main__":
    main()
