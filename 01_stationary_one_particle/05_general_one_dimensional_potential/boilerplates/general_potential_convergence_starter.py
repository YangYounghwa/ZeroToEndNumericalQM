"""Starter convergence study for the reusable stationary solver."""


def grid_convergence() -> None:
    # TODO: Compare ground-energy errors while refining the grid.
    raise NotImplementedError


def potential_reuse_study() -> None:
    # TODO: Run the same solver with several potential callables.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Increase the domain at fixed spacing, keeping the potential fixed.
    # Report energy changes separately from grid-refinement error.
    raise NotImplementedError


def main() -> None:
    domain_convergence()
    grid_convergence()
    potential_reuse_study()


if __name__ == "__main__":
    main()
