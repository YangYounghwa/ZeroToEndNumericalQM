"""Starter convergence studies for the radial hydrogen equation."""


def grid_convergence() -> None:
    # TODO: Refine the radial grid at fixed r_max.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Increase r_max at nearly fixed spacing for an excited state.
    raise NotImplementedError


def angular_momentum_study() -> None:
    # TODO: Compare several ell sectors with exact Coulomb energies.
    raise NotImplementedError


def main() -> None:
    grid_convergence()
    domain_convergence()
    angular_momentum_study()


if __name__ == "__main__":
    main()
