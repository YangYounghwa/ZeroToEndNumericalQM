"""Starter time-step convergence study for a harmonic packet."""


def time_step_convergence() -> None:
    # TODO: Compare Crank-Nicolson with a matrix-exponential reference.
    raise NotImplementedError


def grid_convergence() -> None:
    # TODO: Refine the PyTorch grid at fixed domain.
    # Compare matrix-exponential evolution with the continuum packet.
    raise NotImplementedError


def domain_convergence() -> None:
    # TODO: Enlarge the domain at fixed dx and compare with the continuum packet.
    # Use matrix-exponential evolution to remove CN time-step error.
    raise NotImplementedError


def main() -> None:
    time_step_convergence()
    grid_convergence()
    domain_convergence()


if __name__ == "__main__":
    main()
