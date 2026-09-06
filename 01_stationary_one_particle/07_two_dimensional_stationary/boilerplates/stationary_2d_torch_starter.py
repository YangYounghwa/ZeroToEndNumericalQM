"""Starter template for a PyTorch 2D stationary solver."""

from collections.abc import Callable

from torch import Tensor

Potential2D = Callable[[Tensor, Tensor], Tensor]


def solve_stationary_2d(potential_function: Potential2D) -> object:
    # TODO: Build the dense Kronecker sum and reshape selected states.
    raise NotImplementedError


def solve_potential_batch(
    kinetic_hamiltonian: Tensor,
    potentials: Tensor,
    spacing_x: float,
    spacing_y: float,
    num_states: int = 4,
) -> object:
    # TODO: Diagonalize potentials shaped (batch, Ny, Nx).
    raise NotImplementedError


def subspace_overlaps(first: Tensor, second: Tensor, area_element: float) -> Tensor:
    # TODO: For weighted-orthonormal (grid_points, states) columns,
    # return singular values of area_element * first.mH @ second.
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and solve a nonseparable potential.
    raise NotImplementedError


if __name__ == "__main__":
    main()
