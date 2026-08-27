"""Starter template for a PyTorch 2D stationary solver."""

from torch import Tensor


def solve_stationary_2d() -> object:
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


def main() -> None:
    # TODO: Select a device and report energies and positions.
    raise NotImplementedError


if __name__ == "__main__":
    main()
