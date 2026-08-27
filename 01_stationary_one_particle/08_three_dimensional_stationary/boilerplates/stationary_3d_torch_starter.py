"""Starter template for a deliberately small dense PyTorch 3D solver."""

from torch import Tensor


def solve_stationary_3d() -> object:
    # TODO: Build and solve a small dense three-term Kronecker sum.
    raise NotImplementedError


def solve_potential_batch(
    kinetic_hamiltonian: Tensor,
    potentials: Tensor,
    spacing_x: float,
    spacing_y: float,
    spacing_z: float,
    num_states: int = 3,
) -> object:
    # TODO: Solve potentials shaped (batch,Nz,Ny,Nx).
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and keep the dense grid intentionally small.
    raise NotImplementedError


if __name__ == "__main__":
    main()
