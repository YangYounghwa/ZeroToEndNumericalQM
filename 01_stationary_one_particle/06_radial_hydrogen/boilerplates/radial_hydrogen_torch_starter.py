"""Starter template for the PyTorch reduced radial hydrogen solver."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RadialResult:
    grid: Tensor
    spacing: float
    effective_potential: Tensor
    energies: Tensor
    radial_wavefunctions: Tensor
    hamiltonian: Tensor
    angular_momentum: int
    nuclear_charge: float


def make_radial_grid(
    num_points: int,
    r_max: float = 40.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    # TODO: Return float64 interior radial points on the requested device.
    raise NotImplementedError


def effective_potential(
    grid: Tensor,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    # TODO: Add the Coulomb and centrifugal terms.
    raise NotImplementedError


def solve_radial_hydrogen(
    num_points: int = 400,
    num_states: int = 3,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> RadialResult:
    # TODO: Build, diagonalize, select, and normalize the radial states.
    raise NotImplementedError


def solve_angular_momentum_batch(
    angular_momenta: Tensor,
    num_points: int = 300,
    num_states: int = 3,
    r_max: float = 40.0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> object:
    # TODO: Build and diagonalize a batch with shape (B, N, N).
    raise NotImplementedError


def main() -> None:
    # TODO: Compare the PyTorch spectrum with the analytical result.
    raise NotImplementedError


if __name__ == "__main__":
    main()
