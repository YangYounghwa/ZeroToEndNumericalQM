"""Starter template for the PyTorch finite-square-well solver."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FiniteSquareWellResult:
    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def make_grid(
    num_points: int,
    x_max: float = 8.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    # TODO: Create a float64 interior grid on device.
    raise NotImplementedError


def finite_square_well_potential(
    grid: Tensor, half_width: float = 1.0, depth: float = 20.0
) -> Tensor:
    # TODO: Use torch.where and preserve device and dtype.
    raise NotImplementedError


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    # TODO: Build a float64 Hamiltonian on device.
    raise NotImplementedError


def solve_finite_square_well(
    num_points: int = 400,
    num_states: int = 8,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> FiniteSquareWellResult:
    # TODO: Use torch.linalg.eigh and keep results on device.
    raise NotImplementedError


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"selected device: {device}")
    raise NotImplementedError


if __name__ == "__main__":
    main()
