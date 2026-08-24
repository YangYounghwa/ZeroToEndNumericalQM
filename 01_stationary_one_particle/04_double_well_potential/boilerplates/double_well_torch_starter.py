"""Starter template for the PyTorch symmetric-double-well solver."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DoubleWellResult:
    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def make_grid(
    num_points: int,
    x_max: float = 6.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    # TODO: Return a float64 interior grid on device.
    raise NotImplementedError


def double_well_potential(
    grid: Tensor,
    separation: float = 1.5,
    barrier_height: float = 8.0,
) -> Tensor:
    # TODO: Evaluate the quartic potential without leaving device.
    raise NotImplementedError


def solve_double_well(
    num_points: int = 300,
    num_states: int = 6,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> DoubleWellResult:
    # TODO: Build and diagonalize a float64 Hamiltonian on device.
    raise NotImplementedError


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"selected device: {device}")
    raise NotImplementedError


if __name__ == "__main__":
    main()
