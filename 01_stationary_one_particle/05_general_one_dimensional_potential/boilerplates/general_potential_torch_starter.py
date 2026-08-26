"""Starter template for the reusable PyTorch stationary solver."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

PotentialFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class StationaryResult:
    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


@dataclass(frozen=True)
class BatchedStationaryResult:
    grid: Tensor
    spacing: float
    potentials: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonians: Tensor


def make_grid(
    num_points: int,
    x_min: float = -8.0,
    x_max: float = 8.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    # TODO: Return float64 interior points on the requested device.
    raise NotImplementedError


def kinetic_energy_matrix(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    # TODO: Construct the float64 kinetic matrix on the requested device.
    raise NotImplementedError


def evaluate_potential(grid: Tensor, potential_function: PotentialFunction) -> Tensor:
    # TODO: Evaluate the callable and validate shape, reality, and finiteness.
    raise NotImplementedError


def solve_stationary(
    potential_function: PotentialFunction,
    num_points: int = 300,
    num_states: int = 6,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> StationaryResult:
    # TODO: Build H, diagonalize, select states, and normalize the columns.
    raise NotImplementedError


def solve_potential_batch(
    grid: Tensor,
    spacing: float,
    potentials: Tensor,
    num_states: int = 4,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> BatchedStationaryResult:
    # TODO: Build and diagonalize Hamiltonians with shape (B, N, N).
    raise NotImplementedError


def main() -> None:
    # TODO: Compare a shifted harmonic result with its exact spectrum.
    raise NotImplementedError


if __name__ == "__main__":
    main()
