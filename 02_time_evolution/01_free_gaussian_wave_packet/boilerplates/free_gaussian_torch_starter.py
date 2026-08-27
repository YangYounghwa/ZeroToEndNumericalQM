"""Starter template for PyTorch free Gaussian propagation."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class EvolutionResult:
    grid: Tensor
    spacing: float
    times: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def gaussian_wave_packet(
    grid: Tensor,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
) -> Tensor:
    # TODO: Construct a complex128 Gaussian packet.
    raise NotImplementedError


def propagate_crank_nicolson_batch(
    initial_states: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    # TODO: Propagate state columns using one LU factorization.
    raise NotImplementedError


def solve_free_packet(
    device: str | torch.device = "cpu",
) -> EvolutionResult:
    # TODO: Assemble and run the default free-packet problem.
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and report conserved quantities.
    raise NotImplementedError


if __name__ == "__main__":
    main()
