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


def matrix_exponential_state(
    initial_state: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> Tensor:
    # TODO: Normalize, then apply torch.linalg.matrix_exp(-1j * time * H / hbar).
    # Keep this dense reference small and retain the state's device.
    raise NotImplementedError


def state_l2_error(numerical: Tensor, reference: Tensor, spacing: float) -> Tensor:
    # TODO: Align the global phase using the weighted conjugate overlap.
    # Return the weighted L2 norm of the difference.
    raise NotImplementedError


def main() -> None:
    # TODO: Select a device and report conserved quantities.
    raise NotImplementedError


if __name__ == "__main__":
    main()
