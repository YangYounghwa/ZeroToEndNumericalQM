"""Starter: two coupled spins. Complete a copy in workbench/."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CoupledEvolution:
    """History (times,batch,4), basis (++,+-,-+,--), first spin index first."""

    hamiltonians: Tensor
    times: Tensor
    states: Tensor


def pauli_matrices(device: str | torch.device = "cpu") -> Tensor:
    """TODO: Build the three complex Pauli matrices in x,y,z order."""
    raise NotImplementedError


def local_operators(device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
    """TODO: Use Kronecker products with identity to act on either site in the fixed basis."""
    raise NotImplementedError


def normalize_states(states: Tensor) -> Tensor:
    """TODO: Normalize the final four-amplitude axis and reject zero or nonfinite states."""
    raise NotImplementedError


def product_states(first: Tensor, second: Tensor) -> Tensor:
    """TODO: Form one 2x2 outer product per spinor pair, then flatten only the spin axes."""
    raise NotImplementedError


def hamiltonians(
    exchange: Tensor, fields: Tensor, gamma: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """TODO: Combine J/4 times the three pair operators with the two signed Zeeman terms."""
    raise NotImplementedError


def evolve_constant(
    initial_states: Tensor, operators: Tensor, times: Tensor, hbar: float = 1.0
) -> CoupledEvolution:
    """TODO: Validate the batch and evaluate the static propagator at all requested times."""
    raise NotImplementedError


def propagate_cn(
    initial_states: Tensor,
    operators: Tensor,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """TODO: Reuse the constant CN solve and save initial, final, and requested intermediate states."""
    raise NotImplementedError


def singlet_triplet_basis(device: str | torch.device = "cpu") -> Tensor:
    """TODO: Return t+, t0, t-, and singlet as columns in the ordered product basis."""
    raise NotImplementedError


def exchange_reference(
    times: Tensor, exchange: float = 1.0, detuning: float = 0.0, hbar: float = 1.0
) -> Tensor:
    """TODO: Exponentiate the longitudinal two-state block with a continuous zero-splitting limit."""
    raise NotImplementedError


def joint_probabilities(states: Tensor) -> Tensor:
    """TODO: Return squared amplitudes in (++,+-,-+,--) order without renormalizing."""
    raise NotImplementedError


def local_bloch_vectors(states: Tensor) -> Tensor:
    """TODO: Contract joint states with each site's three Pauli operators."""
    raise NotImplementedError


def correlations(states: Tensor, connected: bool = False) -> Tensor:
    """TODO: Compute all nine pair expectations, optionally subtracting products of local means."""
    raise NotImplementedError


def product_determinant(states: Tensor) -> Tensor:
    """TODO: Evaluate the magnitude of the 2x2 coefficient determinant as a pure-state product test."""
    raise NotImplementedError


def energy_expectations(result: CoupledEvolution) -> Tensor:
    """TODO: Contract each state with its own Hamiltonian, preserving time and batch axes."""
    raise NotImplementedError


def phase_aligned_errors(states: Tensor, reference: Tensor) -> Tensor:
    """TODO: Remove each relative global phase before computing spinor distances."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the example or write a long experiment report to Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
