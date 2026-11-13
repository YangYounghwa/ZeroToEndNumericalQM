"""Open transverse-field Ising chain with bit-string PyTorch construction."""

import torch
from torch import Tensor


def dimension(num_sites: int) -> int:
    """Limit dense teaching calculations to ten spins."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def hamiltonian(
    num_sites: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """H=-J sum Z_i Z_(i+1)-g sum X_i, open ends, bit zero is site zero."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def parity_embedding(
    num_sites: int, parity: int, *, device: str | torch.device = "cpu"
) -> Tensor:
    """Columns (|b> + parity*|complement b>)/sqrt(2), for b<D/2."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def _states(states: Tensor, num_sites: int) -> Tensor:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def magnetization(states: Tensor, num_sites: int, axis: str) -> Tensor:
    """Mean Pauli magnetization per site; supports state batches."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def zz_correlation(states: Tensor, num_sites: int, left: int, right: int) -> Tensor:
    """Return <Z_left Z_right>, including the identity when sites coincide."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def parity_expectation(states: Tensor, num_sites: int) -> Tensor:
    """Global X product complements all bits."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def main() -> None:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


if __name__ == "__main__":
    main()
