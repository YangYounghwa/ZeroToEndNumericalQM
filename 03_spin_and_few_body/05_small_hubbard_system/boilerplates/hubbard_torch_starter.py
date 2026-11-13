"""Spinful Hubbard dimer: Fock operators followed by sector projection."""

import torch
from torch import Tensor

SECTOR = (3, 9, 6, 12)


def annihilation(orbital: int, *, device: str | torch.device = "cpu") -> Tensor:
    """Fermion annihilation on the full 16-state Fock space."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def hamiltonian(
    *,
    hopping: float = 1.0,
    interaction: float = 0.0,
    bias: float = 0.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return the complete (1 up, 1 down) sector; bias means v0-v1."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def _states(states: Tensor) -> Tensor:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def evolve(h: Tensor, states: Tensor, times: Tensor) -> Tensor:
    """Direct matrix exponentials, shape (T, ..., 4); hbar=1."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def double_occupancy(states: Tensor) -> Tensor:
    """Expected total doublon count, also the same-site probability here."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def site_occupations(states: Tensor) -> Tensor:
    """Return mean counts on sites 0 and 1; sum equals two."""
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


def main() -> None:
    raise NotImplementedError("Implement from theory.md and coding_hints.md")


if __name__ == "__main__":
    main()
