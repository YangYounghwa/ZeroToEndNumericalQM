"""Starter: complete a copy in workbench/. See the chapter coding hints."""

from typing import Literal

import torch
from torch import Tensor

type Statistics = Literal["distinguishable", "boson", "fermion"]


def basis_pairs(num_sites: int, statistics: Statistics) -> list[tuple[int, int]]:
    """TODO: Ordered pairs for labels; sorted pairs for identical particles."""
    raise NotImplementedError


def single_particle_hamiltonian(
    num_sites: int,
    *,
    hopping: float = 1.0,
    onsite: Tensor | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    """TODO: Open chain: h[i,i]=v[i], h[i,i+1]=h[i+1,i]=-hopping."""
    raise NotImplementedError


def basis_embedding(
    num_sites: int, statistics: Statistics, *, device: str | torch.device = "cpu"
) -> Tensor:
    """TODO: Columns embed normalized basis states into index i*num_sites+j."""
    raise NotImplementedError


def two_particle_hamiltonian(h: Tensor, statistics: Statistics) -> Tensor:
    """TODO: Project h tensor I + I tensor h onto the requested complete sector."""
    raise NotImplementedError


def localized_state(
    num_sites: int,
    statistics: Statistics,
    pair: tuple[int, int],
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """TODO: One basis state; identical-particle pairs must use canonical order."""
    raise NotImplementedError


def evolve(
    hamiltonian: Tensor, states: Tensor, times: Tensor, *, hbar: float = 1.0
) -> Tensor:
    """TODO: Return (T, ..., D) states using exp(-i H t / hbar), with no time steps."""
    raise NotImplementedError


def pair_amplitudes(states: Tensor, num_sites: int, statistics: Statistics) -> Tensor:
    """TODO: Expand sector states to (..., L, L) ordered-coordinate amplitudes."""
    raise NotImplementedError


def site_occupations(states: Tensor, num_sites: int, statistics: Statistics) -> Tensor:
    """TODO: Mean particle counts per site; their sum is two, not one."""
    raise NotImplementedError


def coincidence_probability(
    states: Tensor, num_sites: int, statistics: Statistics
) -> Tensor:
    """TODO: Probability that both particles occupy the same site."""
    raise NotImplementedError


def main() -> None:
    """TODO: Implement main using the chapter's derivation."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
