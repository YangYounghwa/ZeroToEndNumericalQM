"""Two noninteracting lattice particles, using explicit symmetry projection."""

from itertools import combinations, combinations_with_replacement
from math import isfinite, pi, sqrt
from typing import Literal

import torch
from torch import Tensor

type Statistics = Literal["distinguishable", "boson", "fermion"]


def basis_pairs(num_sites: int, statistics: Statistics) -> list[tuple[int, int]]:
    """Ordered pairs for labels; sorted pairs for identical particles."""
    if isinstance(num_sites, bool) or not isinstance(num_sites, int) or num_sites < 2:
        raise ValueError("num_sites must be an integer of at least two")
    sites = range(num_sites)
    if statistics == "distinguishable":
        return [(i, j) for i in sites for j in sites]
    if statistics == "boson":
        return list(combinations_with_replacement(sites, 2))
    if statistics == "fermion":
        return list(combinations(sites, 2))
    raise ValueError("unknown particle statistics")


def single_particle_hamiltonian(
    num_sites: int,
    *,
    hopping: float = 1.0,
    onsite: Tensor | None = None,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Open chain: h[i,i]=v[i], h[i,i+1]=h[i+1,i]=-hopping."""
    basis_pairs(num_sites, "distinguishable")
    if not isfinite(hopping):
        raise ValueError("hopping must be finite")
    if onsite is None:
        values = torch.zeros(num_sites, dtype=torch.float64, device=device)
    else:
        if onsite.shape != (num_sites,) or onsite.is_complex():
            raise ValueError("onsite must be a real vector of length num_sites")
        values = onsite.to(dtype=torch.float64, device=device)
        if not torch.isfinite(values).all():
            raise ValueError("onsite must be finite")
    off = torch.full((num_sites - 1,), -hopping, dtype=torch.float64, device=device)
    return (torch.diag(values) + torch.diag(off, 1) + torch.diag(off, -1)).to(
        torch.complex128
    )


def _validate_hamiltonian(hamiltonian: Tensor) -> None:
    if (
        hamiltonian.ndim != 2
        or hamiltonian.shape[0] == 0
        or hamiltonian.shape[0] != hamiltonian.shape[1]
        or not torch.isfinite(hamiltonian).all()
    ):
        raise ValueError("Hamiltonian must be finite, nonempty, and square")
    if not torch.allclose(hamiltonian, hamiltonian.mH, atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian")


def _validate_states(states: Tensor, dimension: int) -> None:
    if (
        states.ndim < 1
        or states.numel() == 0
        or states.shape[-1] != dimension
        or not torch.isfinite(states).all()
    ):
        raise ValueError("states must have finite, nonempty shape (..., dimension)")
    norms = states.abs().square().sum(-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-10, rtol=0):
        raise ValueError("states must be normalized; no automatic renormalization")


def basis_embedding(
    num_sites: int, statistics: Statistics, *, device: str | torch.device = "cpu"
) -> Tensor:
    """Columns embed normalized basis states into index i*num_sites+j."""
    pairs = basis_pairs(num_sites, statistics)
    embedding = torch.zeros(
        (num_sites**2, len(pairs)), dtype=torch.complex128, device=device
    )
    for column, (i, j) in enumerate(pairs):
        if statistics == "distinguishable" or i == j:
            embedding[i * num_sites + j, column] = 1
        else:
            embedding[i * num_sites + j, column] = 1 / sqrt(2)
            sign = 1 if statistics == "boson" else -1
            embedding[j * num_sites + i, column] = sign / sqrt(2)
    return embedding


def two_particle_hamiltonian(h: Tensor, statistics: Statistics) -> Tensor:
    """Project h tensor I + I tensor h onto the requested complete sector."""
    _validate_hamiltonian(h)
    h = h.to(dtype=torch.complex128).contiguous()
    num_sites = h.shape[0]
    embedding = basis_embedding(num_sites, statistics, device=h.device)
    identity = torch.eye(num_sites, dtype=h.dtype, device=h.device)
    labeled = torch.kron(h, identity) + torch.kron(identity, h)
    return embedding.mH @ labeled @ embedding


def localized_state(
    num_sites: int,
    statistics: Statistics,
    pair: tuple[int, int],
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """One basis state; identical-particle pairs must use canonical order."""
    pairs = basis_pairs(num_sites, statistics)
    if len(pair) != 2 or any(
        isinstance(i, bool) or not isinstance(i, int) for i in pair
    ):
        raise ValueError("pair must contain two integer sites")
    if pair not in pairs:
        raise ValueError("pair is absent from the canonical basis")
    state = torch.zeros(len(pairs), dtype=torch.complex128, device=device)
    state[pairs.index(pair)] = 1
    return state


def evolve(
    hamiltonian: Tensor, states: Tensor, times: Tensor, *, hbar: float = 1.0
) -> Tensor:
    """Return (T, ..., D) states using exp(-i H t / hbar), with no time steps."""
    _validate_hamiltonian(hamiltonian)
    _validate_states(states, hamiltonian.shape[0])
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if (
        times.ndim != 1
        or times.numel() == 0
        or times.is_complex()
        or not torch.isfinite(times).all()
    ):
        raise ValueError("times must be a finite, nonempty real vector")
    h = hamiltonian.to(dtype=torch.complex128)
    states = states.to(dtype=torch.complex128, device=h.device)
    times = times.to(dtype=torch.float64, device=h.device)
    propagators = torch.linalg.matrix_exp(-1j * times[:, None, None] * h / hbar)
    return torch.einsum("tij,...j->t...i", propagators, states)


def pair_amplitudes(states: Tensor, num_sites: int, statistics: Statistics) -> Tensor:
    """Expand sector states to (..., L, L) ordered-coordinate amplitudes."""
    embedding = basis_embedding(num_sites, statistics, device=states.device)
    _validate_states(states, embedding.shape[1])
    expanded = torch.einsum(
        "pd,...d->...p", embedding, states.to(dtype=torch.complex128)
    )
    return expanded.reshape(*states.shape[:-1], num_sites, num_sites)


def site_occupations(states: Tensor, num_sites: int, statistics: Statistics) -> Tensor:
    """Mean particle counts per site; their sum is two, not one."""
    probabilities = pair_amplitudes(states, num_sites, statistics).abs().square()
    return probabilities.sum(-1) + probabilities.sum(-2)


def coincidence_probability(
    states: Tensor, num_sites: int, statistics: Statistics
) -> Tensor:
    """Probability that both particles occupy the same site."""
    probabilities = pair_amplitudes(states, num_sites, statistics).abs().square()
    return probabilities.diagonal(dim1=-2, dim2=-1).sum(-1)


def main() -> None:
    h = single_particle_hamiltonian(2)
    times = torch.tensor([0, pi / 8, pi / 4, pi / 2], dtype=torch.float64)
    for statistics in ("distinguishable", "boson", "fermion"):
        h2 = two_particle_hamiltonian(h, statistics)
        initial = localized_state(2, statistics, (0, 1))
        history = evolve(h2, initial, times)
        print(f"{statistics}: energies={torch.linalg.eigvalsh(h2).tolist()}")
        print(
            f"  same-site probability={coincidence_probability(history, 2, statistics).tolist()}"
        )


if __name__ == "__main__":
    main()
