"""Spinful Hubbard dimer: Fock operators followed by sector projection."""

from math import isfinite

import torch
from torch import Tensor

# Orbitals: 0 up, 0 down, 1 up, 1 down. Basis: D0, up/down, down/up, D1.
SECTOR = (3, 9, 6, 12)


def annihilation(orbital: int, *, device: str | torch.device = "cpu") -> Tensor:
    """Fermion annihilation on the full 16-state Fock space."""
    if (
        isinstance(orbital, bool)
        or not isinstance(orbital, int)
        or not 0 <= orbital < 4
    ):
        raise ValueError("orbital must be an integer from zero to three")
    operator = torch.zeros((16, 16), dtype=torch.complex128, device=device)
    for mask in range(16):
        if mask & (1 << orbital):
            parity = (mask & ((1 << orbital) - 1)).bit_count()
            operator[mask ^ (1 << orbital), mask] = (-1) ** parity
    return operator


def hamiltonian(
    *,
    hopping: float = 1.0,
    interaction: float = 0.0,
    bias: float = 0.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return the complete (1 up, 1 down) sector; bias means v0-v1."""
    if not all(isfinite(x) for x in (hopping, interaction, bias)):
        raise ValueError("parameters must be finite")
    c = [annihilation(q, device=device) for q in range(4)]
    n = [a.mH @ a for a in c]
    full = interaction * (n[0] @ n[1] + n[2] @ n[3])
    full = full + bias / 2 * (n[0] + n[1] - n[2] - n[3])
    for q in (0, 1):
        move = c[q].mH @ c[q + 2]
        full = full - hopping * (move + move.mH)
    indices = torch.tensor(SECTOR, device=device)
    return full[indices[:, None], indices[None, :]]


def _states(states: Tensor) -> Tensor:
    if states.ndim < 1 or states.shape[-1] != 4 or states.numel() == 0:
        raise ValueError("states must have nonempty shape (..., 4)")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite")
    norm = states.abs().square().sum(-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-10, rtol=0):
        raise ValueError("states must be normalized")
    return states.to(dtype=torch.complex128)


def evolve(h: Tensor, states: Tensor, times: Tensor) -> Tensor:
    """Direct matrix exponentials, shape (T, ..., 4); hbar=1."""
    if h.shape != (4, 4) or not torch.isfinite(h).all():
        raise ValueError("h must be a finite 4 by 4 matrix")
    if not torch.allclose(h, h.mH, atol=1e-12, rtol=0):
        raise ValueError("h must be Hermitian")
    states = _states(states).to(device=h.device)
    if times.ndim != 1 or times.numel() == 0 or times.is_complex():
        raise ValueError("times must be a nonempty real vector")
    if not torch.isfinite(times).all():
        raise ValueError("times must be finite")
    times = times.to(dtype=torch.float64, device=h.device)
    propagators = torch.linalg.matrix_exp(
        -1j * times[:, None, None] * h.to(dtype=torch.complex128)
    )
    return torch.einsum("tij,...j->t...i", propagators, states)


def double_occupancy(states: Tensor) -> Tensor:
    """Expected total doublon count, also the same-site probability here."""
    probabilities = _states(states).abs().square()
    return probabilities[..., 0] + probabilities[..., 3]


def site_occupations(states: Tensor) -> Tensor:
    """Return mean counts on sites 0 and 1; sum equals two."""
    probabilities = _states(states).abs().square()
    weights = torch.tensor(
        [[2.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 2.0]],
        dtype=torch.float64,
        device=states.device,
    )
    return probabilities @ weights


def main() -> None:
    h = hamiltonian(interaction=4)
    energies, vectors = torch.linalg.eigh(h)
    print("Energies:", energies.tolist())
    print("Ground-state double occupancy:", float(double_occupancy(vectors[:, 0])))
    print("Site occupations:", site_occupations(vectors[:, 0]).tolist())


if __name__ == "__main__":
    main()
