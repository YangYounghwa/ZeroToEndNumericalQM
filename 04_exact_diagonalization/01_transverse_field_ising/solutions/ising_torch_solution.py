"""Open transverse-field Ising chain with bit-string PyTorch construction."""

from math import isfinite, sqrt

import torch
from torch import Tensor


def dimension(num_sites: int) -> int:
    """Limit dense teaching calculations to ten spins."""
    if (
        isinstance(num_sites, bool)
        or not isinstance(num_sites, int)
        or not 1 <= num_sites <= 10
    ):
        raise ValueError("num_sites must be an integer from 1 to 10")
    return 1 << num_sites


def hamiltonian(
    num_sites: int,
    *,
    coupling: float = 1.0,
    field: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """H=-J sum Z_i Z_(i+1)-g sum X_i, open ends, bit zero is site zero."""
    size = dimension(num_sites)
    if not all(isfinite(x) for x in (coupling, field)):
        raise ValueError("coupling and field must be finite")
    masks = torch.arange(size, device=device)
    z = 1 - 2 * ((masks[:, None] >> torch.arange(num_sites, device=device)) & 1)
    diagonal = -coupling * (z[:, :-1] * z[:, 1:]).sum(-1).to(torch.float64)
    h = torch.diag(diagonal)
    for site in range(num_sites):
        h[masks ^ (1 << site), masks] -= field
    return h


def parity_embedding(
    num_sites: int,
    parity: int,
    *,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Columns (|b> + parity*|complement b>)/sqrt(2), for b<D/2."""
    size = dimension(num_sites)
    if isinstance(parity, bool) or parity not in (-1, 1):
        raise ValueError("parity must be -1 or +1")
    indices = torch.arange(size // 2, device=device)
    q = torch.zeros((size, size // 2), dtype=torch.float64, device=device)
    q[indices, indices] = 1 / sqrt(2)
    q[(size - 1) ^ indices, indices] = parity / sqrt(2)
    return q


def _states(states: Tensor, num_sites: int) -> Tensor:
    size = dimension(num_sites)
    if states.ndim < 1 or states.shape[-1] != size or states.numel() == 0:
        raise ValueError("states must have nonempty shape (..., 2**num_sites)")
    if not torch.isfinite(states).all():
        raise ValueError("states must be finite")
    norm = states.abs().square().sum(-1)
    if not torch.allclose(norm, torch.ones_like(norm), atol=1e-10, rtol=0):
        raise ValueError("states must be normalized")
    return states.to(dtype=torch.complex128)


def magnetization(states: Tensor, num_sites: int, axis: str) -> Tensor:
    """Mean Pauli magnetization per site; supports state batches."""
    states = _states(states, num_sites)
    if axis not in ("x", "z"):
        raise ValueError("axis must be x or z")
    masks = torch.arange(states.shape[-1], device=states.device)
    if axis == "z":
        z = 1 - 2 * (
            (masks[:, None] >> torch.arange(num_sites, device=states.device)) & 1
        )
        return (states.abs().square() * z.to(torch.float64).mean(-1)).sum(-1)
    values = [
        (states.conj() * states[..., masks ^ (1 << i)]).sum(-1).real
        for i in range(num_sites)
    ]
    return torch.stack(values, dim=-1).mean(-1)


def zz_correlation(states: Tensor, num_sites: int, left: int, right: int) -> Tensor:
    """Return <Z_left Z_right>, including the identity when sites coincide."""
    states = _states(states, num_sites)
    for site in (left, right):
        if (
            isinstance(site, bool)
            or not isinstance(site, int)
            or not 0 <= site < num_sites
        ):
            raise ValueError("sites must be valid integer indices")
    masks = torch.arange(states.shape[-1], device=states.device)
    weights = (1 - 2 * ((masks >> left) & 1)) * (1 - 2 * ((masks >> right) & 1))
    return (states.abs().square() * weights).sum(-1)


def parity_expectation(states: Tensor, num_sites: int) -> Tensor:
    """Global X product complements all bits."""
    states = _states(states, num_sites)
    return (states.conj() * states.flip(-1)).sum(-1).real


def main() -> None:
    h = hamiltonian(6)
    energies, vectors = torch.linalg.eigh(h)
    ground = vectors[:, 0]
    print(
        f"L=6 E0={float(energies[0]):.12f} gap={float(energies[1] - energies[0]):.12f}"
    )
    print("Mx:", float(magnetization(ground, 6, "x")))
    print("Mz:", float(magnetization(ground, 6, "z")))
    print("End correlation:", float(zz_correlation(ground, 6, 0, 5)))
    print("Parity:", float(parity_expectation(ground, 6)))


if __name__ == "__main__":
    main()
