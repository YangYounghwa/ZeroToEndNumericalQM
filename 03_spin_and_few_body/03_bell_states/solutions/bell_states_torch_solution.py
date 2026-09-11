"""Batched density matrices and two-spin entanglement with PyTorch."""

from math import isfinite, sqrt

import torch
from torch import Tensor


def pure_density(states: Tensor) -> Tensor:
    """Return |psi><psi| for normalized states of shape (..., D)."""
    states = states.to(dtype=torch.complex128)
    if states.ndim < 1 or states.numel() == 0 or not torch.isfinite(states).all():
        raise ValueError("states must contain finite, nonempty state vectors")
    if not torch.allclose(
        states.abs().square().sum(-1),
        torch.ones_like(states.real[..., 0]),
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("states must be normalized; no automatic renormalization")
    return states.unsqueeze(-1) * states.conj().unsqueeze(-2)


def validate_density(rho: Tensor, *, tolerance: float = 1e-12) -> None:
    """Reject nonfinite, non-Hermitian, nonunit-trace, or nonpositive input."""
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if rho.ndim < 2 or rho.shape[-1] != rho.shape[-2] or rho.numel() == 0:
        raise ValueError("rho must have nonempty shape (..., D, D)")
    if not torch.isfinite(rho).all():
        raise ValueError("rho must be finite")
    if not torch.allclose(rho, rho.mH, atol=tolerance, rtol=0):
        raise ValueError("rho must be Hermitian")
    trace = rho.diagonal(dim1=-2, dim2=-1).sum(-1)
    if not torch.allclose(trace, torch.ones_like(trace), atol=tolerance, rtol=0):
        raise ValueError("rho must have unit trace")
    if bool((torch.linalg.eigvalsh(rho) < -tolerance).any()):
        raise ValueError("rho must be positive semidefinite")


def mixed_density(states: Tensor, probabilities: Tensor) -> Tensor:
    """Mix (..., K, D) normalized states using matching (..., K) weights."""
    if states.ndim < 2 or probabilities.shape != states.shape[:-1]:
        raise ValueError("probabilities must match the ensemble axes (..., K)")
    if probabilities.is_complex():
        raise ValueError("probabilities must be real")
    weights = probabilities.to(dtype=torch.float64, device=states.device)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("probabilities must be finite and nonnegative")
    if not torch.allclose(
        weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-12, rtol=0
    ):
        raise ValueError("probabilities must sum to one")
    projectors = pure_density(states)
    return (weights[..., None, None] * projectors).sum(dim=-3)


def partial_trace(
    rho: Tensor, dims: tuple[int, int] = (2, 2), *, keep: int = 0
) -> Tensor:
    """Keep subsystem 0 (A) or 1 (B); index of |a,b> is a*dB+b."""
    validate_density(rho)
    if (
        len(dims) != 2
        or any(d <= 0 for d in dims)
        or dims[0] * dims[1] != rho.shape[-1]
    ):
        raise ValueError("positive subsystem dimensions must multiply to D")
    if keep not in (0, 1):
        raise ValueError("keep must be 0 or 1")
    da, db = dims
    tensor = rho.reshape(*rho.shape[:-2], da, db, da, db)
    # Repeated b traces B; repeated a traces A. The ellipsis preserves batches.
    if keep == 0:
        return torch.einsum("...abcb->...ac", tensor)
    return torch.einsum("...abad->...bd", tensor)


def purity(rho: Tensor) -> Tensor:
    """Tr(rho^2), equal to one for pure density matrices."""
    validate_density(rho)
    return torch.einsum("...ij,...ji->...", rho, rho).real


def entropy(rho: Tensor) -> Tensor:
    """Von Neumann entropy in bits; retain small positive eigenvalues."""
    validate_density(rho)
    eigenvalues: Tensor = torch.linalg.eigvalsh(rho)
    values = eigenvalues.clamp_min(0)
    # Only the logarithm's zero argument is replaced. No epsilon is added to rho.
    safe = torch.where(values > 0, values, torch.ones_like(values))
    return -(values * torch.log2(safe)).sum(-1)


def entanglement_entropy(states: Tensor, dims: tuple[int, int] = (2, 2)) -> Tensor:
    """Reduced entropy of PURE joint state vectors, not arbitrary mixed states."""
    return entropy(partial_trace(pure_density(states), dims, keep=0))


def bell_states(*, device: str | torch.device = "cpu") -> Tensor:
    """Rows are Phi+, Phi-, Psi+, Psi- in (00,01,10,11) order."""
    return torch.tensor(
        [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]],
        dtype=torch.complex128,
        device=device,
    ) / sqrt(2)


def schmidt_states(angles: Tensor, *, phase: float = 0.0) -> Tensor:
    """cos(theta)|00> + exp(i*phase) sin(theta)|11>, preserving angle axes."""
    if angles.is_complex() or not torch.isfinite(angles).all() or not isfinite(phase):
        raise ValueError("angles and phase must be finite and real")
    theta = angles.to(dtype=torch.float64)
    rotation = torch.exp(
        torch.tensor(1j * phase, dtype=torch.complex128, device=theta.device)
    )
    zero = torch.zeros_like(theta)
    return torch.stack(
        (torch.cos(theta), zero, zero, rotation * torch.sin(theta)), dim=-1
    )


def exchange_states(
    times: Tensor, *, exchange: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """Exact zero-field exchange from |01>, omitting an irrelevant global phase."""
    if (
        times.is_complex()
        or not torch.isfinite(times).all()
        or not isfinite(exchange)
        or not isfinite(hbar)
        or hbar <= 0
    ):
        raise ValueError("times and exchange must be finite; hbar must be positive")
    angle = exchange * times.to(dtype=torch.float64) / (2 * hbar)
    zero = torch.zeros_like(angle)
    return torch.stack((zero, torch.cos(angle), -1j * torch.sin(angle), zero), dim=-1)


def pauli_correlations(rho: Tensor) -> Tensor:
    """Return <sigma_a tensor sigma_b> with shape (..., 3, 3)."""
    validate_density(rho)
    if rho.shape[-1] != 4:
        raise ValueError("Pauli correlations require two qubits")
    sigma = torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
        device=rho.device,
    )
    operators = torch.stack(
        [torch.stack([torch.kron(a, b) for b in sigma]) for a in sigma]
    )
    return torch.einsum("...ij,abji->...ab", rho.to(torch.complex128), operators).real


def main() -> None:
    rho = pure_density(bell_states())
    print("Bell order: Phi+, Phi-, Psi+, Psi-")
    print("Joint purity:", purity(rho).tolist())
    print("Local entropy (bits):", entropy(partial_trace(rho)).tolist())
    basis = torch.eye(4, dtype=torch.complex128)[[0, 3]]
    mixture = mixed_density(basis, torch.tensor([0.5, 0.5], dtype=torch.float64))
    print(
        "Classical mixture: joint entropy =",
        float(entropy(mixture)),
        "; xx correlation =",
        float(pauli_correlations(mixture)[0, 0]),
    )


if __name__ == "__main__":
    main()
