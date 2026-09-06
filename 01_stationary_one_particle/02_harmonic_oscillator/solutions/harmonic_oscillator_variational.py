"""Early PyTorch autograd and basis-expansion exercises in oscillator units."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
import torch
from harmonic_oscillator_torch_solution import build_hamiltonian
from numpy.typing import NDArray
from torch import Tensor


@dataclass(frozen=True)
class VariationalResult:
    """Optimized Gaussian exponent and its continuum energy."""

    alpha: float
    energy: float


def gaussian_energy(log_alpha: Tensor, coupling: float = 0.0) -> Tensor:
    """Return <H> for psi ∝ exp(-alpha x²/2), H = (p²+x²)/2 + coupling*x⁴."""
    if not isfinite(coupling) or coupling < 0:
        raise ValueError("coupling must be finite and nonnegative")
    alpha = torch.exp(log_alpha)
    return (alpha + alpha.reciprocal()) / 4 + 3 * coupling / (4 * alpha**2)


def optimize_gaussian(
    coupling: float = 0.0,
    initial_alpha: float = 0.4,
    num_steps: int = 300,
    learning_rate: float = 0.05,
    device: str | torch.device = "cpu",
) -> VariationalResult:
    """Use autograd and Adam; log(alpha) keeps the trial state normalizable."""
    if not isfinite(initial_alpha) or initial_alpha <= 0 or num_steps < 1:
        raise ValueError(
            "initial_alpha must be positive and finite; num_steps positive"
        )
    if not isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    log_alpha = torch.tensor(initial_alpha, dtype=torch.float64, device=device).log()
    log_alpha.requires_grad_()
    optimizer = torch.optim.Adam([log_alpha], lr=learning_rate)
    for _ in range(num_steps):
        optimizer.zero_grad()
        energy = gaussian_energy(log_alpha, coupling)
        torch.autograd.backward(energy)
        optimizer.step()
    return VariationalResult(
        torch.exp(log_alpha).detach().item(),
        gaussian_energy(log_alpha, coupling).detach().item(),
    )


def oscillator_basis_hamiltonian(
    basis_size: int,
    coupling: float = 0.1,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Project the continuum anharmonic Hamiltonian onto oscillator states."""
    if basis_size < 1 or not isfinite(coupling) or coupling < 0:
        raise ValueError("basis_size must be positive; coupling finite and nonnegative")
    # Four ladder operations can leave the retained basis and return. Padding
    # before taking x^4 preserves those paths, including at the cutoff.
    levels = torch.arange(1, basis_size + 4, dtype=torch.float64, device=device)
    off = torch.sqrt(levels / 2)
    position = torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    quartic = torch.linalg.matrix_power(position, 4)[:basis_size, :basis_size]
    harmonic = torch.diag(
        torch.arange(basis_size, dtype=torch.float64, device=device) + 0.5
    )
    hamiltonian: Tensor = harmonic + coupling * quartic
    return hamiltonian


def numpy_basis_energies(basis_size: int, coupling: float = 0.1) -> NDArray[np.float64]:
    """Small NumPy comparison for the same projected basis problem."""
    if basis_size < 1 or not isfinite(coupling) or coupling < 0:
        raise ValueError("basis_size must be positive; coupling finite and nonnegative")
    off = np.sqrt(np.arange(1, basis_size + 4, dtype=np.float64) / 2)
    position = np.diag(off, k=1) + np.diag(off, k=-1)
    quartic = np.linalg.matrix_power(position, 4)[:basis_size, :basis_size]
    hamiltonian = (
        np.diag(np.arange(basis_size, dtype=np.float64) + 0.5) + coupling * quartic
    )
    return np.linalg.eigvalsh(hamiltonian)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PyTorch variational exercise on {device}; m = hbar = omega = 1")
    for coupling in (0.0, 0.1):
        result = optimize_gaussian(coupling=coupling, device=device)
        print(
            f"lambda={coupling:.1f}: alpha={result.alpha:.8f}, Gaussian E={result.energy:.10f}"
        )
    print("Basis convergence for lambda = 0.1")
    print("basis size    Torch E0       NumPy delta")
    for size in (4, 8, 12, 20):
        energies = torch.linalg.eigvalsh(
            oscillator_basis_hamiltonian(size, device=device)
        )
        reference = numpy_basis_energies(size)
        print(
            f"{size:>10}  {energies[0].item():>12.9f}  {abs(energies[0].item() - reference[0]):>12.3e}"
        )

    print("Grid comparison for lambda = 0.1 on [-8, 8]")
    print("points    spacing       grid E0")
    for points in (80, 160, 320):
        grid, spacing, _, harmonic = build_hamiltonian(points, device=device)
        hamiltonian = harmonic + torch.diag(0.1 * grid**4)
        energy = torch.linalg.eigvalsh(hamiltonian)[0].item()
        print(f"{points:>6}  {spacing:>10.3e}  {energy:>12.9f}")
    print("Finite differences need not approach the continuum energy from above.")


if __name__ == "__main__":
    main()
