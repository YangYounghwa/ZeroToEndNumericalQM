"""PyTorch reference solution of the one-dimensional infinite square well."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class InfiniteWellResult:
    """Numerical eigenstates on the interior spatial grid."""

    grid: Tensor
    spacing: float
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def _validate_inputs(
    num_points: int,
    num_states: int,
    length: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if length <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("length, mass, and hbar must be positive")


def make_grid(
    num_points: int,
    length: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return the interior grid and uniform grid spacing."""
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if length <= 0:
        raise ValueError("length must be positive")

    spacing = length / (num_points + 1)
    grid = spacing * torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return grid, spacing


def build_hamiltonian(
    num_points: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor]:
    """Construct the finite-difference Hamiltonian on the selected device."""
    _validate_inputs(num_points, 1, length, mass, hbar)
    grid, spacing = make_grid(num_points, length, device)

    main_diagonal = torch.full((num_points,), -2.0, dtype=torch.float64, device=device)
    off_diagonal = torch.ones(num_points - 1, dtype=torch.float64, device=device)
    second_derivative = (
        torch.diag(main_diagonal)
        + torch.diag(off_diagonal, diagonal=1)
        + torch.diag(off_diagonal, diagonal=-1)
    ) / spacing**2
    hamiltonian = -(hbar**2 / (2.0 * mass)) * second_derivative
    return grid, spacing, hamiltonian


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = torch.sqrt(spacing * torch.sum(torch.abs(wavefunctions) ** 2, dim=0))
    return wavefunctions / norms


def solve_infinite_well(
    num_points: int = 200,
    num_states: int = 4,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> InfiniteWellResult:
    """Solve for the lowest-energy states of an infinite square well."""
    _validate_inputs(num_points, num_states, length, mass, hbar)
    grid, spacing, hamiltonian = build_hamiltonian(
        num_points, length, mass, hbar, device
    )
    energies, wavefunctions = torch.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)

    return InfiniteWellResult(
        grid=grid,
        spacing=spacing,
        energies=energies[:num_states],
        wavefunctions=wavefunctions,
        hamiltonian=hamiltonian,
    )


def analytical_energies(
    num_states: int,
    length: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return exact energies for quantum numbers 1 through num_states."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if length <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("length, mass, and hbar must be positive")
    quantum_numbers = torch.arange(
        1, num_states + 1, dtype=torch.float64, device=device
    )
    pi = torch.tensor(torch.pi, dtype=torch.float64, device=device)
    return quantum_numbers**2 * pi**2 * hbar**2 / (2.0 * mass * length**2)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_infinite_well(device=device)
    exact = analytical_energies(len(result.energies), device=device)

    print(f"device: {device}")
    print("state  numerical energy  analytical energy  relative error")
    for state in range(len(result.energies)):
        numerical = result.energies[state].item()
        analytical = exact[state].item()
        relative_error = abs(numerical - analytical) / analytical
        print(
            f"{state + 1:>5}  {numerical:>16.8f}  {analytical:>17.8f}  "
            f"{relative_error:>14.3e}"
        )


if __name__ == "__main__":
    main()
