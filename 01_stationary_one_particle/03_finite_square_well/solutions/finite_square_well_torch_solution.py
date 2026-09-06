"""PyTorch reference solution of the one-dimensional finite square well."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FiniteSquareWellResult:
    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def _validate_inputs(
    num_points: int,
    num_states: int,
    x_max: float,
    half_width: float,
    depth: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if x_max <= 0 or half_width <= 0 or depth <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("lengths, depth, mass, and hbar must be positive")
    if half_width >= x_max:
        raise ValueError("half_width must be smaller than x_max")


def make_grid(
    num_points: int,
    x_max: float = 8.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    spacing = 2.0 * x_max / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return -x_max + spacing * indices, spacing


def finite_square_well_potential(
    grid: Tensor,
    half_width: float = 1.0,
    depth: float = 20.0,
) -> Tensor:
    if half_width <= 0 or depth <= 0:
        raise ValueError("half_width and depth must be positive")
    return torch.where(torch.abs(grid) < half_width, -depth, 0.0)


def build_hamiltonian(
    num_points: int,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    _validate_inputs(num_points, 1, x_max, half_width, depth, mass, hbar)
    grid, spacing = make_grid(num_points, x_max, device)
    main = torch.full((num_points,), -2.0, dtype=torch.float64, device=device)
    off = torch.ones(num_points - 1, dtype=torch.float64, device=device)
    second_derivative = (
        torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    ) / spacing**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = finite_square_well_potential(grid, half_width, depth)
    return grid, spacing, potential, kinetic + torch.diag(potential)


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    norms = torch.sqrt(spacing * torch.sum(torch.abs(wavefunctions) ** 2, dim=0))
    return wavefunctions / norms


def solve_finite_square_well(
    num_points: int = 400,
    num_states: int = 8,
    x_max: float = 8.0,
    half_width: float = 1.0,
    depth: float = 20.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> FiniteSquareWellResult:
    _validate_inputs(num_points, num_states, x_max, half_width, depth, mass, hbar)
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, x_max, half_width, depth, mass, hbar, device
    )
    energies, wavefunctions = torch.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return FiniteSquareWellResult(
        grid, spacing, potential, energies[:num_states], wavefunctions, hamiltonian
    )


def bound_state_mask(result: FiniteSquareWellResult) -> Tensor:
    return result.energies < 0.0


def probability_inside_well(
    result: FiniteSquareWellResult,
    half_width: float = 1.0,
) -> Tensor:
    inside = torch.abs(result.grid) < half_width
    density = torch.abs(result.wavefunctions[inside, :]) ** 2
    return result.spacing * torch.sum(density, dim=0)


def residual_norms(result: FiniteSquareWellResult) -> Tensor:
    """Measure algebraic error in energy units, using the spatial integration weight."""
    residual = (
        result.hamiltonian @ result.wavefunctions
        - result.wavefunctions * result.energies[None, :]
    )
    return torch.sqrt(result.spacing * torch.sum(torch.abs(residual) ** 2, dim=0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_finite_square_well(device=device)
    print(f"device: {device}")
    print(f"bound states among requested states: {int(bound_state_mask(result).sum())}")
    print("state  energy          bound  probability inside")
    probabilities = probability_inside_well(result)
    for state in range(len(result.energies)):
        print(
            f"{state:>5}  {result.energies[state].item():>14.8f}  "
            f"{bool(result.energies[state] < 0)!s:>5}  "
            f"{probabilities[state].item():>18.8f}"
        )


if __name__ == "__main__":
    main()
