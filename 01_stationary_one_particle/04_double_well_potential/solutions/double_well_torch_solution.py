"""PyTorch reference solution of a symmetric quartic double well."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class DoubleWellResult:
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
    separation: float,
    barrier_height: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if x_max <= 0 or separation <= 0 or barrier_height <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("lengths, barrier_height, mass, and hbar must be positive")
    if separation >= x_max:
        raise ValueError("separation must be smaller than x_max")


def make_grid(
    num_points: int,
    x_max: float = 6.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    spacing = 2.0 * x_max / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return -x_max + spacing * indices, spacing


def double_well_potential(
    grid: Tensor,
    separation: float = 1.5,
    barrier_height: float = 8.0,
) -> Tensor:
    if separation <= 0 or barrier_height <= 0:
        raise ValueError("separation and barrier_height must be positive")
    return barrier_height * ((grid / separation) ** 2 - 1.0) ** 2


def build_hamiltonian(
    num_points: int,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    _validate_inputs(num_points, 1, x_max, separation, barrier_height, mass, hbar)
    grid, spacing = make_grid(num_points, x_max, device)
    main = torch.full((num_points,), -2.0, dtype=torch.float64, device=device)
    off = torch.ones(num_points - 1, dtype=torch.float64, device=device)
    second_derivative = (
        torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    ) / spacing**2
    kinetic = -(hbar**2 / (2.0 * mass)) * second_derivative
    potential = double_well_potential(grid, separation, barrier_height)
    return grid, spacing, potential, kinetic + torch.diag(potential)


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    norms = torch.sqrt(spacing * torch.sum(torch.abs(wavefunctions) ** 2, dim=0))
    return wavefunctions / norms


def solve_double_well(
    num_points: int = 300,
    num_states: int = 6,
    x_max: float = 6.0,
    separation: float = 1.5,
    barrier_height: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> DoubleWellResult:
    _validate_inputs(
        num_points,
        num_states,
        x_max,
        separation,
        barrier_height,
        mass,
        hbar,
    )
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, x_max, separation, barrier_height, mass, hbar, device
    )
    energies, wavefunctions = torch.linalg.eigh(hamiltonian)
    wavefunctions = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return DoubleWellResult(
        grid, spacing, potential, energies[:num_states], wavefunctions, hamiltonian
    )


def tunneling_splitting(result: DoubleWellResult) -> Tensor:
    if len(result.energies) < 2:
        raise ValueError("result must contain at least two states")
    return result.energies[1] - result.energies[0]


def probability_left(result: DoubleWellResult) -> Tensor:
    density = torch.abs(result.wavefunctions[result.grid < 0.0, :]) ** 2
    return result.spacing * torch.sum(density, dim=0)


def localized_pair(result: DoubleWellResult) -> tuple[Tensor, Tensor]:
    if result.wavefunctions.shape[1] < 2:
        raise ValueError("result must contain at least two states")
    even = result.wavefunctions[:, 0]
    odd = result.wavefunctions[:, 1]
    left = (even + odd) / torch.sqrt(
        torch.tensor(2.0, dtype=torch.float64, device=result.grid.device)
    )
    right = (even - odd) / torch.sqrt(
        torch.tensor(2.0, dtype=torch.float64, device=result.grid.device)
    )
    if torch.sum(torch.abs(left[result.grid < 0.0]) ** 2) < torch.sum(
        torch.abs(right[result.grid < 0.0]) ** 2
    ):
        left, right = right, left
    return left, right


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_double_well(device=device)
    print(f"device: {device}")
    print("state  energy          parity  probability left")
    probabilities = probability_left(result)
    for state in range(len(result.energies)):
        parity = "even" if state % 2 == 0 else "odd"
        print(
            f"{state:>5}  {result.energies[state].item():>14.8f}  {parity:>6}  "
            f"{probabilities[state].item():>16.8f}"
        )
    print(
        f"\nground-pair tunneling splitting: {tunneling_splitting(result).item():.8e}"
    )


if __name__ == "__main__":
    main()
