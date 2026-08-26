"""Reusable PyTorch solver for real one-dimensional stationary potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

PotentialFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class StationaryResult:
    """Low-energy eigenstates on a finite interior spatial grid."""

    grid: Tensor
    spacing: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


@dataclass(frozen=True)
class BatchedStationaryResult:
    """Eigenstates for a batch of sampled potentials on one grid."""

    grid: Tensor
    spacing: float
    potentials: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonians: Tensor


def _validate_inputs(
    num_points: int,
    num_states: int,
    x_min: float,
    x_max: float,
    mass: float,
    hbar: float,
) -> None:
    values = torch.tensor([x_min, x_max, mass, hbar], dtype=torch.float64)
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("domain, mass, and hbar must be finite")
    if x_min >= x_max:
        raise ValueError("x_min must be smaller than x_max")
    if mass <= 0.0 or hbar <= 0.0:
        raise ValueError("mass and hbar must be positive")


def make_grid(
    num_points: int,
    x_min: float = -8.0,
    x_max: float = 8.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return interior points of [x_min, x_max] and their uniform spacing."""
    _validate_inputs(num_points, 1, x_min, x_max, 1.0, 1.0)
    spacing = (x_max - x_min) / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return x_min + spacing * indices, spacing


def kinetic_energy_matrix(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Construct the second-order finite-difference kinetic operator."""
    _validate_inputs(num_points, 1, 0.0, spacing, mass, hbar)
    diagonal = torch.full((num_points,), -2.0, dtype=torch.float64, device=device)
    off_diagonal = torch.ones(num_points - 1, dtype=torch.float64, device=device)
    second_derivative = (
        torch.diag(diagonal)
        + torch.diag(off_diagonal, diagonal=1)
        + torch.diag(off_diagonal, diagonal=-1)
    ) / spacing**2
    return -(hbar**2 / (2.0 * mass)) * second_derivative


def evaluate_potential(grid: Tensor, potential_function: PotentialFunction) -> Tensor:
    """Evaluate and validate a real, finite potential on the spatial grid."""
    raw_potential = potential_function(grid)
    if not isinstance(raw_potential, Tensor):
        raise TypeError("potential_function must return a torch.Tensor")
    if raw_potential.shape != grid.shape:
        raise ValueError("potential_function must return one value per grid point")
    if torch.is_complex(raw_potential):
        raise ValueError("potential_function must return real values")
    potential = raw_potential.to(dtype=torch.float64, device=grid.device)
    if not bool(torch.isfinite(potential).all()):
        raise ValueError("potential values must be finite")
    return potential


def build_hamiltonian(
    potential_function: PotentialFunction,
    num_points: int = 300,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    """Construct H = T + diag(V) for a supplied potential function."""
    _validate_inputs(num_points, 1, x_min, x_max, mass, hbar)
    grid, spacing = make_grid(num_points, x_min, x_max, device)
    kinetic = kinetic_energy_matrix(num_points, spacing, mass, hbar, device)
    potential = evaluate_potential(grid, potential_function)
    return grid, spacing, potential, kinetic + torch.diag(potential)


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize wavefunction columns using the discrete spatial integral."""
    norms = torch.sqrt(spacing * torch.sum(torch.abs(wavefunctions) ** 2, dim=-2))
    return wavefunctions / norms.unsqueeze(-2)


def solve_stationary(
    potential_function: PotentialFunction,
    num_points: int = 300,
    num_states: int = 6,
    x_min: float = -8.0,
    x_max: float = 8.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> StationaryResult:
    """Solve for the lowest states of an arbitrary real potential."""
    _validate_inputs(num_points, num_states, x_min, x_max, mass, hbar)
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        potential_function, num_points, x_min, x_max, mass, hbar, device
    )
    energies, wavefunctions = torch.linalg.eigh(hamiltonian)
    selected = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return StationaryResult(
        grid,
        spacing,
        potential,
        energies[:num_states],
        selected,
        hamiltonian,
    )


def solve_potential_batch(
    grid: Tensor,
    spacing: float,
    potentials: Tensor,
    num_states: int = 4,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> BatchedStationaryResult:
    """Solve a batch of sampled potentials with shape (batch, num_points)."""
    if grid.ndim != 1 or potentials.ndim != 2:
        raise ValueError("grid must be 1D and potentials must be 2D")
    num_points = grid.shape[0]
    if potentials.shape[1] != num_points:
        raise ValueError("each potential must have one value per grid point")
    if potentials.shape[0] < 1:
        raise ValueError("the potential batch must not be empty")
    if potentials.device != grid.device:
        raise ValueError("grid and potentials must be on the same device")
    if torch.is_complex(potentials) or not bool(torch.isfinite(potentials).all()):
        raise ValueError("potentials must contain real, finite values")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    kinetic = kinetic_energy_matrix(num_points, spacing, mass, hbar, device=grid.device)
    sampled = potentials.to(dtype=torch.float64)
    hamiltonians = kinetic.unsqueeze(0) + torch.diag_embed(sampled)
    energies, wavefunctions = torch.linalg.eigh(hamiltonians)
    selected = normalize_wavefunctions(wavefunctions[:, :, :num_states], spacing)
    return BatchedStationaryResult(
        grid,
        spacing,
        sampled,
        energies[:, :num_states],
        selected,
        hamiltonians,
    )


def expectation_position(result: StationaryResult) -> Tensor:
    """Return <x> for every stored eigenstate."""
    density = torch.abs(result.wavefunctions) ** 2
    return result.spacing * torch.sum(result.grid[:, None] * density, dim=0)


def residual_norms(result: StationaryResult) -> Tensor:
    """Return discrete L2 norms of H psi_n - E_n psi_n."""
    residuals = (
        result.hamiltonian @ result.wavefunctions
        - result.wavefunctions * result.energies[None, :]
    )
    return torch.sqrt(result.spacing * torch.sum(torch.abs(residuals) ** 2, dim=0))


def shifted_harmonic_potential(
    grid: Tensor,
    center: float = 0.75,
    angular_frequency: float = 1.25,
    mass: float = 1.0,
    offset: float = 0.4,
) -> Tensor:
    """Return a shifted harmonic potential used as an analytical reference."""
    if angular_frequency <= 0.0 or mass <= 0.0:
        raise ValueError("angular_frequency and mass must be positive")
    return offset + 0.5 * mass * angular_frequency**2 * (grid - center) ** 2


def analytical_harmonic_energies(
    num_states: int,
    angular_frequency: float = 1.25,
    hbar: float = 1.0,
    offset: float = 0.4,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return exact energies for the shifted harmonic reference potential."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if angular_frequency <= 0.0 or hbar <= 0.0:
        raise ValueError("angular_frequency and hbar must be positive")
    quantum_numbers = torch.arange(num_states, dtype=torch.float64, device=device)
    return offset + hbar * angular_frequency * (quantum_numbers + 0.5)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    potential: PotentialFunction = shifted_harmonic_potential
    result = solve_stationary(potential, device=device)
    exact = analytical_harmonic_energies(len(result.energies), device=device)
    positions = expectation_position(result)
    print(f"device: {device}")
    print("state  numerical energy  exact energy     abs error       <x>")
    for state in range(len(result.energies)):
        error = torch.abs(result.energies[state] - exact[state])
        print(
            f"{state:>5}  {result.energies[state].item():>16.8f}  "
            f"{exact[state].item():>12.8f}  {error.item():>12.3e}  "
            f"{positions[state].item():>9.5f}"
        )


if __name__ == "__main__":
    main()
