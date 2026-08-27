"""PyTorch solver for stationary two-dimensional potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

Potential2D = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Stationary2DResult:
    """Low-energy states stored with shape (y, x, state)."""

    x_grid: Tensor
    y_grid: Tensor
    spacing_x: float
    spacing_y: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


@dataclass(frozen=True)
class BatchedStationary2DResult:
    """Results for sampled potentials with shape (batch, y, x)."""

    energies: Tensor
    wavefunctions: Tensor
    hamiltonians: Tensor


def make_axis(
    num_points: int,
    minimum: float,
    maximum: float,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return one float64 interior coordinate axis."""
    if num_points < 3 or minimum >= maximum:
        raise ValueError("require num_points >= 3 and minimum < maximum")
    spacing = (maximum - minimum) / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return minimum + spacing * indices, spacing


def kinetic_1d(
    num_points: int,
    spacing: float,
    mass: float,
    hbar: float,
    device: str | torch.device,
) -> Tensor:
    """Return one dense float64 kinetic operator."""
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = torch.full((num_points,), 2.0 * scale, dtype=torch.float64, device=device)
    off = torch.full((num_points - 1,), -scale, dtype=torch.float64, device=device)
    return torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)


def anisotropic_oscillator_potential(
    x_mesh: Tensor,
    y_mesh: Tensor,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.4,
    mass: float = 1.0,
) -> Tensor:
    """Return a two-dimensional anisotropic harmonic potential."""
    return (
        0.5
        * mass
        * (angular_frequency_x**2 * x_mesh**2 + angular_frequency_y**2 * y_mesh**2)
    )


def build_hamiltonian(
    potential_function: Potential2D,
    num_x: int = 18,
    num_y: int = 16,
    x_min: float = -7.0,
    x_max: float = 7.0,
    y_min: float = -6.0,
    y_max: float = 6.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, float, float, Tensor, Tensor]:
    """Build the dense Kronecker-sum Hamiltonian."""
    x_grid, dx = make_axis(num_x, x_min, x_max, device)
    y_grid, dy = make_axis(num_y, y_min, y_max, device)
    y_mesh, x_mesh = torch.meshgrid(y_grid, x_grid, indexing="ij")
    potential = potential_function(x_mesh, y_mesh)
    if potential.shape != (num_y, num_x) or torch.is_complex(potential):
        raise ValueError("potential must be real with shape (num_y, num_x)")
    potential = potential.to(dtype=torch.float64, device=device)
    if not bool(torch.isfinite(potential).all()):
        raise ValueError("potential must contain finite values")
    kinetic_x = kinetic_1d(num_x, dx, mass, hbar, device)
    kinetic_y = kinetic_1d(num_y, dy, mass, hbar, device)
    identity_x = torch.eye(num_x, dtype=torch.float64, device=device)
    identity_y = torch.eye(num_y, dtype=torch.float64, device=device)
    kinetic = torch.kron(identity_y, kinetic_x) + torch.kron(kinetic_y, identity_x)
    hamiltonian = kinetic + torch.diag(potential.reshape(-1))
    return x_grid, y_grid, dx, dy, potential, hamiltonian


def solve_stationary_2d(
    potential_function: Potential2D = anisotropic_oscillator_potential,
    num_x: int = 18,
    num_y: int = 16,
    num_states: int = 6,
    x_min: float = -7.0,
    x_max: float = 7.0,
    y_min: float = -6.0,
    y_max: float = 6.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Stationary2DResult:
    """Solve a two-dimensional potential using dense PyTorch eigh."""
    if not 1 <= num_states <= num_x * num_y:
        raise ValueError("num_states is outside the grid Hilbert-space size")
    x_grid, y_grid, dx, dy, potential, hamiltonian = build_hamiltonian(
        potential_function,
        num_x,
        num_y,
        x_min,
        x_max,
        y_min,
        y_max,
        mass,
        hbar,
        device,
    )
    energies, vectors = torch.linalg.eigh(hamiltonian)
    selected = vectors[:, :num_states]
    norms = torch.sqrt(dx * dy * torch.sum(torch.abs(selected) ** 2, dim=0))
    wavefunctions = (selected / norms).reshape(num_y, num_x, num_states)
    return Stationary2DResult(
        x_grid,
        y_grid,
        dx,
        dy,
        potential,
        energies[:num_states],
        wavefunctions,
        hamiltonian,
    )


def solve_potential_batch(
    kinetic_hamiltonian: Tensor,
    potentials: Tensor,
    spacing_x: float,
    spacing_y: float,
    num_states: int = 4,
) -> BatchedStationary2DResult:
    """Solve sampled potentials with shape (batch, y, x)."""
    if potentials.ndim != 3 or potentials.shape[0] < 1:
        raise ValueError("potentials must have shape (batch, y, x)")
    total_points = potentials.shape[1] * potentials.shape[2]
    if kinetic_hamiltonian.shape != (total_points, total_points):
        raise ValueError("kinetic_hamiltonian size does not match potentials")
    hamiltonians = kinetic_hamiltonian.unsqueeze(0) + torch.diag_embed(
        potentials.reshape(potentials.shape[0], -1)
    )
    energies, vectors = torch.linalg.eigh(hamiltonians)
    selected = vectors[:, :, :num_states]
    norms = torch.sqrt(
        spacing_x * spacing_y * torch.sum(torch.abs(selected) ** 2, dim=1)
    )
    wavefunctions = selected / norms[:, None, :]
    return BatchedStationary2DResult(
        energies[:, :num_states], wavefunctions, hamiltonians
    )


def expectation_position(result: Stationary2DResult) -> tuple[Tensor, Tensor]:
    """Return (<x>, <y>) for every state."""
    density = torch.abs(result.wavefunctions) ** 2
    weight = result.spacing_x * result.spacing_y
    x_values = weight * torch.sum(density * result.x_grid[None, :, None], dim=(0, 1))
    y_values = weight * torch.sum(density * result.y_grid[:, None, None], dim=(0, 1))
    return x_values, y_values


def residual_norms(result: Stationary2DResult) -> Tensor:
    vectors = result.wavefunctions.reshape(-1, len(result.energies))
    residuals = result.hamiltonian @ vectors - vectors * result.energies[None, :]
    weight = result.spacing_x * result.spacing_y
    return torch.sqrt(weight * torch.sum(torch.abs(residuals) ** 2, dim=0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_stationary_2d(device=device)
    x_values, y_values = expectation_position(result)
    print(f"device: {device}")
    print("state   energy          <x>          <y>")
    for state in range(len(result.energies)):
        print(
            f"{state:>5}  {result.energies[state].item():>11.7f}  "
            f"{x_values[state].item():>11.2e}  {y_values[state].item():>11.2e}"
        )


if __name__ == "__main__":
    main()
