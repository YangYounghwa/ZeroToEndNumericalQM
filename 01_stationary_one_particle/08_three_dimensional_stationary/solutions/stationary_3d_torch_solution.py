"""Small dense PyTorch reference for stationary three-dimensional potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

Potential3D = Callable[[Tensor, Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class Stationary3DResult:
    """Low-energy states stored with shape (z, y, x, state)."""

    x_grid: Tensor
    y_grid: Tensor
    z_grid: Tensor
    spacing_x: float
    spacing_y: float
    spacing_z: float
    potential: Tensor
    energies: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


@dataclass(frozen=True)
class BatchedStationary3DResult:
    energies: Tensor
    wavefunctions: Tensor
    hamiltonians: Tensor


def make_axis(
    num_points: int,
    minimum: float,
    maximum: float,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
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
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = torch.full((num_points,), 2.0 * scale, dtype=torch.float64, device=device)
    off = torch.full((num_points - 1,), -scale, dtype=torch.float64, device=device)
    return torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)


def anisotropic_oscillator_potential(
    x_mesh: Tensor,
    y_mesh: Tensor,
    z_mesh: Tensor,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.3,
    angular_frequency_z: float = 1.7,
    mass: float = 1.0,
) -> Tensor:
    return (
        0.5
        * mass
        * (
            angular_frequency_x**2 * x_mesh**2
            + angular_frequency_y**2 * y_mesh**2
            + angular_frequency_z**2 * z_mesh**2
        )
    )


def build_hamiltonian(
    potential_function: Potential3D,
    num_x: int = 7,
    num_y: int = 6,
    num_z: int = 5,
    x_min: float = -5.0,
    x_max: float = 5.0,
    y_min: float = -4.5,
    y_max: float = 4.5,
    z_min: float = -4.0,
    z_max: float = 4.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor, float, float, float, Tensor, Tensor]:
    """Build a small dense 3D Kronecker-sum Hamiltonian."""
    x_grid, dx = make_axis(num_x, x_min, x_max, device)
    y_grid, dy = make_axis(num_y, y_min, y_max, device)
    z_grid, dz = make_axis(num_z, z_min, z_max, device)
    z_mesh, y_mesh, x_mesh = torch.meshgrid(z_grid, y_grid, x_grid, indexing="ij")
    potential = potential_function(x_mesh, y_mesh, z_mesh)
    expected_shape = (num_z, num_y, num_x)
    if potential.shape != expected_shape or torch.is_complex(potential):
        raise ValueError("potential must be real with shape (num_z, num_y, num_x)")
    potential = potential.to(dtype=torch.float64, device=device)
    kinetic_x = kinetic_1d(num_x, dx, mass, hbar, device)
    kinetic_y = kinetic_1d(num_y, dy, mass, hbar, device)
    kinetic_z = kinetic_1d(num_z, dz, mass, hbar, device)
    identity_x = torch.eye(num_x, dtype=torch.float64, device=device)
    identity_y = torch.eye(num_y, dtype=torch.float64, device=device)
    identity_z = torch.eye(num_z, dtype=torch.float64, device=device)
    term_x = torch.kron(torch.kron(identity_z, identity_y), kinetic_x)
    term_y = torch.kron(torch.kron(identity_z, kinetic_y), identity_x)
    term_z = torch.kron(torch.kron(kinetic_z, identity_y), identity_x)
    hamiltonian = term_x + term_y + term_z + torch.diag(potential.reshape(-1))
    return x_grid, y_grid, z_grid, dx, dy, dz, potential, hamiltonian


def solve_stationary_3d(
    potential_function: Potential3D = anisotropic_oscillator_potential,
    num_x: int = 7,
    num_y: int = 6,
    num_z: int = 5,
    num_states: int = 5,
    device: str | torch.device = "cpu",
) -> Stationary3DResult:
    """Solve a deliberately small dense 3D reference problem."""
    values = build_hamiltonian(potential_function, num_x, num_y, num_z, device=device)
    x_grid, y_grid, z_grid, dx, dy, dz, potential, hamiltonian = values
    energies, vectors = torch.linalg.eigh(hamiltonian)
    selected = vectors[:, :num_states]
    norms = torch.sqrt(dx * dy * dz * torch.sum(torch.abs(selected) ** 2, dim=0))
    wavefunctions = (selected / norms).reshape(num_z, num_y, num_x, num_states)
    return Stationary3DResult(
        x_grid,
        y_grid,
        z_grid,
        dx,
        dy,
        dz,
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
    spacing_z: float,
    num_states: int = 3,
) -> BatchedStationary3DResult:
    """Solve sampled potentials with shape (batch, z, y, x)."""
    if potentials.ndim != 4 or potentials.shape[0] < 1:
        raise ValueError("potentials must have shape (batch, z, y, x)")
    total_points = potentials.shape[1] * potentials.shape[2] * potentials.shape[3]
    if kinetic_hamiltonian.shape != (total_points, total_points):
        raise ValueError("kinetic_hamiltonian size does not match potentials")
    hamiltonians = kinetic_hamiltonian.unsqueeze(0) + torch.diag_embed(
        potentials.reshape(potentials.shape[0], -1)
    )
    energies, vectors = torch.linalg.eigh(hamiltonians)
    selected = vectors[:, :, :num_states]
    norms = torch.sqrt(
        spacing_x * spacing_y * spacing_z * torch.sum(torch.abs(selected) ** 2, dim=1)
    )
    return BatchedStationary3DResult(
        energies[:, :num_states], selected / norms[:, None, :], hamiltonians
    )


def expectation_position(
    result: Stationary3DResult,
) -> tuple[Tensor, Tensor, Tensor]:
    density = torch.abs(result.wavefunctions) ** 2
    weight = result.spacing_x * result.spacing_y * result.spacing_z
    x_values = weight * torch.sum(
        density * result.x_grid[None, None, :, None], dim=(0, 1, 2)
    )
    y_values = weight * torch.sum(
        density * result.y_grid[None, :, None, None], dim=(0, 1, 2)
    )
    z_values = weight * torch.sum(
        density * result.z_grid[:, None, None, None], dim=(0, 1, 2)
    )
    return x_values, y_values, z_values


def residual_norms(result: Stationary3DResult) -> Tensor:
    vectors = result.wavefunctions.reshape(-1, len(result.energies))
    residuals = result.hamiltonian @ vectors - vectors * result.energies[None, :]
    weight = result.spacing_x * result.spacing_y * result.spacing_z
    return torch.sqrt(weight * torch.sum(torch.abs(residuals) ** 2, dim=0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_stationary_3d(device=device)
    print(f"device: {device}")
    print("state   energy")
    for state in range(len(result.energies)):
        print(f"{state:>5}  {result.energies[state].item():>11.7f}")


if __name__ == "__main__":
    main()
