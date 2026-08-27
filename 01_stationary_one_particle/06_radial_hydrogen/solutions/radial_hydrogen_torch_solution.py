"""PyTorch solver for the reduced radial hydrogen equation."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RadialResult:
    """Low-energy reduced radial states for one angular-momentum sector."""

    grid: Tensor
    spacing: float
    effective_potential: Tensor
    energies: Tensor
    radial_wavefunctions: Tensor
    hamiltonian: Tensor
    angular_momentum: int
    nuclear_charge: float


@dataclass(frozen=True)
class BatchedRadialResult:
    """Radial eigenstates for a batch of angular-momentum sectors."""

    grid: Tensor
    spacing: float
    angular_momenta: Tensor
    effective_potentials: Tensor
    energies: Tensor
    radial_wavefunctions: Tensor
    hamiltonians: Tensor


def _validate_inputs(
    num_points: int,
    num_states: int,
    r_max: float,
    angular_momentum: int,
    nuclear_charge: float,
    mass: float,
    hbar: float,
) -> None:
    values = torch.tensor([r_max, nuclear_charge, mass, hbar], dtype=torch.float64)
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not 1 <= num_states <= num_points:
        raise ValueError("num_states must be between 1 and num_points")
    if not isinstance(angular_momentum, int) or angular_momentum < 0:
        raise ValueError("angular_momentum must be a nonnegative integer")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("physical parameters must be finite")
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")


def make_radial_grid(
    num_points: int,
    r_max: float = 40.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return float64 interior radial points on the requested device."""
    _validate_inputs(num_points, 1, r_max, 0, 1.0, 1.0, 1.0)
    spacing = r_max / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return spacing * indices, spacing


def effective_potential(
    grid: Tensor,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Return the Coulomb plus centrifugal effective potential."""
    if not isinstance(angular_momentum, int) or angular_momentum < 0:
        raise ValueError("angular_momentum must be a nonnegative integer")
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")
    centrifugal = (
        hbar**2 * angular_momentum * (angular_momentum + 1) / (2.0 * mass * grid**2)
    )
    return centrifugal - nuclear_charge / grid


def build_hamiltonian(
    num_points: int = 400,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    """Construct the dense tridiagonal radial Hamiltonian."""
    _validate_inputs(num_points, 1, r_max, angular_momentum, nuclear_charge, mass, hbar)
    grid, spacing = make_radial_grid(num_points, r_max, device)
    potential = effective_potential(grid, angular_momentum, nuclear_charge, mass, hbar)
    kinetic_diagonal = hbar**2 / (mass * spacing**2)
    kinetic_off_diagonal = -(hbar**2) / (2.0 * mass * spacing**2)
    main = (
        torch.full((num_points,), kinetic_diagonal, dtype=torch.float64, device=device)
        + potential
    )
    off = torch.full(
        (num_points - 1,),
        kinetic_off_diagonal,
        dtype=torch.float64,
        device=device,
    )
    hamiltonian = (
        torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    )
    return grid, spacing, potential, hamiltonian


def normalize_wavefunctions(radial_wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize the wavefunction columns along their radial-grid axis."""
    norms = torch.sqrt(
        spacing * torch.sum(torch.abs(radial_wavefunctions) ** 2, dim=-2)
    )
    return radial_wavefunctions / norms.unsqueeze(-2)


def solve_radial_hydrogen(
    num_points: int = 400,
    num_states: int = 3,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> RadialResult:
    """Solve one angular-momentum sector using dense Hermitian diagonalization."""
    _validate_inputs(
        num_points,
        num_states,
        r_max,
        angular_momentum,
        nuclear_charge,
        mass,
        hbar,
    )
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points,
        r_max,
        angular_momentum,
        nuclear_charge,
        mass,
        hbar,
        device,
    )
    energies, wavefunctions = torch.linalg.eigh(hamiltonian)
    selected = normalize_wavefunctions(wavefunctions[:, :num_states], spacing)
    return RadialResult(
        grid,
        spacing,
        potential,
        energies[:num_states],
        selected,
        hamiltonian,
        angular_momentum,
        nuclear_charge,
    )


def solve_angular_momentum_batch(
    angular_momenta: Tensor,
    num_points: int = 300,
    num_states: int = 3,
    r_max: float = 40.0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> BatchedRadialResult:
    """Solve several integer angular-momentum sectors in one tensor batch."""
    if angular_momenta.ndim != 1 or angular_momenta.numel() < 1:
        raise ValueError("angular_momenta must be a nonempty one-dimensional tensor")
    if angular_momenta.is_floating_point() or torch.is_complex(angular_momenta):
        raise ValueError("angular_momenta must have an integer dtype")
    if bool(torch.any(angular_momenta < 0)):
        raise ValueError("angular_momenta must be nonnegative")
    _validate_inputs(num_points, num_states, r_max, 0, nuclear_charge, mass, hbar)
    device = angular_momenta.device
    grid, spacing = make_radial_grid(num_points, r_max, device)
    ell = angular_momenta.to(dtype=torch.float64)[:, None]
    potentials = (
        hbar**2 * ell * (ell + 1.0) / (2.0 * mass * grid[None, :] ** 2)
        - nuclear_charge / grid[None, :]
    )
    kinetic_diagonal = hbar**2 / (mass * spacing**2)
    kinetic_off_diagonal = -(hbar**2) / (2.0 * mass * spacing**2)
    main = torch.full(
        (num_points,), kinetic_diagonal, dtype=torch.float64, device=device
    )
    off = torch.full(
        (num_points - 1,),
        kinetic_off_diagonal,
        dtype=torch.float64,
        device=device,
    )
    kinetic = (
        torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    )
    hamiltonians = kinetic.unsqueeze(0) + torch.diag_embed(potentials)
    energies, wavefunctions = torch.linalg.eigh(hamiltonians)
    selected = normalize_wavefunctions(wavefunctions[:, :, :num_states], spacing)
    return BatchedRadialResult(
        grid,
        spacing,
        angular_momenta,
        potentials,
        energies[:, :num_states],
        selected,
        hamiltonians,
    )


def analytical_energies(
    num_states: int,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return exact Coulomb energies for a fixed angular-momentum sector."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if angular_momentum < 0:
        raise ValueError("angular_momentum must be nonnegative")
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")
    quantum_numbers = torch.arange(
        angular_momentum + 1,
        angular_momentum + num_states + 1,
        dtype=torch.float64,
        device=device,
    )
    return -mass * nuclear_charge**2 / (2.0 * hbar**2 * quantum_numbers**2)


def expectation_radius(result: RadialResult) -> Tensor:
    """Return <r> for every stored reduced radial state."""
    density = torch.abs(result.radial_wavefunctions) ** 2
    return result.spacing * torch.sum(result.grid[:, None] * density, dim=0)


def residual_norms(result: RadialResult) -> Tensor:
    """Return discrete norms of H u_n - E_n u_n."""
    residuals = (
        result.hamiltonian @ result.radial_wavefunctions
        - result.radial_wavefunctions * result.energies[None, :]
    )
    return torch.sqrt(result.spacing * torch.sum(torch.abs(residuals) ** 2, dim=0))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_radial_hydrogen(device=device)
    exact = analytical_energies(len(result.energies), device=device)
    radii = expectation_radius(result)
    print(f"device: {device}")
    print(" n   numerical E       exact E       abs error       <r>")
    for index in range(len(result.energies)):
        principal_n = index + result.angular_momentum + 1
        error = torch.abs(result.energies[index] - exact[index])
        print(
            f"{principal_n:>2}  {result.energies[index].item():>13.8f}  "
            f"{exact[index].item():>12.8f}  {error.item():>12.3e}  "
            f"{radii[index].item():>8.4f}"
        )


if __name__ == "__main__":
    main()
