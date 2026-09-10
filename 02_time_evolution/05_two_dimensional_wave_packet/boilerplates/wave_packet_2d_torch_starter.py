"""Starter: two-dimensional wave packets. Complete a copy in workbench/."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Grid2D:
    """Independent periodic axes; all sampled fields use (Ny,Nx) ordering."""

    x: Tensor
    y: Tensor
    kx: Tensor
    ky: Tensor
    dx: float
    dy: float

    @property
    def area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class Evolution2D:
    grid: Grid2D
    kinetic: Tensor
    potential: Tensor
    times: Tensor
    wavefunctions: Tensor


def make_grid(
    nx: int = 96,
    ny: int = 80,
    half_x: float = 10.0,
    half_y: float = 8.0,
    device: str | torch.device = "cpu",
) -> Grid2D:
    """TODO: Build independent endpoint-excluded periodic axes and unshifted angular wave numbers."""
    raise NotImplementedError


def kinetic_energy(grid: Grid2D, mass: float = 1.0, hbar: float = 1.0) -> Tensor:
    """TODO: Broadcast squared ky and kx, with hbar squared over twice the mass."""
    raise NotImplementedError


def stiffness_matrix(
    omega_x: float = 1.0,
    omega_y: float = 1.3,
    coupling: float = 0.35,
    device: str | torch.device = "cpu",
) -> Tensor:
    """TODO: Construct the symmetric matrix and enforce a stable positive-definite oscillator."""
    raise NotImplementedError


def normal_modes(stiffness: Tensor) -> tuple[Tensor, Tensor]:
    """TODO: Diagonalize the positive symmetric matrix; take square roots of its eigenvalues."""
    raise NotImplementedError


def coupled_potential(grid: Grid2D, stiffness: Tensor, mass: float = 1.0) -> Tensor:
    """TODO: Broadcast x and y to construct m*r.T*K*r/2 with the xy term included."""
    raise NotImplementedError


def coherent_packet(
    grid: Grid2D,
    stiffness: Tensor,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """TODO: Construct the Gaussian quadratic form from the positive matrix square root, then add the momentum phase."""
    raise NotImplementedError


def coherent_reference(
    times: Tensor,
    stiffness: Tensor,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """TODO: Evolve the center and momentum in normal-mode coordinates, rotate back, and compute covariance and energy."""
    raise NotImplementedError


def free_gaussian(
    grid: Grid2D,
    time: float = 0.0,
    center: tuple[float, float] = (-3.0, 1.0),
    sigma: tuple[float, float] = (1.0, 1.4),
    momentum: tuple[float, float] = (0.8, -0.4),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """TODO: Multiply the two analytical Gaussian factors, including spreading and phase."""
    raise NotImplementedError


def normalize_batch(states: Tensor, area: float) -> Tensor:
    """TODO: Sum density over spatial axes only, including dx*dy; reject zero or nonfinite norms."""
    raise NotImplementedError


def propagate_batch(
    initial_states: Tensor,
    kinetic: Tensor,
    potential: Tensor,
    area: float,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """TODO: Apply symmetric FFT2 steps on axes (0,1), keeping the batch axis unchanged and saving requested times."""
    raise NotImplementedError


def solve_coupled(
    nx: int = 96,
    ny: int = 80,
    half_x: float = 10.0,
    half_y: float = 8.0,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
    device: str | torch.device = "cpu",
) -> Evolution2D:
    """TODO: Build the default coupled problem, propagate its initial packet, and package a single-packet result."""
    raise NotImplementedError


def position_statistics(result: Evolution2D) -> tuple[Tensor, Tensor]:
    """TODO: Calculate the centers and all covariance entries, with xy correlation included."""
    raise NotImplementedError


def energy_expectations(result: Evolution2D) -> Tensor:
    """TODO: Combine the kinetic FFT2 sum with the position-space potential sum, using the area element."""
    raise NotImplementedError


def boundary_probability(result: Evolution2D, strip_width: float = 1.5) -> Tensor:
    """TODO: Integrate a Boolean union of four edge strips, counting corners once."""
    raise NotImplementedError


def phase_aligned_error(state: Tensor, reference: Tensor, area: float) -> float:
    """TODO: Flatten for a conjugate overlap, align the global phase, and measure area-weighted state error."""
    raise NotImplementedError


def spectral_hamiltonian(kinetic: Tensor, potential: Tensor) -> Tensor:
    """TODO: Apply the Fourier kinetic operator to reshaped identity columns and add diagonal potential."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise and write long experiment output to Markdown."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
