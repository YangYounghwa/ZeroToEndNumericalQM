"""Starter: FFT tunnelling. Complete a copy in workbench/."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class TunnellingResult:
    """Single-packet snapshots of shape (saved_times, space)."""

    grid: Tensor
    spacing: float
    kinetic: Tensor
    potential: Tensor
    times: Tensor
    wavefunctions: Tensor


def make_grid(
    num_points: int = 1024,
    half_extent: float = 64.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor]:
    """TODO: Exclude the duplicate right endpoint; return unshifted angular wave numbers."""
    raise NotImplementedError


def gaussian_barrier(grid: Tensor, height: float = 2.5, width: float = 0.8) -> Tensor:
    """TODO: Sample the Gaussian potential with a positive width and nonnegative height."""
    raise NotImplementedError


def gaussian_packet(
    grid: Tensor, center: float = -20.0, sigma: float = 3.0, wave_number: float = 1.5
) -> Tensor:
    """TODO: Combine the Gaussian envelope with a complex traveling phase."""
    raise NotImplementedError


def incident_tail_probabilities(
    sigma: float = 3.0,
    wave_number: float = 1.5,
    height: float = 2.5,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[float, float]:
    """TODO: Integrate the two Gaussian energy tails with erfc; also return the negative-momentum weight."""
    raise NotImplementedError


def normalize_columns(states: Tensor, spacing: float) -> Tensor:
    """TODO: Use dx-weighted norms for (space,batch) columns, rejecting zero/nonfinite states."""
    raise NotImplementedError


def propagate_batch(
    initial_states: Tensor,
    kinetic: Tensor,
    potential: Tensor,
    spacing: float,
    time_step: float = 0.02,
    num_steps: int = 1500,
    store_every: int = 25,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """TODO: Apply potential half-step, FFT, kinetic phase, inverse FFT, potential half-step; save requested snapshots."""
    raise NotImplementedError


def solve_tunnelling(
    num_points: int = 1024,
    half_extent: float = 64.0,
    height: float = 2.5,
    width: float = 0.8,
    center: float = -20.0,
    sigma: float = 3.0,
    wave_number: float = 1.5,
    time_step: float = 0.02,
    num_steps: int = 1500,
    store_every: int = 25,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> TunnellingResult:
    """TODO: Prepare the grid, packet, and phases; propagate and package the snapshots."""
    raise NotImplementedError


def region_probabilities(result: TunnellingResult, cut: float = 4.0) -> Tensor:
    """TODO: Integrate three exhaustive spatial masks with dx."""
    raise NotImplementedError


def energy_expectations(result: TunnellingResult) -> Tensor:
    """TODO: Combine kinetic energy from the orthonormal FFT spectrum and potential energy from position density."""
    raise NotImplementedError


def boundary_probability(result: TunnellingResult, strip_width: float = 6.0) -> Tensor:
    """TODO: Integrate strips near both periodic edges at every saved time."""
    raise NotImplementedError


def high_frequency_probability(
    result: TunnellingResult, fraction: float = 0.8
) -> Tensor:
    """TODO: Integrate the outer represented momentum band using Parseval normalization."""
    raise NotImplementedError


def spectral_hamiltonian(kinetic: Tensor, potential: Tensor) -> Tensor:
    """TODO: Apply the Fourier kinetic operator to identity columns, then add diagonal V."""
    raise NotImplementedError


def phase_aligned_error(state: Tensor, reference: Tensor, spacing: float) -> float:
    """TODO: Remove the relative global phase using a conjugate inner product before measuring error."""
    raise NotImplementedError


def main() -> None:
    """TODO: Run the exercise; send long experiment output to a Markdown file."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
