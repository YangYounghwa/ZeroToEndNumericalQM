"""PyTorch wave-packet scattering from a rectangular potential barrier."""

from dataclasses import dataclass
from math import isfinite, log, pi, sqrt

import torch
from torch import Tensor


@dataclass(frozen=True)
class ScatteringResult:
    """Snapshots of one packet; wavefunctions have shape (time, space)."""

    grid: Tensor
    spacing: float
    potential: Tensor
    hamiltonian: Tensor
    times: Tensor
    wavefunctions: Tensor
    barrier_width: float


@dataclass(frozen=True)
class RegionProbabilities:
    """An exhaustive partition, without assuming scattering has finished."""

    left: Tensor
    near: Tensor
    right: Tensor


def _positive_scales(*values: float) -> None:
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("scales must be finite and positive")


def make_grid(
    num_points: int = 439,
    half_extent: float = 40.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return interior points of [-half_extent, half_extent], omitting zero walls."""
    _positive_scales(half_extent)
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    spacing = 2 * half_extent / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return -half_extent + spacing * indices, spacing


def rectangular_barrier(
    grid: Tensor, height: float = 2.5, width: float = 2.0
) -> Tensor:
    """Sample V=height for |x|<width/2 and V=0 elsewhere."""
    _positive_scales(width)
    if not isfinite(height) or height < 0:
        raise ValueError("height must be finite and nonnegative")
    if grid.ndim != 1 or torch.is_complex(grid) or not bool(torch.isfinite(grid).all()):
        raise ValueError("grid must be a finite real vector")
    return torch.where(
        torch.abs(grid) < width / 2,
        torch.full_like(grid, height, dtype=torch.float64),
        torch.zeros_like(grid, dtype=torch.float64),
    )


def build_hamiltonian(
    num_points: int = 439,
    half_extent: float = 40.0,
    height: float = 2.5,
    width: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor, Tensor]:
    """Build a real symmetric dense finite-difference Hamiltonian."""
    _positive_scales(mass, hbar)
    if width >= 2 * half_extent:
        raise ValueError("barrier must fit strictly inside the domain")
    grid, spacing = make_grid(num_points, half_extent, device)
    potential = rectangular_barrier(grid, height, width)
    scale = hbar**2 / (2 * mass * spacing**2)
    off = torch.full((num_points - 1,), -scale, dtype=torch.float64, device=device)
    hamiltonian = (
        torch.diag(2 * scale + potential)
        + torch.diag(off, diagonal=1)
        + torch.diag(off, diagonal=-1)
    )
    return grid, spacing, potential, hamiltonian


def gaussian_packet(
    grid: Tensor, center: float = -12.0, sigma: float = 2.0, wave_number: float = 2.0
) -> Tensor:
    """Return an unnormalized packet; sigma is its position standard deviation."""
    _positive_scales(sigma)
    if not all(isfinite(value) for value in (center, wave_number)):
        raise ValueError("center and wave_number must be finite")
    return torch.exp(
        -((grid - center) ** 2) / (4 * sigma**2) + 1j * wave_number * (grid - center)
    ).to(torch.complex128)


def normalize_columns(states: Tensor, spacing: float) -> Tensor:
    """Normalize a (space, batch) array under the dx-weighted inner product."""
    _positive_scales(spacing)
    if states.ndim != 2 or min(states.shape) < 1:
        raise ValueError("states must have shape (space, nonempty batch)")
    states = states.to(torch.complex128)
    norms = torch.sqrt(spacing * torch.sum(torch.abs(states) ** 2, dim=0))
    if not bool(torch.isfinite(norms).all()) or bool(torch.any(norms == 0)):
        raise ValueError("states must have finite, nonzero norms")
    return states / norms[None, :]


def propagate_batch(
    initial_states: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time_step: float = 0.02,
    num_steps: int = 700,
    store_every: int = 10,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Apply CN to state columns, factoring the shared left matrix once.

    Return times and history of shape (saved_times, space, batch). Always save
    the initial and final states, including when store_every does not divide steps.
    """
    _positive_scales(spacing, hbar)
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time_step must be finite/nonzero and step counts positive")
    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be square")
    if initial_states.ndim != 2 or initial_states.shape[0] != hamiltonian.shape[0]:
        raise ValueError("initial_states must have shape (space, batch)")
    if initial_states.device != hamiltonian.device:
        raise ValueError("states and hamiltonian must share a device")
    if not bool(torch.isfinite(hamiltonian).all()) or not torch.allclose(
        hamiltonian, hamiltonian.mH, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("hamiltonian must be finite and Hermitian")
    state = normalize_columns(initial_states, spacing)
    operator = hamiltonian.to(torch.complex128)
    identity = torch.eye(
        operator.shape[0], dtype=torch.complex128, device=operator.device
    )
    coefficient = 0.5j * time_step / hbar
    lu, pivots = torch.linalg.lu_factor(identity + coefficient * operator)
    right = identity - coefficient * operator
    saved = [state]
    saved_steps = [0]
    for step in range(1, num_steps + 1):
        state = torch.linalg.lu_solve(lu, pivots, right @ state)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    times = time_step * torch.tensor(
        saved_steps, dtype=torch.float64, device=operator.device
    )
    return times, torch.stack(saved)


def solve_scattering(
    num_points: int = 439,
    half_extent: float = 40.0,
    height: float = 2.5,
    width: float = 2.0,
    center: float = -12.0,
    sigma: float = 2.0,
    wave_number: float = 2.0,
    time_step: float = 0.02,
    num_steps: int = 700,
    store_every: int = 10,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> ScatteringResult:
    """Launch a packet from the left; diagnostics determine usable measurement times."""
    if not -half_extent < center < -width / 2 or wave_number <= 0:
        raise ValueError("packet center must be left of the barrier with positive k")
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, half_extent, height, width, mass, hbar, device
    )
    initial = gaussian_packet(grid, center, sigma, wave_number)
    times, history = propagate_batch(
        initial[:, None], hamiltonian, spacing, time_step, num_steps, store_every, hbar
    )
    return ScatteringResult(
        grid, spacing, potential, hamiltonian, times, history[:, :, 0], width
    )


def region_probabilities(
    result: ScatteringResult, padding: float = 1.0
) -> RegionProbabilities:
    """Partition all probability into left / barrier-neighborhood / right."""
    if not isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    cut = result.barrier_width / 2 + padding
    if cut >= float(result.grid[-1]):
        raise ValueError("measurement regions must fit inside the grid")
    left = result.grid < -cut
    right = result.grid > cut
    near = ~(left | right)
    density = torch.abs(result.wavefunctions) ** 2
    return RegionProbabilities(
        result.spacing * density[:, left].sum(dim=1),
        result.spacing * density[:, near].sum(dim=1),
        result.spacing * density[:, right].sum(dim=1),
    )


def boundary_probability(result: ScatteringResult, strip_width: float = 3.0) -> Tensor:
    """Probability near either wall at every saved time, not removed probability."""
    _positive_scales(strip_width)
    half_extent = float(result.grid[-1]) + result.spacing
    if strip_width >= half_extent:
        raise ValueError("edge strips must not overlap")
    edge = torch.abs(result.grid) > half_extent - strip_width
    return result.spacing * torch.sum(
        torch.abs(result.wavefunctions[:, edge]) ** 2, dim=1
    )


def energy_expectations(result: ScatteringResult) -> Tensor:
    applied = (result.hamiltonian.to(torch.complex128) @ result.wavefunctions.mT).mT
    return (
        result.spacing * torch.sum(result.wavefunctions.conj() * applied, dim=1)
    ).real


def link_current(
    state: Tensor, spacing: float, mass: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """Current across interior links for the nearest-neighbor Hamiltonian."""
    _positive_scales(spacing, mass, hbar)
    if state.ndim != 1 or state.numel() < 2:
        raise ValueError("state must be a vector with at least two points")
    return hbar / (mass * spacing) * (state[:-1].conj() * state[1:]).imag


def plane_wave_transmission(
    energies: Tensor,
    height: float = 2.5,
    width: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Continuum rectangular-barrier T(E), including the E=height limit."""
    _positive_scales(mass, hbar)
    if not all(isfinite(value) and value >= 0 for value in (height, width)):
        raise ValueError("height and width must be finite and nonnegative")
    if (
        torch.is_complex(energies)
        or not bool(torch.isfinite(energies).all())
        or bool(torch.any(energies < 0))
    ):
        raise ValueError("energies must be finite, real, and nonnegative")
    energy = energies.to(torch.float64)
    if height == 0 or width == 0:
        return torch.ones_like(energy)
    transmission = torch.zeros_like(energy)
    above = energy >= height
    phase = width * torch.sqrt(2 * mass * (energy[above] - height)) / hbar
    factor = mass * height**2 * width**2 / (2 * hbar**2 * energy[above])
    transmission[above] = 1 / (1 + factor * torch.sinc(phase / pi) ** 2)
    below = (energy > 0) & (energy < height)
    gap = height - energy[below]
    decay = width * torch.sqrt(2 * mass * gap) / hbar
    # log(sinh(decay)) avoids overflowing for thick barriers.
    log_sinh = decay + torch.log(-torch.expm1(-2 * decay)) - log(2)
    log_factor = (
        2 * log(height)
        - log(4)
        - torch.log(energy[below])
        - torch.log(gap)
        + 2 * log_sinh
    )
    transmission[below] = torch.sigmoid(-log_factor)
    return transmission


def packet_transmission_reference(
    wave_number: float = 2.0,
    sigma: float = 2.0,
    height: float = 2.5,
    width: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
    num_samples: int = 4001,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Integrate T(k) against the positive-k part of the normalized Gaussian spectrum.

    This is an asymptotic infinite-domain reference. Negative-k weight is not
    renormalized away; choose k0*sigma large for a clean incident packet.
    """
    _positive_scales(wave_number, sigma, mass, hbar)
    if num_samples < 3:
        raise ValueError("num_samples must be at least 3")
    k = torch.linspace(
        0,
        wave_number + 8 / (2 * sigma),
        num_samples,
        dtype=torch.float64,
        device=device,
    )
    density = sqrt(2 / pi) * sigma * torch.exp(-2 * sigma**2 * (k - wave_number) ** 2)
    transmission = plane_wave_transmission(
        hbar**2 * k**2 / (2 * mass), height, width, mass, hbar
    )
    return torch.trapezoid(density * transmission, k)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_scattering(device=device)
    regions = region_probabilities(result)
    norms = regions.left + regions.near + regions.right
    energy = energy_expectations(result)
    print(f"PyTorch barrier scattering on {device}; dx={result.spacing:.6f}")
    print("time       left        near        right")
    for index in (0, len(result.times) // 2, len(result.times) - 1):
        print(
            f"{result.times[index].item():>5.1f}  {regions.left[index].item():>10.6f}  {regions.near[index].item():>10.6f}  {regions.right[index].item():>10.6f}"
        )
    print(f"maximum norm error: {torch.max(torch.abs(norms - 1)).item():.3e}")
    print(
        f"maximum energy drift: {torch.max(torch.abs(energy - energy[0])).item():.3e}"
    )
    print(
        f"maximum saved edge probability: {boundary_probability(result).max().item():.3e}"
    )
    print(
        f"packet-averaged continuum T: {packet_transmission_reference(device=device).item():.8f}"
    )
    print("Interpret left/right as R/T only in the validated post-collision window.")


if __name__ == "__main__":
    main()
