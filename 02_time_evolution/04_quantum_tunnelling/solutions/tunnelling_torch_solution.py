"""PyTorch tunnelling with symmetric split-operator Fourier propagation."""

from dataclasses import dataclass
from math import erfc, isfinite, sqrt

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


def _positive(*values: float) -> None:
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("scales must be finite and positive")


def make_grid(
    num_points: int = 1024,
    half_extent: float = 64.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float, Tensor]:
    """Return [-L, L), dx, and unshifted angular wave numbers (not frequencies)."""
    _positive(half_extent)
    if num_points < 4:
        raise ValueError("num_points must be at least 4")
    spacing = 2 * half_extent / num_points
    grid = -half_extent + spacing * torch.arange(
        num_points, dtype=torch.float64, device=device
    )
    wave_numbers = (
        2
        * torch.pi
        * torch.fft.fftfreq(num_points, d=spacing, dtype=torch.float64, device=device)
    )
    return grid, spacing, wave_numbers


def gaussian_barrier(grid: Tensor, height: float = 2.5, width: float = 0.8) -> Tensor:
    """V(x)=height*exp(-x^2/(2*width^2)); width is not a full barrier width."""
    _positive(width)
    if not isfinite(height) or height < 0:
        raise ValueError("height must be finite and nonnegative")
    return height * torch.exp(-0.5 * (grid / width) ** 2)


def gaussian_packet(
    grid: Tensor, center: float = -20.0, sigma: float = 3.0, wave_number: float = 1.5
) -> Tensor:
    """Unnormalized packet; sigma is the position standard deviation."""
    _positive(sigma)
    if not all(isfinite(value) for value in (center, wave_number)):
        raise ValueError("center and wave_number must be finite")
    return torch.exp(
        -((grid - center) ** 2) / (4 * sigma**2) + 1j * wave_number * (grid - center)
    ).to(torch.complex128)


def incident_tail_probabilities(
    sigma: float = 3.0,
    wave_number: float = 1.5,
    height: float = 2.5,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[float, float]:
    """Infinite-line Gaussian probabilities P(k<0), P(hbar^2*k^2/(2m)>height).

    These characterize the incoming free packet, not its spectrum during collision.
    The second value includes both signs of k and bounds above-barrier transmission.
    """
    _positive(sigma, height, mass, hbar)
    if not isfinite(wave_number):
        raise ValueError("wave_number must be finite")
    threshold = sqrt(2 * mass * height) / hbar
    factor = sqrt(2) * sigma
    negative = 0.5 * erfc(factor * wave_number)
    above = 0.5 * (
        erfc(factor * (threshold - wave_number))
        + erfc(factor * (threshold + wave_number))
    )
    return negative, above


def normalize_columns(states: Tensor, spacing: float) -> Tensor:
    """Normalize (space, batch) columns once; propagation never renormalizes."""
    _positive(spacing)
    if states.ndim != 2 or min(states.shape) < 1:
        raise ValueError("states must have shape (space, nonempty batch)")
    states = states.to(torch.complex128)
    norms = torch.sqrt(spacing * torch.sum(states.abs() ** 2, dim=0))
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("states must have finite nonzero norms")
    return states / norms[None, :]


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
    """Strang splitting with history (saved_times, space, batch); FFT along space.

    kinetic contains hbar^2*k^2/(2m) in unshifted FFT order. Always save the initial
    and final states. Negative time steps allow a reversibility check.
    """
    _positive(spacing, hbar)
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time_step must be finite/nonzero and step counts positive")
    if initial_states.ndim != 2 or kinetic.shape != (initial_states.shape[0],):
        raise ValueError("kinetic must match the space axis of state columns")
    if potential.shape != kinetic.shape:
        raise ValueError("potential and kinetic must be vectors of the same size")
    for diagonal in (kinetic, potential):
        if diagonal.device != initial_states.device:
            raise ValueError("states, kinetic, and potential must share a device")
        if torch.is_complex(diagonal) or not bool(torch.isfinite(diagonal).all()):
            raise ValueError("kinetic and potential must be finite and real")
    state = normalize_columns(initial_states, spacing)
    half_potential = torch.exp(-0.5j * time_step * potential.to(torch.float64) / hbar)
    full_kinetic = torch.exp(-1j * time_step * kinetic.to(torch.float64) / hbar)
    saved = [state]
    saved_steps = [0]
    for step in range(1, num_steps + 1):
        state = half_potential[:, None] * state
        spectrum = torch.fft.fft(state, dim=0, norm="ortho")
        state = torch.fft.ifft(full_kinetic[:, None] * spectrum, dim=0, norm="ortho")
        state = half_potential[:, None] * state
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    times = time_step * torch.tensor(
        saved_steps, dtype=torch.float64, device=state.device
    )
    return times, torch.stack(saved)


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
    """Launch a packet; use diagnostics to decide if the domain/time is adequate."""
    _positive(mass, hbar)
    if not -half_extent < center < 0 or wave_number <= 0:
        raise ValueError("launch from inside the left half with positive wave number")
    grid, spacing, wave_numbers = make_grid(num_points, half_extent, device)
    kinetic = hbar**2 * wave_numbers**2 / (2 * mass)
    potential = gaussian_barrier(grid, height, width)
    initial = gaussian_packet(grid, center, sigma, wave_number)
    times, history = propagate_batch(
        initial[:, None],
        kinetic,
        potential,
        spacing,
        time_step,
        num_steps,
        store_every,
        hbar,
    )
    return TunnellingResult(grid, spacing, kinetic, potential, times, history[:, :, 0])


def region_probabilities(result: TunnellingResult, cut: float = 4.0) -> Tensor:
    """Return (saved_times, 3) probabilities for x<-cut, |x|<=cut, x>cut."""
    _positive(cut)
    if cut >= float(result.grid[-1]):
        raise ValueError("region cut must fit inside the domain")
    left, right = result.grid < -cut, result.grid > cut
    density = result.wavefunctions.abs() ** 2
    return result.spacing * torch.stack(
        [
            density[:, left].sum(1),
            density[:, ~(left | right)].sum(1),
            density[:, right].sum(1),
        ],
        dim=1,
    )


def energy_expectations(result: TunnellingResult) -> Tensor:
    """Expectation of the spectral Hamiltonian; split evolution need not conserve it."""
    spectrum: Tensor = torch.fft.fft(result.wavefunctions, dim=1, norm="ortho")
    return result.spacing * (
        (spectrum.abs() ** 2 * result.kinetic).sum(1)
        + (result.wavefunctions.abs() ** 2 * result.potential).sum(1)
    )


def boundary_probability(result: TunnellingResult, strip_width: float = 6.0) -> Tensor:
    """Monitor both periodic edges; small final mass alone cannot exclude wraparound."""
    _positive(strip_width)
    half_extent = -float(result.grid[0])
    if strip_width >= half_extent:
        raise ValueError("boundary strips must not overlap")
    mask = result.grid.abs() >= half_extent - strip_width
    return result.spacing * (result.wavefunctions[:, mask].abs() ** 2).sum(1)


def high_frequency_probability(
    result: TunnellingResult, fraction: float = 0.8
) -> Tensor:
    """Probability in the outer fraction-to-1 band of represented |k| (alias check)."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    spectrum: Tensor = torch.fft.fft(result.wavefunctions, dim=1, norm="ortho")
    mask = result.kinetic >= fraction**2 * result.kinetic.max()
    return result.spacing * (spectrum[:, mask].abs() ** 2).sum(1)


def spectral_hamiltonian(kinetic: Tensor, potential: Tensor) -> Tensor:
    """Small-system reference only: F^-1 diag(kinetic) F + diag(potential)."""
    identity = torch.eye(len(kinetic), dtype=torch.complex128, device=kinetic.device)
    transformed = torch.fft.fft(identity, dim=0, norm="ortho")
    applied: Tensor = torch.fft.ifft(
        kinetic[:, None] * transformed, dim=0, norm="ortho"
    )
    return applied + torch.diag(potential.to(torch.complex128))


def phase_aligned_error(state: Tensor, reference: Tensor, spacing: float) -> float:
    """Weighted state error after removing an unobservable global phase."""
    overlap = spacing * torch.vdot(reference, state)
    phase = overlap / overlap.abs() if float(overlap.abs()) > 0 else 1.0
    return float(torch.sqrt(spacing * (state - phase * reference).abs().square().sum()))


def main() -> None:
    result = solve_tunnelling()
    budget = region_probabilities(result)
    negative, above = incident_tail_probabilities()
    energies = energy_expectations(result)
    print("Gaussian barrier: V0=2.5, width=0.8; packet k0=1.5, sigma=3; m=hbar=1")
    print(f"Incoming P(k<0)={negative:.3e}; P(E>V0)={above:.3e}")
    print(f"Final left / near / right: {budget[-1].tolist()}")
    print(f"Maximum norm error: {float((budget.sum(1) - 1).abs().max()):.3e}")
    print(f"Maximum energy drift: {float((energies - energies[0]).abs().max()):.3e}")
    print(
        f"Maximum saved edge probability: {float(boundary_probability(result).max()):.3e}"
    )
    print(
        f"Maximum high-k probability: {float(high_frequency_probability(result).max()):.3e}"
    )
    print(
        "Interpret right probability as transmission only after separation, before wraparound."
    )


if __name__ == "__main__":
    main()
