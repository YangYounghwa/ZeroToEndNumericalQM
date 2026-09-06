"""PyTorch time evolution of a free Gaussian wave packet."""

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor


@dataclass(frozen=True)
class EvolutionResult:
    """A wavefunction sampled at every stored time."""

    grid: Tensor
    spacing: float
    times: Tensor
    wavefunctions: Tensor
    hamiltonian: Tensor


def make_grid(
    num_points: int,
    x_min: float = -20.0,
    x_max: float = 20.0,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, float]:
    """Return float64 interior points on the requested device."""
    if num_points < 3 or x_min >= x_max:
        raise ValueError("require num_points >= 3 and x_min < x_max")
    spacing = (x_max - x_min) / (num_points + 1)
    indices = torch.arange(1, num_points + 1, dtype=torch.float64, device=device)
    return x_min + spacing * indices, spacing


def build_free_hamiltonian(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Construct the dense complex128 free-particle Hamiltonian."""
    if num_points < 3 or spacing <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("grid and physical scales must be positive")
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = torch.full((num_points,), 2.0 * scale, dtype=torch.float64, device=device)
    off = torch.full((num_points - 1,), -scale, dtype=torch.float64, device=device)
    real_hamiltonian = (
        torch.diag(main) + torch.diag(off, diagonal=1) + torch.diag(off, diagonal=-1)
    )
    return real_hamiltonian.to(torch.complex128)


def normalize_wavefunctions(wavefunctions: Tensor, spacing: float) -> Tensor:
    """Normalize states along the spatial axis, which is dimension -2."""
    norms = torch.sqrt(spacing * torch.sum(torch.abs(wavefunctions) ** 2, dim=-2))
    if bool(torch.any(norms == 0.0)) or not bool(torch.isfinite(norms).all()):
        raise ValueError("wavefunctions must have finite, nonzero norms")
    return wavefunctions / norms.unsqueeze(-2)


def gaussian_wave_packet(
    grid: Tensor,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
) -> Tensor:
    """Return a complex128 Gaussian packet sampled on grid."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    envelope = torch.exp(-((grid - center) ** 2) / (4.0 * width**2))
    phase = torch.exp(1j * wave_number * (grid - center))
    return (envelope * phase).to(torch.complex128)


def propagate_crank_nicolson_batch(
    initial_states: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Propagate one or more state columns using one reused LU factorization."""
    num_points = hamiltonian.shape[0]
    if hamiltonian.shape != (num_points, num_points):
        raise ValueError("hamiltonian must be square")
    if initial_states.ndim != 2 or initial_states.shape[0] != num_points:
        raise ValueError("initial_states must have shape (num_points, batch)")
    if num_steps < 1 or time_step == 0.0 or spacing <= 0.0 or hbar <= 0.0:
        raise ValueError("step count and scales must be valid and nonzero")
    states = normalize_wavefunctions(initial_states.to(torch.complex128), spacing)
    identity = torch.eye(num_points, dtype=torch.complex128, device=hamiltonian.device)
    coefficient = 0.5j * time_step / hbar
    left = identity + coefficient * hamiltonian
    right = identity - coefficient * hamiltonian
    lu, pivots = torch.linalg.lu_factor(left)
    history = torch.empty(
        (num_steps + 1, num_points, states.shape[1]),
        dtype=torch.complex128,
        device=hamiltonian.device,
    )
    history[0] = states
    for step in range(1, num_steps + 1):
        states = torch.linalg.lu_solve(lu, pivots, right @ states)
        history[step] = states
    times = time_step * torch.arange(
        num_steps + 1, dtype=torch.float64, device=hamiltonian.device
    )
    return times, history


def propagate_crank_nicolson(
    initial_state: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Propagate one state and remove the temporary batch dimension."""
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")
    times, history = propagate_crank_nicolson_batch(
        initial_state[:, None],
        hamiltonian,
        spacing,
        time_step,
        num_steps,
        hbar,
    )
    return times, history[:, :, 0]


def solve_free_packet(
    num_points: int = 240,
    x_min: float = -20.0,
    x_max: float = 20.0,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
    time_step: float = 0.005,
    num_steps: int = 400,
    mass: float = 1.0,
    hbar: float = 1.0,
    device: str | torch.device = "cpu",
) -> EvolutionResult:
    """Build and evolve the default free-packet problem."""
    grid, spacing = make_grid(num_points, x_min, x_max, device)
    hamiltonian = build_free_hamiltonian(num_points, spacing, mass, hbar, device)
    initial = gaussian_wave_packet(grid, center, width, wave_number)
    times, wavefunctions = propagate_crank_nicolson(
        initial, hamiltonian, spacing, time_step, num_steps, hbar
    )
    return EvolutionResult(grid, spacing, times, wavefunctions, hamiltonian)


def analytical_wavefunction(
    grid: Tensor,
    time: float,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Sample the infinite-domain free Gaussian using complex PyTorch tensors."""
    if width <= 0 or mass <= 0 or hbar <= 0:
        raise ValueError("width, mass, and hbar must be positive")
    tau = hbar * time / (2.0 * mass * width**2)
    spreading = torch.tensor(1.0 + 1j * tau, dtype=torch.complex128, device=grid.device)
    displacement = grid - center - hbar * wave_number * time / mass
    amplitude = (2.0 * torch.pi * width**2) ** (-0.25) / torch.sqrt(spreading)
    envelope = torch.exp(-(displacement**2) / (4.0 * width**2 * spreading))
    phase = torch.exp(
        1j * wave_number * (grid - center) - 0.5j * hbar * wave_number**2 * time / mass
    )
    result: Tensor = amplitude * envelope * phase
    return result


def matrix_exponential_state(
    initial_state: Tensor,
    hamiltonian: Tensor,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> Tensor:
    """Compute a small-system reference entirely with PyTorch."""
    if not all(isfinite(value) for value in (spacing, time, hbar)):
        raise ValueError("spacing, time, and hbar must be finite")
    if spacing <= 0 or hbar <= 0:
        raise ValueError("spacing and hbar must be positive")
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")
    if hamiltonian.shape != (initial_state.numel(), initial_state.numel()):
        raise ValueError("hamiltonian must be square and match initial_state")
    if initial_state.device != hamiltonian.device:
        raise ValueError("state and hamiltonian must share a device")
    state = normalize_wavefunctions(
        initial_state.to(torch.complex128)[:, None], spacing
    )[:, 0]
    propagator: Tensor = torch.linalg.matrix_exp((-1j * time / hbar) * hamiltonian)
    result: Tensor = propagator @ state
    return result


def state_l2_error(numerical: Tensor, reference: Tensor, spacing: float) -> Tensor:
    """Return weighted L2 error after removing an irrelevant global phase."""
    overlap = spacing * torch.vdot(reference, numerical)
    aligned = numerical * torch.exp(-1j * torch.angle(overlap))
    return torch.sqrt(spacing * torch.sum(torch.abs(aligned - reference) ** 2))


def probability_norms(result: EvolutionResult) -> Tensor:
    """Return the probability norm at every stored time."""
    return result.spacing * torch.sum(torch.abs(result.wavefunctions) ** 2, dim=1)


def expectation_position(result: EvolutionResult) -> Tensor:
    """Return <x>(t)."""
    density = torch.abs(result.wavefunctions) ** 2
    return result.spacing * torch.sum(density * result.grid[None, :], dim=1)


def position_width(result: EvolutionResult) -> Tensor:
    """Return the position standard deviation at every stored time."""
    density = torch.abs(result.wavefunctions) ** 2
    mean = expectation_position(result)
    mean_square = result.spacing * torch.sum(density * result.grid[None, :] ** 2, dim=1)
    return torch.sqrt(torch.clamp(mean_square - mean**2, min=0.0))


def energy_expectations(result: EvolutionResult) -> Tensor:
    """Return the real energy expectation at every stored time."""
    applied = (result.hamiltonian @ result.wavefunctions.mT).mT
    values = result.spacing * torch.sum(result.wavefunctions.conj() * applied, dim=1)
    return values.real


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = solve_free_packet(device=device)
    norms = probability_norms(result)
    positions = expectation_position(result)
    widths = position_width(result)
    energies = energy_expectations(result)
    print(f"device: {device}")
    print(f"initial norm:       {norms[0].item():.12f}")
    print(f"maximum norm drift: {torch.max(torch.abs(norms - norms[0])).item():.3e}")
    print(f"initial <x>:        {positions[0].item():.6f}")
    print(f"final <x>:          {positions[-1].item():.6f}")
    print(f"initial width:      {widths[0].item():.6f}")
    print(f"final width:        {widths[-1].item():.6f}")
    print(
        "maximum energy drift: "
        f"{torch.max(torch.abs(energies - energies[0])).item():.3e}"
    )


if __name__ == "__main__":
    main()
