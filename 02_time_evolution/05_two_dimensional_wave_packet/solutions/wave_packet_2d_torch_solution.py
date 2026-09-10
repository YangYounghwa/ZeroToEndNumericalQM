"""Two-dimensional FFT propagation and exact coupled-oscillator comparisons."""

from dataclasses import dataclass
from math import isfinite

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
    wavefunctions: Tensor  # (saved_times, Ny, Nx), for a single packet


def _positive(*values: float) -> None:
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("scales must be finite and positive")


def make_grid(
    nx: int = 96,
    ny: int = 80,
    half_x: float = 10.0,
    half_y: float = 8.0,
    device: str | torch.device = "cpu",
) -> Grid2D:
    _positive(half_x, half_y)
    if min(nx, ny) < 4:
        raise ValueError("each axis needs at least four points")
    dx, dy = 2 * half_x / nx, 2 * half_y / ny
    x = -half_x + dx * torch.arange(nx, dtype=torch.float64, device=device)
    y = -half_y + dy * torch.arange(ny, dtype=torch.float64, device=device)
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=dx, dtype=torch.float64, device=device)
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=dy, dtype=torch.float64, device=device)
    return Grid2D(x, y, kx, ky, dx, dy)


def kinetic_energy(grid: Grid2D, mass: float = 1.0, hbar: float = 1.0) -> Tensor:
    _positive(mass, hbar)
    return hbar**2 * (grid.ky[:, None] ** 2 + grid.kx[None, :] ** 2) / (2 * mass)


def stiffness_matrix(
    omega_x: float = 1.0,
    omega_y: float = 1.3,
    coupling: float = 0.35,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return K with V=m*r^T*K*r/2. Coupling has units of frequency squared."""
    _positive(omega_x, omega_y)
    if not isfinite(coupling) or abs(coupling) >= omega_x * omega_y:
        raise ValueError("coupling must give a positive-definite stiffness matrix")
    return torch.tensor(
        [[omega_x**2, coupling], [coupling, omega_y**2]],
        dtype=torch.float64,
        device=device,
    )


def normal_modes(stiffness: Tensor) -> tuple[Tensor, Tensor]:
    """K=Q diag(omega^2) Q.T; return frequencies and orthonormal mode columns."""
    if (
        stiffness.shape != (2, 2)
        or torch.is_complex(stiffness)
        or not bool(torch.isfinite(stiffness).all())
    ):
        raise ValueError("stiffness must be a finite real 2x2 matrix")
    if not torch.allclose(stiffness, stiffness.mT, atol=1e-12, rtol=0):
        raise ValueError("stiffness must be symmetric")
    eigenvalues, modes = torch.linalg.eigh(stiffness.to(torch.float64))
    if bool((eigenvalues <= 0).any()):
        raise ValueError("stiffness must be positive definite")
    return torch.sqrt(eigenvalues), modes


def coupled_potential(grid: Grid2D, stiffness: Tensor, mass: float = 1.0) -> Tensor:
    _positive(mass)
    normal_modes(stiffness)
    if stiffness.device != grid.x.device:
        raise ValueError("grid and stiffness must share a device")
    x, y = grid.x[None, :], grid.y[:, None]
    return (
        0.5
        * mass
        * (
            stiffness[0, 0] * x**2
            + 2 * stiffness[0, 1] * x * y
            + stiffness[1, 1] * y**2
        )
    )


def coherent_packet(
    grid: Grid2D,
    stiffness: Tensor,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Displaced coupled ground state, unnormalized; momentum is p, not k."""
    _positive(mass, hbar)
    if not all(isfinite(v) for v in (*center, *momentum)):
        raise ValueError("center and momentum must be finite")
    frequencies, modes = normal_modes(stiffness)
    if stiffness.device != grid.x.device:
        raise ValueError("grid and stiffness must share a device")
    omega = (modes * frequencies[None, :]) @ modes.mT
    x, y = grid.x[None, :] - center[0], grid.y[:, None] - center[1]
    quadratic = omega[0, 0] * x**2 + 2 * omega[0, 1] * x * y + omega[1, 1] * y**2
    return torch.exp(
        -mass * quadratic / (2 * hbar) + 1j * (momentum[0] * x + momentum[1] * y) / hbar
    ).to(torch.complex128)


def coherent_reference(
    times: Tensor,
    stiffness: Tensor,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[Tensor, Tensor, Tensor, float]:
    """Infinite-plane centers, momenta, fixed covariance, and constant energy."""
    _positive(mass, hbar)
    frequencies, modes = normal_modes(stiffness)
    r0 = torch.tensor(center, dtype=torch.float64, device=stiffness.device)
    p0 = torch.tensor(momentum, dtype=torch.float64, device=stiffness.device)
    q0, mode_p0 = modes.mT @ r0, modes.mT @ p0
    angles = times[:, None] * frequencies[None, :]
    q = angles.cos() * q0 + angles.sin() * mode_p0 / (mass * frequencies)
    mode_p = angles.cos() * mode_p0 - mass * frequencies * angles.sin() * q0
    covariance = hbar / (2 * mass) * ((modes / frequencies[None, :]) @ modes.mT)
    energy = float(
        hbar * frequencies.sum() / 2
        + torch.dot(p0, p0) / (2 * mass)
        + mass * torch.dot(r0, stiffness @ r0) / 2
    )
    return q @ modes.mT, mode_p @ modes.mT, covariance, energy


def free_gaussian(
    grid: Grid2D,
    time: float = 0.0,
    center: tuple[float, float] = (-3.0, 1.0),
    sigma: tuple[float, float] = (1.0, 1.4),
    momentum: tuple[float, float] = (0.8, -0.4),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Tensor:
    """Normalized infinite-plane free Gaussian at time t, including its phase."""
    _positive(*sigma, mass, hbar)
    if not all(isfinite(v) for v in (time, *center, *momentum)):
        raise ValueError("time, center, and momentum must be finite")
    factors: list[Tensor] = []
    for coordinate, origin, width, p in zip(
        (grid.x, grid.y), center, sigma, momentum, strict=True
    ):
        spread = 1 + 1j * hbar * time / (2 * mass * width**2)
        envelope = -((coordinate - origin - p * time / mass) ** 2) / (
            4 * width**2 * spread
        )
        phase = 1j * (p * (coordinate - origin) - p**2 * time / (2 * mass)) / hbar
        factors.append(
            (2 * torch.pi * width**2) ** (-0.25)
            / spread**0.5
            * torch.exp(envelope + phase)
        )
    return (factors[1][:, None] * factors[0][None, :]).to(torch.complex128)


def normalize_batch(states: Tensor, area: float) -> Tensor:
    _positive(area)
    if states.ndim != 3 or min(states.shape) < 1:
        raise ValueError("states must have shape (Ny,Nx,nonempty batch)")
    states = states.to(torch.complex128)
    norms = torch.sqrt(area * states.abs().square().sum(dim=(0, 1)))
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("states must have finite nonzero norms")
    return states / norms[None, None, :]


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
    """FFT2 Strang steps; history shape is (saved_times,Ny,Nx,batch)."""
    _positive(area, hbar)
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    if (
        initial_states.ndim != 3
        or kinetic.shape != initial_states.shape[:2]
        or potential.shape != kinetic.shape
    ):
        raise ValueError("diagonals must match the (Ny,Nx) spatial shape")
    for diagonal in (kinetic, potential):
        if torch.is_complex(diagonal) or not bool(torch.isfinite(diagonal).all()):
            raise ValueError("diagonals must be finite and real")
        if diagonal.device != initial_states.device:
            raise ValueError("states and diagonals must share a device")
    state = normalize_batch(initial_states, area)
    half_v = torch.exp(-0.5j * time_step * potential.to(torch.float64) / hbar)[
        :, :, None
    ]
    full_t = torch.exp(-1j * time_step * kinetic.to(torch.float64) / hbar)[:, :, None]
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        spectrum = torch.fft.fft2(half_v * state, dim=(0, 1), norm="ortho")
        state = half_v * torch.fft.ifft2(full_t * spectrum, dim=(0, 1), norm="ortho")
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    times = time_step * torch.tensor(
        saved_steps, dtype=torch.float64, device=state.device
    )
    return times, torch.stack(saved)


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
    """Default m=hbar=1 coupled coherent-state experiment."""
    grid = make_grid(nx, ny, half_x, half_y, device)
    stiffness = stiffness_matrix(device=device)
    potential = coupled_potential(grid, stiffness)
    kinetic = kinetic_energy(grid)
    initial = coherent_packet(grid, stiffness)
    times, history = propagate_batch(
        initial[:, :, None],
        kinetic,
        potential,
        grid.area,
        time_step,
        num_steps,
        store_every,
    )
    return Evolution2D(grid, kinetic, potential, times, history[:, :, :, 0])


def position_statistics(result: Evolution2D) -> tuple[Tensor, Tensor]:
    """Return centers (S,2) and covariance matrices (S,2,2), normalized by each norm."""
    density = result.wavefunctions.abs().square()
    probability = density / density.sum(dim=(1, 2))[:, None, None]
    x, y = result.grid.x[None, None, :], result.grid.y[None, :, None]
    mx, my = (probability * x).sum((1, 2)), (probability * y).sum((1, 2))
    cx, cy = x - mx[:, None, None], y - my[:, None, None]
    xx = (probability * cx**2).sum((1, 2))
    yy = (probability * cy**2).sum((1, 2))
    xy = (probability * cx * cy).sum((1, 2))
    covariance = torch.stack([torch.stack([xx, xy], 1), torch.stack([xy, yy], 1)], 1)
    return torch.stack([mx, my], 1), covariance


def energy_expectations(result: Evolution2D) -> Tensor:
    spectrum: Tensor = torch.fft.fft2(result.wavefunctions, dim=(1, 2), norm="ortho")
    return result.grid.area * (
        (spectrum.abs().square() * result.kinetic).sum((1, 2))
        + (result.wavefunctions.abs().square() * result.potential).sum((1, 2))
    )


def boundary_probability(result: Evolution2D, strip_width: float = 1.5) -> Tensor:
    """Union of four edge strips; count corners once."""
    _positive(strip_width)
    grid = result.grid
    if strip_width >= min(-float(grid.x[0]), -float(grid.y[0])):
        raise ValueError("boundary strips must not fill either axis")
    edge = (grid.y.abs() >= -float(grid.y[0]) - strip_width)[:, None] | (
        grid.x.abs() >= -float(grid.x[0]) - strip_width
    )[None, :]
    return grid.area * result.wavefunctions[:, edge].abs().square().sum(1)


def phase_aligned_error(state: Tensor, reference: Tensor, area: float) -> float:
    overlap = area * torch.vdot(reference.reshape(-1), state.reshape(-1))
    phase = overlap / overlap.abs() if float(overlap.abs()) > 0 else 1.0
    return float(torch.sqrt(area * (state - phase * reference).abs().square().sum()))


def spectral_hamiltonian(kinetic: Tensor, potential: Tensor) -> Tensor:
    """Tiny-grid reference only; columns are C-order flattened position basis states."""
    ny, nx = kinetic.shape
    basis = torch.eye(nx * ny, dtype=torch.complex128, device=kinetic.device).reshape(
        ny, nx, nx * ny
    )
    transformed = torch.fft.fft2(basis, dim=(0, 1), norm="ortho")
    applied: Tensor = torch.fft.ifft2(
        kinetic[:, :, None] * transformed, dim=(0, 1), norm="ortho"
    )
    return applied.reshape(nx * ny, nx * ny) + torch.diag(
        potential.reshape(-1).to(torch.complex128)
    )


def main() -> None:
    result = solve_coupled()
    centers, covariance = position_statistics(result)
    exact, _, exact_covariance, exact_energy = coherent_reference(
        result.times, stiffness_matrix()
    )
    energies = energy_expectations(result)
    norms = result.grid.area * result.wavefunctions.abs().square().sum((1, 2))
    print("2D coupled oscillator: shape (Ny,Nx)=(80,96), dx=0.208333, dy=0.2")
    print(f"Maximum center error: {float((centers - exact).abs().max()):.3e}")
    print(
        f"Maximum covariance error: {float((covariance - exact_covariance).abs().max()):.3e}"
    )
    print(
        f"Maximum energy error to continuum: {float((energies - exact_energy).abs().max()):.3e}"
    )
    print(f"Maximum norm error: {float((norms - 1).abs().max()):.3e}")
    print(
        f"Maximum saved edge probability: {float(boundary_probability(result).max()):.3e}"
    )


if __name__ == "__main__":
    main()
