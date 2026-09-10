"""Self-contained NumPy comparison for two-dimensional FFT wave packets."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class Grid2D:
    x: FloatArray
    y: FloatArray
    kx: FloatArray
    ky: FloatArray
    dx: float
    dy: float

    @property
    def area(self) -> float:
        return self.dx * self.dy


@dataclass(frozen=True)
class Evolution2D:
    grid: Grid2D
    kinetic: FloatArray
    potential: FloatArray
    times: FloatArray
    wavefunctions: ComplexArray


def _positive(*values: float) -> None:
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("scales must be finite and positive")


def make_grid(
    nx: int = 96, ny: int = 80, half_x: float = 10.0, half_y: float = 8.0
) -> Grid2D:
    _positive(half_x, half_y)
    if min(nx, ny) < 4:
        raise ValueError("each axis needs at least four points")
    dx, dy = 2 * half_x / nx, 2 * half_y / ny
    x = -half_x + dx * np.arange(nx, dtype=np.float64)
    y = -half_y + dy * np.arange(ny, dtype=np.float64)
    return Grid2D(
        x,
        y,
        2 * np.pi * np.fft.fftfreq(nx, d=dx),
        2 * np.pi * np.fft.fftfreq(ny, d=dy),
        dx,
        dy,
    )


def kinetic_energy(grid: Grid2D, mass: float = 1.0, hbar: float = 1.0) -> FloatArray:
    _positive(mass, hbar)
    return hbar**2 * (grid.ky[:, None] ** 2 + grid.kx[None, :] ** 2) / (2 * mass)


def stiffness_matrix(
    omega_x: float = 1.0, omega_y: float = 1.3, coupling: float = 0.35
) -> FloatArray:
    _positive(omega_x, omega_y)
    if not isfinite(coupling) or abs(coupling) >= omega_x * omega_y:
        raise ValueError("coupling must give a positive-definite stiffness matrix")
    return np.array([[omega_x**2, coupling], [coupling, omega_y**2]], dtype=np.float64)


def normal_modes(stiffness: FloatArray) -> tuple[FloatArray, FloatArray]:
    if (
        stiffness.shape != (2, 2)
        or np.iscomplexobj(stiffness)
        or not np.isfinite(stiffness).all()
    ):
        raise ValueError("stiffness must be a finite real 2x2 matrix")
    if not np.allclose(stiffness, stiffness.T, atol=1e-12, rtol=0):
        raise ValueError("stiffness must be symmetric")
    eigenvalues, modes = np.linalg.eigh(stiffness)
    if np.any(eigenvalues <= 0):
        raise ValueError("stiffness must be positive definite")
    return np.sqrt(eigenvalues), modes


def coupled_potential(
    grid: Grid2D, stiffness: FloatArray, mass: float = 1.0
) -> FloatArray:
    _positive(mass)
    normal_modes(stiffness)
    x, y = grid.x[None, :], grid.y[:, None]
    return np.asarray(
        0.5
        * mass
        * (
            stiffness[0, 0] * x**2
            + 2 * stiffness[0, 1] * x * y
            + stiffness[1, 1] * y**2
        ),
        dtype=np.float64,
    )


def coherent_packet(
    grid: Grid2D,
    stiffness: FloatArray,
    center: tuple[float, float] = (-2.0, 1.0),
    momentum: tuple[float, float] = (0.4, -0.6),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    _positive(mass, hbar)
    if not all(isfinite(v) for v in (*center, *momentum)):
        raise ValueError("center and momentum must be finite")
    frequencies, modes = normal_modes(stiffness)
    omega = (modes * frequencies[None, :]) @ modes.T
    x, y = grid.x[None, :] - center[0], grid.y[:, None] - center[1]
    quadratic = omega[0, 0] * x**2 + 2 * omega[0, 1] * x * y + omega[1, 1] * y**2
    return np.asarray(
        np.exp(
            -mass * quadratic / (2 * hbar)
            + 1j * (momentum[0] * x + momentum[1] * y) / hbar
        ),
        dtype=np.complex128,
    )


def free_gaussian(
    grid: Grid2D,
    time: float = 0.0,
    center: tuple[float, float] = (-3.0, 1.0),
    sigma: tuple[float, float] = (1.0, 1.4),
    momentum: tuple[float, float] = (0.8, -0.4),
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    _positive(*sigma, mass, hbar)
    if not all(isfinite(v) for v in (time, *center, *momentum)):
        raise ValueError("time, center, and momentum must be finite")
    factors = []
    for coordinate, origin, width, p in zip(
        (grid.x, grid.y), center, sigma, momentum, strict=True
    ):
        spread = 1 + 1j * hbar * time / (2 * mass * width**2)
        envelope = -((coordinate - origin - p * time / mass) ** 2) / (
            4 * width**2 * spread
        )
        phase = 1j * (p * (coordinate - origin) - p**2 * time / (2 * mass)) / hbar
        factors.append(
            (2 * np.pi * width**2) ** (-0.25) / spread**0.5 * np.exp(envelope + phase)
        )
    return np.asarray(factors[1][:, None] * factors[0][None, :], dtype=np.complex128)


def normalize_batch(states: ComplexArray, area: float) -> ComplexArray:
    _positive(area)
    if states.ndim != 3 or min(states.shape) < 1:
        raise ValueError("states must have shape (Ny,Nx,nonempty batch)")
    norms = np.sqrt(area * np.sum(np.abs(states) ** 2, axis=(0, 1)))
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("states must have finite nonzero norms")
    return np.asarray(states / norms[None, None, :], dtype=np.complex128)


def propagate_batch(
    initial_states: ComplexArray,
    kinetic: FloatArray,
    potential: FloatArray,
    area: float,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    _positive(area, hbar)
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    if (
        initial_states.ndim != 3
        or kinetic.shape != initial_states.shape[:2]
        or potential.shape != kinetic.shape
    ):
        raise ValueError("diagonals must match the (Ny,Nx) spatial shape")
    if any(
        np.iscomplexobj(d) or not np.isfinite(d).all() for d in (kinetic, potential)
    ):
        raise ValueError("diagonals must be finite and real")
    state = normalize_batch(initial_states, area)
    half_v = np.exp(-0.5j * time_step * potential / hbar)[:, :, None]
    full_t = np.exp(-1j * time_step * kinetic / hbar)[:, :, None]
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        spectrum = np.fft.fft2(half_v * state, axes=(0, 1), norm="ortho")
        state = np.asarray(
            half_v * np.fft.ifft2(full_t * spectrum, axes=(0, 1), norm="ortho"),
            dtype=np.complex128,
        )
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    return time_step * np.asarray(saved_steps, dtype=np.float64), np.stack(saved)


def solve_coupled(
    nx: int = 96,
    ny: int = 80,
    half_x: float = 10.0,
    half_y: float = 8.0,
    time_step: float = 0.02,
    num_steps: int = 300,
    store_every: int = 10,
) -> Evolution2D:
    grid = make_grid(nx, ny, half_x, half_y)
    stiffness = stiffness_matrix()
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


def main() -> None:
    result = solve_coupled()
    density = np.abs(result.wavefunctions) ** 2
    norms = result.grid.area * density.sum(axis=(1, 2))
    mean_x = result.grid.area * np.sum(density[-1] * result.grid.x[None, :])
    mean_y = result.grid.area * np.sum(density[-1] * result.grid.y[:, None])
    print(f"NumPy final center: ({mean_x:.9f}, {mean_y:.9f})")
    print(f"Maximum norm error: {np.max(np.abs(norms - 1)):.3e}")


if __name__ == "__main__":
    main()
