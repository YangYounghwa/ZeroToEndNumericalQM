"""NumPy FFT comparison and SciPy periodic finite-difference CN reference."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TunnellingResult:
    grid: FloatArray
    spacing: float
    kinetic: FloatArray
    potential: FloatArray
    times: FloatArray
    wavefunctions: ComplexArray


def make_grid(
    num_points: int = 1024, half_extent: float = 64.0
) -> tuple[FloatArray, float, FloatArray]:
    """Periodic grid [-L,L), excluding the duplicate right endpoint."""
    if num_points < 4 or not isfinite(half_extent) or half_extent <= 0:
        raise ValueError("need at least four points and a finite positive extent")
    spacing = 2 * half_extent / num_points
    grid = -half_extent + spacing * np.arange(num_points, dtype=np.float64)
    return grid, spacing, 2 * np.pi * np.fft.fftfreq(num_points, d=spacing)


def gaussian_barrier(
    grid: FloatArray, height: float = 2.5, width: float = 0.8
) -> FloatArray:
    if not isfinite(width) or width <= 0 or not isfinite(height) or height < 0:
        raise ValueError("width must be positive and height nonnegative, both finite")
    return height * np.exp(-0.5 * (grid / width) ** 2)


def gaussian_packet(
    grid: FloatArray,
    center: float = -20.0,
    sigma: float = 3.0,
    wave_number: float = 1.5,
) -> ComplexArray:
    if not all(isfinite(v) for v in (center, sigma, wave_number)) or sigma <= 0:
        raise ValueError("packet parameters must be finite and sigma positive")
    return np.asarray(
        np.exp(
            -((grid - center) ** 2) / (4 * sigma**2)
            + 1j * wave_number * (grid - center)
        ),
        dtype=np.complex128,
    )


def normalize_columns(states: ComplexArray, spacing: float) -> ComplexArray:
    if (
        not isfinite(spacing)
        or spacing <= 0
        or states.ndim != 2
        or min(states.shape) < 1
    ):
        raise ValueError("need positive spacing and (space, nonempty batch) states")
    norms = np.sqrt(spacing * np.sum(np.abs(states) ** 2, axis=0))
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("states must have finite nonzero norms")
    return np.asarray(states / norms[None, :], dtype=np.complex128)


def _check_steps(
    time_step: float, num_steps: int, store_every: int, hbar: float
) -> None:
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time_step must be finite/nonzero and step counts positive")
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")


def propagate_batch(
    initial_states: ComplexArray,
    kinetic: FloatArray,
    potential: FloatArray,
    spacing: float,
    time_step: float = 0.02,
    num_steps: int = 1500,
    store_every: int = 25,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """NumPy version of symmetric splitting; FFT axis 0 is space, axis 1 is batch."""
    _check_steps(time_step, num_steps, store_every, hbar)
    if initial_states.ndim != 2 or kinetic.shape != (initial_states.shape[0],):
        raise ValueError("kinetic must match the space axis of state columns")
    if potential.shape != kinetic.shape:
        raise ValueError("potential and kinetic must have the same shape")
    if any(
        np.iscomplexobj(d) or not np.isfinite(d).all() for d in (kinetic, potential)
    ):
        raise ValueError("kinetic and potential must be finite and real")
    state = normalize_columns(initial_states, spacing)
    half_potential = np.exp(-0.5j * time_step * potential / hbar)
    full_kinetic = np.exp(-1j * time_step * kinetic / hbar)
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        spectrum = np.fft.fft(half_potential[:, None] * state, axis=0, norm="ortho")
        state = np.asarray(
            half_potential[:, None]
            * np.fft.ifft(full_kinetic[:, None] * spectrum, axis=0, norm="ortho"),
            dtype=np.complex128,
        )
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    return time_step * np.asarray(saved_steps, dtype=np.float64), np.stack(saved)


def periodic_fd_hamiltonian(
    potential: FloatArray, spacing: float, mass: float = 1.0, hbar: float = 1.0
) -> csr_matrix:
    """Three-point kinetic stencil with corner links joining the first/last sites."""
    if not all(isfinite(v) and v > 0 for v in (spacing, mass, hbar)):
        raise ValueError("spacing, mass, and hbar must be finite and positive")
    if (
        potential.ndim != 1
        or len(potential) < 4
        or not np.isfinite(potential).all()
        or np.iscomplexobj(potential)
    ):
        raise ValueError(
            "potential must be a finite real vector of length at least four"
        )
    size = len(potential)
    scale = hbar**2 / (2 * mass * spacing**2)
    off = np.full(size - 1, -scale)
    # Five diagonals: the outer two contain one periodic corner coupling each.
    return diags(
        [np.array([-scale]), off, 2 * scale + potential, off, np.array([-scale])],
        offsets=[-(size - 1), -1, 0, 1, size - 1],
        format="csr",
    )


def propagate_cn(
    initial_states: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float = 0.01,
    num_steps: int = 3000,
    store_every: int = 50,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """Reuse sparse LU for the finite-difference comparison; never form an inverse."""
    _check_steps(time_step, num_steps, store_every, hbar)
    state = normalize_columns(initial_states, spacing)
    size = state.shape[0]
    if hamiltonian.shape != (size, size) or not np.isfinite(hamiltonian.data).all():
        raise ValueError("hamiltonian must be finite and match the state size")
    difference = hamiltonian - hamiltonian.conjugate().transpose()
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError("hamiltonian must be Hermitian")
    identity = eye(size, dtype=np.complex128, format="csr")
    coefficient = 0.5j * time_step / hbar
    lu = splu((identity + coefficient * hamiltonian).tocsc())
    right = identity - coefficient * hamiltonian
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        state = np.asarray(lu.solve(right @ state), dtype=np.complex128)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    return time_step * np.asarray(saved_steps, dtype=np.float64), np.stack(saved)


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
) -> TunnellingResult:
    if not all(isfinite(v) and v > 0 for v in (mass, hbar)):
        raise ValueError("mass and hbar must be finite and positive")
    if not -half_extent < center < 0 or wave_number <= 0:
        raise ValueError("launch from inside the left half with positive wave number")
    grid, spacing, wave_numbers = make_grid(num_points, half_extent)
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


def main() -> None:
    result = solve_tunnelling()
    density = np.abs(result.wavefunctions) ** 2
    left, right = result.grid < -4, result.grid > 4
    budget = result.spacing * np.array(
        [
            density[-1, left].sum(),
            density[-1, ~(left | right)].sum(),
            density[-1, right].sum(),
        ]
    )
    print(f"NumPy FFT final left / near / right: {budget.tolist()}")
    print(
        f"Maximum norm error: {np.max(np.abs(result.spacing * density.sum(1) - 1)):.3e}"
    )


if __name__ == "__main__":
    main()
