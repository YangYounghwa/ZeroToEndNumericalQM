"""NumPy/SciPy sparse comparison for rectangular-barrier packet scattering."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import expm_multiply, splu

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ScatteringResult:
    grid: FloatArray
    spacing: float
    potential: FloatArray
    hamiltonian: csr_matrix
    times: FloatArray
    wavefunctions: ComplexArray
    barrier_width: float


@dataclass(frozen=True)
class RegionProbabilities:
    left: FloatArray
    near: FloatArray
    right: FloatArray


def _positive_scales(*values: float) -> None:
    if not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("scales must be finite and positive")


def make_grid(
    num_points: int = 439, half_extent: float = 40.0
) -> tuple[FloatArray, float]:
    """Use the same interior Dirichlet grid as the PyTorch calculation."""
    _positive_scales(half_extent)
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    spacing = 2 * half_extent / (num_points + 1)
    return -half_extent + spacing * np.arange(
        1, num_points + 1, dtype=np.float64
    ), spacing


def rectangular_barrier(
    grid: FloatArray, height: float = 2.5, width: float = 2.0
) -> FloatArray:
    _positive_scales(width)
    if not isfinite(height) or height < 0:
        raise ValueError("height must be finite and nonnegative")
    if grid.ndim != 1 or np.iscomplexobj(grid) or not np.isfinite(grid).all():
        raise ValueError("grid must be a finite real vector")
    return np.where(np.abs(grid) < width / 2, height, 0.0)


def build_hamiltonian(
    num_points: int = 439,
    half_extent: float = 40.0,
    height: float = 2.5,
    width: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, csr_matrix]:
    """Build only three diagonals; no dense intermediate Hamiltonian."""
    _positive_scales(mass, hbar)
    if width >= 2 * half_extent:
        raise ValueError("barrier must fit strictly inside the domain")
    grid, spacing = make_grid(num_points, half_extent)
    potential = rectangular_barrier(grid, height, width)
    scale = hbar**2 / (2 * mass * spacing**2)
    off = np.full(num_points - 1, -scale, dtype=np.float64)
    # scipy-stubs cannot express differently sized tridiagonal arrays.
    hamiltonian = diags(  # type: ignore[call-overload]
        [off, 2 * scale + potential, off], offsets=[-1, 0, 1], format="csr"
    )
    return grid, spacing, potential, hamiltonian


def gaussian_packet(
    grid: FloatArray,
    center: float = -12.0,
    sigma: float = 2.0,
    wave_number: float = 2.0,
) -> ComplexArray:
    _positive_scales(sigma)
    if not all(isfinite(value) for value in (center, wave_number)):
        raise ValueError("center and wave_number must be finite")
    return np.asarray(
        np.exp(
            -((grid - center) ** 2) / (4 * sigma**2)
            + 1j * wave_number * (grid - center)
        ),
        dtype=np.complex128,
    )


def normalize_state(state: ComplexArray, spacing: float) -> ComplexArray:
    _positive_scales(spacing)
    if state.ndim != 1 or state.size < 1:
        raise ValueError("state must be a nonempty vector")
    norm = float(np.sqrt(spacing * np.sum(np.abs(state) ** 2)))
    if not isfinite(norm) or norm == 0:
        raise ValueError("state must have a finite, nonzero norm")
    return state.astype(np.complex128) / norm


def propagate(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float = 0.02,
    num_steps: int = 700,
    store_every: int = 10,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """Use sparse CN with one LU factorization and selected snapshots."""
    _positive_scales(spacing, hbar)
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time_step must be finite/nonzero and step counts positive")
    if initial_state.ndim != 1 or hamiltonian.shape != (
        initial_state.size,
        initial_state.size,
    ):
        raise ValueError("hamiltonian must be square and match the state")
    difference = hamiltonian - hamiltonian.conj().T
    if not np.isfinite(hamiltonian.data).all() or (
        difference.nnz and np.max(np.abs(difference.data)) > 1e-12
    ):
        raise ValueError("hamiltonian must be finite and Hermitian")
    state = normalize_state(initial_state, spacing)
    identity = eye(state.size, format="csc", dtype=np.complex128)
    coefficient = 0.5j * time_step / hbar
    left = (identity + coefficient * hamiltonian).tocsc()
    right = identity - coefficient * hamiltonian
    factorization = splu(left)
    saved = [state]
    saved_steps = [0]
    for step in range(1, num_steps + 1):
        state = factorization.solve(right @ state)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    return time_step * np.asarray(saved_steps, dtype=np.float64), np.stack(saved)


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
) -> ScatteringResult:
    if not -half_extent < center < -width / 2 or wave_number <= 0:
        raise ValueError("packet center must be left of the barrier with positive k")
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points, half_extent, height, width, mass, hbar
    )
    initial = gaussian_packet(grid, center, sigma, wave_number)
    times, history = propagate(
        initial, hamiltonian, spacing, time_step, num_steps, store_every, hbar
    )
    return ScatteringResult(
        grid, spacing, potential, hamiltonian, times, history, width
    )


def region_probabilities(
    result: ScatteringResult, padding: float = 1.0
) -> RegionProbabilities:
    if not isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    cut = result.barrier_width / 2 + padding
    if cut >= result.grid[-1]:
        raise ValueError("measurement regions must fit inside the grid")
    left = result.grid < -cut
    right = result.grid > cut
    near = ~(left | right)
    density = np.abs(result.wavefunctions) ** 2
    return RegionProbabilities(
        result.spacing * density[:, left].sum(axis=1),
        result.spacing * density[:, near].sum(axis=1),
        result.spacing * density[:, right].sum(axis=1),
    )


def boundary_probability(
    result: ScatteringResult, strip_width: float = 3.0
) -> FloatArray:
    _positive_scales(strip_width)
    half_extent = float(result.grid[-1]) + result.spacing
    if strip_width >= half_extent:
        raise ValueError("edge strips must not overlap")
    edge = np.abs(result.grid) > half_extent - strip_width
    return result.spacing * np.sum(np.abs(result.wavefunctions[:, edge]) ** 2, axis=1)


def energy_expectations(result: ScatteringResult) -> FloatArray:
    applied = (result.hamiltonian @ result.wavefunctions.T).T
    return np.asarray(
        (result.spacing * np.sum(result.wavefunctions.conj() * applied, axis=1)).real,
        dtype=np.float64,
    )


def exponential_state(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> ComplexArray:
    """Sparse exponential-action reference for fixed-grid time convergence."""
    _positive_scales(hbar)
    if not isfinite(time) or hamiltonian.shape != (
        initial_state.size,
        initial_state.size,
    ):
        raise ValueError("time must be finite and Hamiltonian must match state")
    state = normalize_state(initial_state, spacing)
    generator = (-1j * time / hbar) * hamiltonian
    return np.asarray(
        expm_multiply(generator, state, traceA=generator.diagonal().sum()),
        dtype=np.complex128,
    )


def main() -> None:
    result = solve_scattering()
    regions = region_probabilities(result)
    norms = regions.left + regions.near + regions.right
    energies = energy_expectations(result)
    print(f"NumPy/SciPy barrier scattering; dx={result.spacing:.6f}")
    print(f"final time: {result.times[-1]:.2f}")
    print(
        f"left={regions.left[-1]:.8f}, near={regions.near[-1]:.3e}, right={regions.right[-1]:.8f}"
    )
    print(f"maximum norm error: {np.max(np.abs(norms - 1)):.3e}")
    print(f"maximum energy drift: {np.max(np.abs(energies - energies[0])):.3e}")
    print(f"maximum saved edge probability: {boundary_probability(result).max():.3e}")


if __name__ == "__main__":
    main()
