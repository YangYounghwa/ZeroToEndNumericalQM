"""NumPy/SciPy time evolution of a free Gaussian wave packet."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import splu

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EvolutionResult:
    """A wavefunction sampled at every stored time."""

    grid: FloatArray
    spacing: float
    times: FloatArray
    wavefunctions: ComplexArray
    hamiltonian: csr_matrix


def _validate_grid(num_points: int, x_min: float, x_max: float) -> None:
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not np.isfinite([x_min, x_max]).all() or x_min >= x_max:
        raise ValueError("x_min and x_max must be finite with x_min < x_max")


def make_grid(
    num_points: int,
    x_min: float = -20.0,
    x_max: float = 20.0,
) -> tuple[FloatArray, float]:
    """Return interior points with zero Dirichlet endpoint values omitted."""
    _validate_grid(num_points, x_min, x_max)
    spacing = (x_max - x_min) / (num_points + 1)
    indices = np.arange(1, num_points + 1, dtype=np.float64)
    return x_min + spacing * indices, spacing


def build_free_hamiltonian(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> csr_matrix:
    """Construct the sparse finite-difference free-particle Hamiltonian."""
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not np.isfinite([spacing, mass, hbar]).all():
        raise ValueError("spacing, mass, and hbar must be finite")
    if spacing <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("spacing, mass, and hbar must be positive")
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = np.full(num_points, 2.0 * scale, dtype=np.float64)
    off = np.full(num_points - 1, -scale, dtype=np.float64)
    # scipy-stubs cannot express differently sized tridiagonal arrays.
    return diags(  # type: ignore[call-overload,no-any-return]
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="csr",
        dtype=np.float64,
    )


def normalize_wavefunction(wavefunction: ComplexArray, spacing: float) -> ComplexArray:
    """Normalize one sampled wavefunction under the discrete spatial integral."""
    norm = float(np.sqrt(spacing * np.sum(np.abs(wavefunction) ** 2)))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("wavefunction must have a finite, nonzero norm")
    return wavefunction / norm


def gaussian_wave_packet(
    grid: FloatArray,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
) -> ComplexArray:
    """Return a normalized Gaussian with position standard deviation width."""
    if not np.isfinite([center, width, wave_number]).all() or width <= 0.0:
        raise ValueError("center and wave_number must be finite and width positive")
    envelope = np.exp(-((grid - center) ** 2) / (4.0 * width**2))
    phase = np.exp(1j * wave_number * (grid - center))
    return (envelope * phase).astype(np.complex128)


def propagate_crank_nicolson(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """Propagate with the unitary Crank-Nicolson update."""
    num_points = hamiltonian.shape[0]
    if hamiltonian.shape != (num_points, num_points):
        raise ValueError("hamiltonian must be square")
    if initial_state.shape != (num_points,):
        raise ValueError("initial_state shape must match hamiltonian")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if time_step == 0.0 or spacing <= 0.0 or hbar <= 0.0:
        raise ValueError("time_step must be nonzero; spacing and hbar must be positive")
    state = normalize_wavefunction(initial_state.astype(np.complex128), spacing)
    identity = eye(num_points, format="csc", dtype=np.complex128)
    coefficient = 0.5j * time_step / hbar
    left: csc_matrix = identity + coefficient * hamiltonian
    right: csc_matrix = identity - coefficient * hamiltonian
    factorization = splu(left)
    wavefunctions = np.empty((num_steps + 1, num_points), dtype=np.complex128)
    wavefunctions[0] = state
    for step in range(1, num_steps + 1):
        state = factorization.solve(right @ state)
        wavefunctions[step] = state
    times = time_step * np.arange(num_steps + 1, dtype=np.float64)
    return times, wavefunctions


def solve_free_packet(
    num_points: int = 400,
    x_min: float = -20.0,
    x_max: float = 20.0,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
    time_step: float = 0.005,
    num_steps: int = 400,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> EvolutionResult:
    """Build and evolve the default free-packet problem."""
    grid, spacing = make_grid(num_points, x_min, x_max)
    hamiltonian = build_free_hamiltonian(num_points, spacing, mass, hbar)
    initial = gaussian_wave_packet(grid, center, width, wave_number)
    times, wavefunctions = propagate_crank_nicolson(
        initial, hamiltonian, spacing, time_step, num_steps, hbar
    )
    return EvolutionResult(grid, spacing, times, wavefunctions, hamiltonian)


def matrix_exponential_state(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> ComplexArray:
    """Return a small-system dense matrix-exponential reference state."""
    if time < 0.0 or hbar <= 0.0:
        raise ValueError("time must be nonnegative and hbar positive")
    state = normalize_wavefunction(initial_state.astype(np.complex128), spacing)
    propagator = expm((-1j * time / hbar) * hamiltonian.toarray())
    return np.asarray(propagator @ state, dtype=np.complex128)


def analytical_wavefunction(
    grid: FloatArray,
    time: float,
    center: float = -6.0,
    width: float = 0.8,
    wave_number: float = 2.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    """Return the infinite-domain analytical free Gaussian wavefunction."""
    if time < 0.0 or width <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("time must be nonnegative and other scales positive")
    tau = hbar * time / (2.0 * mass * width**2)
    velocity = hbar * wave_number / mass
    displacement = grid - center - velocity * time
    amplitude = (2.0 * np.pi * width**2) ** (-0.25) / np.sqrt(1.0 + 1j * tau)
    envelope = np.exp(-(displacement**2) / (4.0 * width**2 * (1.0 + 1j * tau)))
    phase = np.exp(
        1j * wave_number * (grid - center) - 0.5j * hbar * wave_number**2 * time / mass
    )
    return np.asarray(amplitude * envelope * phase, dtype=np.complex128)


def probability_norms(result: EvolutionResult) -> FloatArray:
    """Return the probability norm at every stored time."""
    return result.spacing * np.sum(np.abs(result.wavefunctions) ** 2, axis=1)


def expectation_position(result: EvolutionResult) -> FloatArray:
    """Return <x>(t)."""
    density = np.abs(result.wavefunctions) ** 2
    return result.spacing * np.sum(density * result.grid[np.newaxis, :], axis=1)


def position_width(result: EvolutionResult) -> FloatArray:
    """Return the position standard deviation at every stored time."""
    density = np.abs(result.wavefunctions) ** 2
    mean = expectation_position(result)
    mean_square = result.spacing * np.sum(
        density * result.grid[np.newaxis, :] ** 2, axis=1
    )
    return np.sqrt(np.maximum(mean_square - mean**2, 0.0))


def energy_expectations(result: EvolutionResult) -> FloatArray:
    """Return the real energy expectation at every stored time."""
    applied = (result.hamiltonian @ result.wavefunctions.T).T
    values = result.spacing * np.sum(result.wavefunctions.conj() * applied, axis=1)
    return np.asarray(values.real, dtype=np.float64)


def state_l2_error(
    numerical: ComplexArray,
    reference: ComplexArray,
    spacing: float,
) -> float:
    """Return the discrete L2 distance between two phase-aligned states."""
    overlap = spacing * np.vdot(reference, numerical)
    if overlap != 0.0:
        numerical = numerical * np.exp(-1j * np.angle(overlap))
    return float(np.sqrt(spacing * np.sum(np.abs(numerical - reference) ** 2)))


def main() -> None:
    result = solve_free_packet()
    norms = probability_norms(result)
    positions = expectation_position(result)
    widths = position_width(result)
    energies = energy_expectations(result)
    print(f"initial norm:       {norms[0]:.12f}")
    print(f"maximum norm drift: {np.max(np.abs(norms - norms[0])):.3e}")
    print(f"initial <x>:        {positions[0]:.6f}")
    print(f"final <x>:          {positions[-1]:.6f}")
    print(f"initial width:      {widths[0]:.6f}")
    print(f"final width:        {widths[-1]:.6f}")
    print(f"maximum energy drift: {np.max(np.abs(energies - energies[0])):.3e}")


if __name__ == "__main__":
    main()
