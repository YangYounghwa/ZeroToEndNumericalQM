"""NumPy/SciPy evolution of a coherent packet in a harmonic potential."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import expm_multiply, splu

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class EvolutionResult:
    """A harmonic-packet trajectory and its discrete Hamiltonian."""

    grid: FloatArray
    spacing: float
    times: FloatArray
    wavefunctions: ComplexArray
    potential: FloatArray
    hamiltonian: csr_matrix
    angular_frequency: float


def make_grid(
    num_points: int,
    x_min: float = -10.0,
    x_max: float = 10.0,
) -> tuple[FloatArray, float]:
    """Return interior points of the finite computational domain."""
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not np.isfinite([x_min, x_max]).all() or x_min >= x_max:
        raise ValueError("x_min and x_max must be finite with x_min < x_max")
    spacing = (x_max - x_min) / (num_points + 1)
    indices = np.arange(1, num_points + 1, dtype=np.float64)
    return x_min + spacing * indices, spacing


def harmonic_potential(
    grid: FloatArray,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
) -> FloatArray:
    """Return V(x) = m omega^2 x^2 / 2."""
    if angular_frequency <= 0.0 or mass <= 0.0:
        raise ValueError("angular_frequency and mass must be positive")
    return 0.5 * mass * angular_frequency**2 * grid**2


def build_hamiltonian(
    grid: FloatArray,
    spacing: float,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, csr_matrix]:
    """Construct the sparse kinetic-plus-harmonic Hamiltonian."""
    if spacing <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("spacing, mass, and hbar must be positive")
    num_points = len(grid)
    potential = harmonic_potential(grid, angular_frequency, mass)
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = np.full(num_points, 2.0 * scale, dtype=np.float64) + potential
    off = np.full(num_points - 1, -scale, dtype=np.float64)
    # scipy-stubs cannot express differently sized tridiagonal arrays.
    hamiltonian = diags(  # type: ignore[call-overload]
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="csr",
        dtype=np.float64,
    )
    return potential, hamiltonian


def normalize_wavefunction(wavefunction: ComplexArray, spacing: float) -> ComplexArray:
    """Normalize one sampled state."""
    norm = float(np.sqrt(spacing * np.sum(np.abs(wavefunction) ** 2)))
    if norm == 0.0 or not np.isfinite(norm):
        raise ValueError("wavefunction must have a finite, nonzero norm")
    return wavefunction / norm


def coherent_state(
    grid: FloatArray,
    center: float = 2.0,
    wave_number: float = 0.7,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> ComplexArray:
    """Return a displaced oscillator-ground-state Gaussian."""
    if angular_frequency <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("angular_frequency, mass, and hbar must be positive")
    width = np.sqrt(hbar / (2.0 * mass * angular_frequency))
    envelope = np.exp(-((grid - center) ** 2) / (4.0 * width**2))
    phase = np.exp(1j * wave_number * (grid - center))
    return np.asarray(envelope * phase, dtype=np.complex128)


def propagate_crank_nicolson(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time_step: float,
    num_steps: int,
    hbar: float = 1.0,
) -> tuple[FloatArray, ComplexArray]:
    """Propagate one state with a reused Crank-Nicolson factorization."""
    num_points = hamiltonian.shape[0]
    if hamiltonian.shape != (num_points, num_points):
        raise ValueError("hamiltonian must be square")
    if initial_state.shape != (num_points,):
        raise ValueError("initial_state shape must match hamiltonian")
    if num_steps < 1 or time_step == 0.0 or spacing <= 0.0 or hbar <= 0.0:
        raise ValueError("step count and scales must be valid and nonzero")
    state = normalize_wavefunction(initial_state.astype(np.complex128), spacing)
    identity = eye(num_points, format="csc", dtype=np.complex128)
    coefficient = 0.5j * time_step / hbar
    left: csc_matrix = identity + coefficient * hamiltonian
    right: csc_matrix = identity - coefficient * hamiltonian
    factorization = splu(left)
    history = np.empty((num_steps + 1, num_points), dtype=np.complex128)
    history[0] = state
    for step in range(1, num_steps + 1):
        state = factorization.solve(right @ state)
        history[step] = state
    times = time_step * np.arange(num_steps + 1, dtype=np.float64)
    return times, history


def solve_harmonic_packet(
    num_points: int = 400,
    x_min: float = -10.0,
    x_max: float = 10.0,
    center: float = 2.0,
    wave_number: float = 0.7,
    angular_frequency: float = 1.0,
    time_step: float = 0.01,
    num_steps: int = 628,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> EvolutionResult:
    """Build and evolve a coherent packet in the harmonic potential."""
    grid, spacing = make_grid(num_points, x_min, x_max)
    potential, hamiltonian = build_hamiltonian(
        grid, spacing, angular_frequency, mass, hbar
    )
    initial = coherent_state(grid, center, wave_number, angular_frequency, mass, hbar)
    times, history = propagate_crank_nicolson(
        initial, hamiltonian, spacing, time_step, num_steps, hbar
    )
    return EvolutionResult(
        grid, spacing, times, history, potential, hamiltonian, angular_frequency
    )


def sparse_exponential_state(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> ComplexArray:
    """Apply exp(-i H t / hbar) without forming a dense propagator."""
    if not np.isfinite([spacing, time, hbar]).all() or spacing <= 0 or hbar <= 0:
        raise ValueError("time must be finite; spacing and hbar finite and positive")
    if hamiltonian.shape != (initial_state.size, initial_state.size):
        raise ValueError("hamiltonian must be square and match initial_state")
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")
    state = normalize_wavefunction(initial_state.astype(np.complex128), spacing)
    generator = (-1j * time / hbar) * hamiltonian
    return np.asarray(
        expm_multiply(generator, state, traceA=generator.diagonal().sum()),
        dtype=np.complex128,
    )


def matrix_exponential_state(
    initial_state: ComplexArray,
    hamiltonian: csr_matrix,
    spacing: float,
    time: float,
    hbar: float = 1.0,
) -> ComplexArray:
    """Return a dense matrix-exponential reference on a small grid."""
    if time < 0.0 or hbar <= 0.0:
        raise ValueError("time must be nonnegative and hbar positive")
    state = normalize_wavefunction(initial_state.astype(np.complex128), spacing)
    propagator = expm((-1j * time / hbar) * hamiltonian.toarray())
    return np.asarray(propagator @ state, dtype=np.complex128)


def analytical_center(
    times: FloatArray,
    initial_center: float = 2.0,
    initial_wave_number: float = 0.7,
    angular_frequency: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return the exact coherent-state center trajectory."""
    initial_momentum = hbar * initial_wave_number
    return np.asarray(
        initial_center * np.cos(angular_frequency * times)
        + initial_momentum
        * np.sin(angular_frequency * times)
        / (mass * angular_frequency),
        dtype=np.float64,
    )


def probability_norms(result: EvolutionResult) -> FloatArray:
    """Return probability norm versus time."""
    return result.spacing * np.sum(np.abs(result.wavefunctions) ** 2, axis=1)


def expectation_position(result: EvolutionResult) -> FloatArray:
    """Return <x>(t)."""
    density = np.abs(result.wavefunctions) ** 2
    return result.spacing * np.sum(density * result.grid[np.newaxis, :], axis=1)


def position_width(result: EvolutionResult) -> FloatArray:
    """Return the position standard deviation versus time."""
    density = np.abs(result.wavefunctions) ** 2
    mean = expectation_position(result)
    mean_square = result.spacing * np.sum(
        density * result.grid[np.newaxis, :] ** 2, axis=1
    )
    return np.sqrt(np.maximum(mean_square - mean**2, 0.0))


def energy_expectations(result: EvolutionResult) -> FloatArray:
    """Return energy expectation versus time."""
    applied = (result.hamiltonian @ result.wavefunctions.T).T
    values = result.spacing * np.sum(result.wavefunctions.conj() * applied, axis=1)
    return np.asarray(values.real, dtype=np.float64)


def state_fidelity(
    first: ComplexArray,
    second: ComplexArray,
    spacing: float,
) -> float:
    """Return the global-phase-independent squared overlap."""
    overlap = spacing * np.vdot(first, second)
    return float(np.abs(overlap) ** 2)


def state_l2_error(
    numerical: ComplexArray,
    reference: ComplexArray,
    spacing: float,
) -> float:
    """Return phase-aligned discrete L2 error."""
    overlap = spacing * np.vdot(reference, numerical)
    if overlap != 0.0:
        numerical = numerical * np.exp(-1j * np.angle(overlap))
    return float(np.sqrt(spacing * np.sum(np.abs(numerical - reference) ** 2)))


def main() -> None:
    result = solve_harmonic_packet()
    norms = probability_norms(result)
    centers = expectation_position(result)
    exact_centers = analytical_center(result.times)
    widths = position_width(result)
    energies = energy_expectations(result)
    print(f"maximum norm drift:   {np.max(np.abs(norms - norms[0])):.3e}")
    print(f"maximum energy drift: {np.max(np.abs(energies - energies[0])):.3e}")
    print(f"maximum center error: {np.max(np.abs(centers - exact_centers)):.3e}")
    print(f"width variation:      {np.max(widths) - np.min(widths):.3e}")
    print(
        f"one-period fidelity:  {state_fidelity(result.wavefunctions[0], result.wavefunctions[-1], result.spacing):.10f}"
    )


if __name__ == "__main__":
    main()
