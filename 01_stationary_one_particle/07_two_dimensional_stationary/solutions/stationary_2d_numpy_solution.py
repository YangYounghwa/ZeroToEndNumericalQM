"""Sparse NumPy/SciPy solver for stationary two-dimensional potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh

FloatArray = NDArray[np.float64]
Potential2D = Callable[[FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class Stationary2DResult:
    """Low-energy states stored with shape (y, x, state)."""

    x_grid: FloatArray
    y_grid: FloatArray
    spacing_x: float
    spacing_y: float
    potential: FloatArray
    energies: FloatArray
    wavefunctions: FloatArray
    hamiltonian: csr_matrix


def make_axis(
    num_points: int,
    minimum: float,
    maximum: float,
) -> tuple[FloatArray, float]:
    """Return one axis of interior Dirichlet-grid points."""
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not np.isfinite([minimum, maximum]).all() or minimum >= maximum:
        raise ValueError("axis bounds must be finite and increasing")
    spacing = (maximum - minimum) / (num_points + 1)
    indices = np.arange(1, num_points + 1, dtype=np.float64)
    return minimum + spacing * indices, spacing


def kinetic_1d(
    num_points: int,
    spacing: float,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> csr_matrix:
    """Return one sparse second-order kinetic operator."""
    if num_points < 3 or spacing <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("grid and physical scales must be positive")
    scale = hbar**2 / (2.0 * mass * spacing**2)
    main = np.full(num_points, 2.0 * scale, dtype=np.float64)
    off = np.full(num_points - 1, -scale, dtype=np.float64)
    return diags(  # type: ignore[call-overload,no-any-return]
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="csr",
        dtype=np.float64,
    )


def anisotropic_oscillator_potential(
    x_mesh: FloatArray,
    y_mesh: FloatArray,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.4,
    mass: float = 1.0,
) -> FloatArray:
    """Return a two-dimensional anisotropic harmonic potential."""
    if angular_frequency_x <= 0.0 or angular_frequency_y <= 0.0 or mass <= 0.0:
        raise ValueError("frequencies and mass must be positive")
    return (
        0.5
        * mass
        * (angular_frequency_x**2 * x_mesh**2 + angular_frequency_y**2 * y_mesh**2)
    )


def build_hamiltonian(
    potential_function: Potential2D,
    num_x: int = 32,
    num_y: int = 30,
    x_min: float = -7.0,
    x_max: float = 7.0,
    y_min: float = -6.0,
    y_max: float = 6.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, FloatArray, float, float, FloatArray, csr_matrix]:
    """Build H = Iy kron Tx + Ty kron Ix + diag(V)."""
    x_grid, spacing_x = make_axis(num_x, x_min, x_max)
    y_grid, spacing_y = make_axis(num_y, y_min, y_max)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid, indexing="xy")
    raw_potential = np.asarray(potential_function(x_mesh, y_mesh))
    if raw_potential.shape != (num_y, num_x):
        raise ValueError("potential_function must return shape (num_y, num_x)")
    if np.iscomplexobj(raw_potential):
        raise ValueError("potential must be real")
    potential = raw_potential.astype(np.float64, copy=False)
    if not np.isfinite(potential).all():
        raise ValueError("potential must contain finite values")
    kinetic_x = kinetic_1d(num_x, spacing_x, mass, hbar)
    kinetic_y = kinetic_1d(num_y, spacing_y, mass, hbar)
    identity_x = eye(num_x, format="csr", dtype=np.float64)
    identity_y = eye(num_y, format="csr", dtype=np.float64)
    kinetic = kron(identity_y, kinetic_x, format="csr") + kron(
        kinetic_y, identity_x, format="csr"
    )
    potential_matrix = diags(potential.ravel(order="C"), format="csr")
    hamiltonian = csr_matrix(kinetic + potential_matrix)
    return (
        x_grid,
        y_grid,
        spacing_x,
        spacing_y,
        potential,
        hamiltonian,
    )


def solve_stationary_2d(
    potential_function: Potential2D = anisotropic_oscillator_potential,
    num_x: int = 32,
    num_y: int = 30,
    num_states: int = 6,
    x_min: float = -7.0,
    x_max: float = 7.0,
    y_min: float = -6.0,
    y_max: float = 6.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Stationary2DResult:
    """Solve for the lowest states of a real two-dimensional potential."""
    total_points = num_x * num_y
    if not 1 <= num_states < total_points:
        raise ValueError("num_states must be between 1 and num_x*num_y - 1")
    x_grid, y_grid, dx, dy, potential, hamiltonian = build_hamiltonian(
        potential_function,
        num_x,
        num_y,
        x_min,
        x_max,
        y_min,
        y_max,
        mass,
        hbar,
    )
    energies, vectors = eigsh(hamiltonian, k=num_states, which="SA")
    order = np.argsort(energies)
    energies = np.asarray(energies[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    norms = np.sqrt(dx * dy * np.sum(np.abs(vectors) ** 2, axis=0))
    wavefunctions = (vectors / norms).reshape(num_y, num_x, num_states)
    return Stationary2DResult(
        x_grid,
        y_grid,
        dx,
        dy,
        potential,
        energies,
        wavefunctions,
        hamiltonian,
    )


def analytical_oscillator_energies(
    num_states: int,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.4,
    hbar: float = 1.0,
) -> FloatArray:
    """Return sorted exact energies of the anisotropic oscillator."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    quantum_numbers = np.arange(num_states, dtype=np.float64)
    energies = hbar * (
        angular_frequency_x * (quantum_numbers[:, None] + 0.5)
        + angular_frequency_y * (quantum_numbers[None, :] + 0.5)
    )
    return np.sort(energies.ravel())[:num_states].astype(np.float64)


def expectation_position(
    result: Stationary2DResult,
) -> tuple[FloatArray, FloatArray]:
    """Return (<x>, <y>) for every stored state."""
    density = np.abs(result.wavefunctions) ** 2
    weight = result.spacing_x * result.spacing_y
    x_values = weight * np.sum(density * result.x_grid[None, :, None], axis=(0, 1))
    y_values = weight * np.sum(density * result.y_grid[:, None, None], axis=(0, 1))
    return x_values.astype(np.float64), y_values.astype(np.float64)


def residual_norms(result: Stationary2DResult) -> FloatArray:
    """Return discrete norms of H psi_n - E_n psi_n."""
    vectors = result.wavefunctions.reshape(-1, len(result.energies))
    residuals = result.hamiltonian @ vectors - vectors * result.energies[None, :]
    weight = result.spacing_x * result.spacing_y
    return np.sqrt(weight * np.sum(np.abs(residuals) ** 2, axis=0))


def main() -> None:
    result = solve_stationary_2d()
    exact = analytical_oscillator_energies(len(result.energies))
    x_values, y_values = expectation_position(result)
    print("state   numerical E      exact E      abs error      <x>       <y>")
    for state, energy in enumerate(result.energies):
        print(
            f"{state:>5}  {energy:>13.8f}  {exact[state]:>11.8f}  "
            f"{abs(energy - exact[state]):>11.3e}  "
            f"{x_values[state]:>8.2e}  {y_values[state]:>8.2e}"
        )


if __name__ == "__main__":
    main()
