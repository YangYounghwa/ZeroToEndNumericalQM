"""Sparse NumPy/SciPy solver for stationary three-dimensional potentials."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import eigsh

FloatArray = NDArray[np.float64]
Potential3D = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]


@dataclass(frozen=True)
class Stationary3DResult:
    """Low-energy states stored with shape (z, y, x, state)."""

    x_grid: FloatArray
    y_grid: FloatArray
    z_grid: FloatArray
    spacing_x: float
    spacing_y: float
    spacing_z: float
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
    """Return one sparse finite-difference kinetic operator."""
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
    z_mesh: FloatArray,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.3,
    angular_frequency_z: float = 1.7,
    mass: float = 1.0,
) -> FloatArray:
    """Return a three-dimensional anisotropic harmonic potential."""
    frequencies = [angular_frequency_x, angular_frequency_y, angular_frequency_z]
    if min(frequencies) <= 0.0 or mass <= 0.0:
        raise ValueError("frequencies and mass must be positive")
    return (
        0.5
        * mass
        * (
            angular_frequency_x**2 * x_mesh**2
            + angular_frequency_y**2 * y_mesh**2
            + angular_frequency_z**2 * z_mesh**2
        )
    )


def build_hamiltonian(
    potential_function: Potential3D,
    num_x: int = 14,
    num_y: int = 13,
    num_z: int = 12,
    x_min: float = -5.0,
    x_max: float = 5.0,
    y_min: float = -4.5,
    y_max: float = 4.5,
    z_min: float = -4.0,
    z_max: float = 4.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[
    FloatArray,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    FloatArray,
    csr_matrix,
]:
    """Build the sparse 3D Kronecker-sum Hamiltonian."""
    x_grid, dx = make_axis(num_x, x_min, x_max)
    y_grid, dy = make_axis(num_y, y_min, y_max)
    z_grid, dz = make_axis(num_z, z_min, z_max)
    z_mesh, y_mesh, x_mesh = np.meshgrid(z_grid, y_grid, x_grid, indexing="ij")
    raw_potential = np.asarray(potential_function(x_mesh, y_mesh, z_mesh))
    expected_shape = (num_z, num_y, num_x)
    if raw_potential.shape != expected_shape:
        raise ValueError("potential_function must return shape (num_z, num_y, num_x)")
    if np.iscomplexobj(raw_potential):
        raise ValueError("potential must be real")
    potential = raw_potential.astype(np.float64, copy=False)
    if not np.isfinite(potential).all():
        raise ValueError("potential must contain finite values")
    kinetic_x = kinetic_1d(num_x, dx, mass, hbar)
    kinetic_y = kinetic_1d(num_y, dy, mass, hbar)
    kinetic_z = kinetic_1d(num_z, dz, mass, hbar)
    identity_x = eye(num_x, format="csr", dtype=np.float64)
    identity_y = eye(num_y, format="csr", dtype=np.float64)
    identity_z = eye(num_z, format="csr", dtype=np.float64)
    term_x = kron(kron(identity_z, identity_y), kinetic_x, format="csr")
    term_y = kron(kron(identity_z, kinetic_y), identity_x, format="csr")
    term_z = kron(kron(kinetic_z, identity_y), identity_x, format="csr")
    potential_matrix = diags(potential.ravel(order="C"), format="csr")
    hamiltonian = csr_matrix(term_x + term_y + term_z + potential_matrix)
    return x_grid, y_grid, z_grid, dx, dy, dz, potential, hamiltonian


def solve_stationary_3d(
    potential_function: Potential3D = anisotropic_oscillator_potential,
    num_x: int = 14,
    num_y: int = 13,
    num_z: int = 12,
    num_states: int = 6,
    x_min: float = -5.0,
    x_max: float = 5.0,
    y_min: float = -4.5,
    y_max: float = 4.5,
    z_min: float = -4.0,
    z_max: float = 4.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> Stationary3DResult:
    """Solve for the lowest states of a real 3D potential."""
    total_points = num_x * num_y * num_z
    if not 1 <= num_states < total_points:
        raise ValueError("num_states must be below the 3D Hilbert-space size")
    values = build_hamiltonian(
        potential_function,
        num_x,
        num_y,
        num_z,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        mass,
        hbar,
    )
    x_grid, y_grid, z_grid, dx, dy, dz, potential, hamiltonian = values
    energies, vectors = eigsh(hamiltonian, k=num_states, which="SA")
    order = np.argsort(energies)
    energies = np.asarray(energies[order], dtype=np.float64)
    vectors = np.asarray(vectors[:, order], dtype=np.float64)
    volume_element = dx * dy * dz
    norms = np.sqrt(volume_element * np.sum(np.abs(vectors) ** 2, axis=0))
    wavefunctions = (vectors / norms).reshape(num_z, num_y, num_x, num_states)
    return Stationary3DResult(
        x_grid,
        y_grid,
        z_grid,
        dx,
        dy,
        dz,
        potential,
        energies,
        wavefunctions,
        hamiltonian,
    )


def analytical_oscillator_energies(
    num_states: int,
    angular_frequency_x: float = 1.0,
    angular_frequency_y: float = 1.3,
    angular_frequency_z: float = 1.7,
    hbar: float = 1.0,
) -> FloatArray:
    """Return sorted exact energies of the anisotropic 3D oscillator."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    quantum_numbers = np.arange(num_states, dtype=np.float64)
    energies = hbar * (
        angular_frequency_x * (quantum_numbers[:, None, None] + 0.5)
        + angular_frequency_y * (quantum_numbers[None, :, None] + 0.5)
        + angular_frequency_z * (quantum_numbers[None, None, :] + 0.5)
    )
    return np.sort(energies.ravel())[:num_states].astype(np.float64)


def expectation_position(
    result: Stationary3DResult,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return (<x>, <y>, <z>) for every state."""
    density = np.abs(result.wavefunctions) ** 2
    weight = result.spacing_x * result.spacing_y * result.spacing_z
    x_values = weight * np.sum(
        density * result.x_grid[None, None, :, None], axis=(0, 1, 2)
    )
    y_values = weight * np.sum(
        density * result.y_grid[None, :, None, None], axis=(0, 1, 2)
    )
    z_values = weight * np.sum(
        density * result.z_grid[:, None, None, None], axis=(0, 1, 2)
    )
    return (
        x_values.astype(np.float64),
        y_values.astype(np.float64),
        z_values.astype(np.float64),
    )


def residual_norms(result: Stationary3DResult) -> FloatArray:
    """Return volume-weighted eigenpair residual norms."""
    vectors = result.wavefunctions.reshape(-1, len(result.energies))
    residuals = result.hamiltonian @ vectors - vectors * result.energies[None, :]
    weight = result.spacing_x * result.spacing_y * result.spacing_z
    return np.sqrt(weight * np.sum(np.abs(residuals) ** 2, axis=0))


def dense_hamiltonian_bytes(num_x: int, num_y: int, num_z: int) -> int:
    """Return float64 bytes required by a dense 3D Hamiltonian."""
    total_points = num_x * num_y * num_z
    return 8 * total_points**2


def main() -> None:
    result = solve_stationary_3d()
    exact = analytical_oscillator_energies(len(result.energies))
    print("state   numerical E      exact E      abs error")
    for state, energy in enumerate(result.energies):
        print(
            f"{state:>5}  {energy:>13.8f}  {exact[state]:>11.8f}  "
            f"{abs(energy - exact[state]):>11.3e}"
        )
    dense_megabytes = dense_hamiltonian_bytes(40, 40, 40) / 1024**2
    print(f"\nDense 40^3 Hamiltonian memory: {dense_megabytes:,.1f} MiB")


if __name__ == "__main__":
    main()
