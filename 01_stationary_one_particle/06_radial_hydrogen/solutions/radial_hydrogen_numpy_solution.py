"""Sparse NumPy/SciPy solver for the reduced radial hydrogen equation."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RadialResult:
    """Low-energy reduced radial states for one angular-momentum sector."""

    grid: FloatArray
    spacing: float
    effective_potential: FloatArray
    energies: FloatArray
    radial_wavefunctions: FloatArray
    hamiltonian: csr_matrix
    angular_momentum: int
    nuclear_charge: float


def _validate_inputs(
    num_points: int,
    num_states: int,
    r_max: float,
    angular_momentum: int,
    nuclear_charge: float,
    mass: float,
    hbar: float,
) -> None:
    if num_points < 3:
        raise ValueError("num_points must be at least 3")
    if not 1 <= num_states < num_points:
        raise ValueError("num_states must be between 1 and num_points - 1")
    if not isinstance(angular_momentum, int) or angular_momentum < 0:
        raise ValueError("angular_momentum must be a nonnegative integer")
    values = np.array([r_max, nuclear_charge, mass, hbar])
    if not np.isfinite(values).all():
        raise ValueError("physical parameters must be finite")
    if r_max <= 0.0:
        raise ValueError("r_max must be positive")
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")


def make_radial_grid(
    num_points: int,
    r_max: float = 40.0,
) -> tuple[FloatArray, float]:
    """Return interior radial points, excluding both Dirichlet boundaries."""
    _validate_inputs(num_points, 1, r_max, 0, 1.0, 1.0, 1.0)
    spacing = r_max / (num_points + 1)
    indices = np.arange(1, num_points + 1, dtype=np.float64)
    return spacing * indices, spacing


def effective_potential(
    grid: FloatArray,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return the Coulomb plus centrifugal effective potential."""
    if not isinstance(angular_momentum, int) or angular_momentum < 0:
        raise ValueError("angular_momentum must be a nonnegative integer")
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")
    centrifugal = (
        hbar**2 * angular_momentum * (angular_momentum + 1) / (2.0 * mass * grid**2)
    )
    return centrifugal - nuclear_charge / grid


def build_hamiltonian(
    num_points: int = 800,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> tuple[FloatArray, float, FloatArray, csr_matrix]:
    """Construct the sparse tridiagonal reduced radial Hamiltonian."""
    _validate_inputs(num_points, 1, r_max, angular_momentum, nuclear_charge, mass, hbar)
    grid, spacing = make_radial_grid(num_points, r_max)
    potential = effective_potential(grid, angular_momentum, nuclear_charge, mass, hbar)
    kinetic_diagonal = hbar**2 / (mass * spacing**2)
    kinetic_off_diagonal = -(hbar**2) / (2.0 * mass * spacing**2)
    main = np.full(num_points, kinetic_diagonal, dtype=np.float64) + potential
    off = np.full(num_points - 1, kinetic_off_diagonal, dtype=np.float64)
    # scipy-stubs cannot express differently sized tridiagonal arrays.
    hamiltonian = diags(  # type: ignore[call-overload]
        [off, main, off],
        offsets=[-1, 0, 1],
        shape=(num_points, num_points),
        format="csr",
        dtype=np.float64,
    )
    return grid, spacing, potential, hamiltonian


def normalize_wavefunctions(
    radial_wavefunctions: FloatArray,
    spacing: float,
) -> FloatArray:
    """Normalize columns so that the discrete integral of |u(r)|^2 is one."""
    norms = np.sqrt(spacing * np.sum(np.abs(radial_wavefunctions) ** 2, axis=0))
    return radial_wavefunctions / norms


def solve_radial_hydrogen(
    num_points: int = 800,
    num_states: int = 3,
    r_max: float = 40.0,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> RadialResult:
    """Solve for the lowest states in a fixed angular-momentum sector."""
    _validate_inputs(
        num_points,
        num_states,
        r_max,
        angular_momentum,
        nuclear_charge,
        mass,
        hbar,
    )
    grid, spacing, potential, hamiltonian = build_hamiltonian(
        num_points,
        r_max,
        angular_momentum,
        nuclear_charge,
        mass,
        hbar,
    )
    energies, wavefunctions = eigsh(hamiltonian, k=num_states, which="SA")
    order = np.argsort(energies)
    energies = energies[order]
    selected = normalize_wavefunctions(wavefunctions[:, order], spacing)
    return RadialResult(
        grid,
        spacing,
        potential,
        energies,
        selected,
        hamiltonian,
        angular_momentum,
        nuclear_charge,
    )


def principal_quantum_numbers(
    num_states: int,
    angular_momentum: int = 0,
) -> NDArray[np.int64]:
    """Return n = l + 1, l + 2, ... for one radial sector."""
    if num_states < 1:
        raise ValueError("num_states must be positive")
    if not isinstance(angular_momentum, int) or angular_momentum < 0:
        raise ValueError("angular_momentum must be a nonnegative integer")
    return np.arange(
        angular_momentum + 1,
        angular_momentum + num_states + 1,
        dtype=np.int64,
    )


def analytical_energies(
    num_states: int,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return exact Coulomb energies for a fixed angular-momentum sector."""
    if nuclear_charge <= 0.0 or mass <= 0.0 or hbar <= 0.0:
        raise ValueError("nuclear_charge, mass, and hbar must be positive")
    quantum_numbers = principal_quantum_numbers(num_states, angular_momentum)
    energies = -mass * nuclear_charge**2 / (2.0 * hbar**2 * quantum_numbers**2)
    return energies.astype(np.float64)


def expectation_radius(result: RadialResult) -> FloatArray:
    """Return <r> for every stored reduced radial state."""
    density = np.abs(result.radial_wavefunctions) ** 2
    return result.spacing * np.sum(result.grid[:, np.newaxis] * density, axis=0)


def analytical_expectation_radius(
    num_states: int,
    angular_momentum: int = 0,
    nuclear_charge: float = 1.0,
    mass: float = 1.0,
    hbar: float = 1.0,
) -> FloatArray:
    """Return exact hydrogenic <r> values for one angular-momentum sector."""
    quantum_numbers = principal_quantum_numbers(num_states, angular_momentum)
    scale = hbar**2 / (2.0 * mass * nuclear_charge)
    radii = scale * (
        3.0 * quantum_numbers**2 - angular_momentum * (angular_momentum + 1)
    )
    return radii.astype(np.float64)


def residual_norms(result: RadialResult) -> FloatArray:
    """Return discrete norms of H u_n - E_n u_n."""
    residuals = (
        result.hamiltonian @ result.radial_wavefunctions
        - result.radial_wavefunctions * result.energies[np.newaxis, :]
    )
    return np.sqrt(result.spacing * np.sum(np.abs(residuals) ** 2, axis=0))


def main() -> None:
    result = solve_radial_hydrogen()
    exact_energies = analytical_energies(len(result.energies))
    radii = expectation_radius(result)
    exact_radii = analytical_expectation_radius(len(result.energies))
    print(" n   numerical E       exact E       abs error       <r>      exact <r>")
    for index, energy in enumerate(result.energies):
        principal_n = index + result.angular_momentum + 1
        print(
            f"{principal_n:>2}  {energy:>13.8f}  {exact_energies[index]:>12.8f}  "
            f"{abs(energy - exact_energies[index]):>12.3e}  "
            f"{radii[index]:>8.4f}  {exact_radii[index]:>11.4f}"
        )


if __name__ == "__main__":
    main()
