import numpy as np
import stationary_3d_numpy_solution as solution
from scipy.sparse import issparse


def test_three_dimensional_shapes_and_flattening() -> None:
    result = solution.solve_stationary_3d(num_x=7, num_y=6, num_z=5, num_states=3)
    assert result.potential.shape == (5, 6, 7)
    assert result.wavefunctions.shape == (5, 6, 7, 3)
    assert result.hamiltonian.shape == (210, 210)


def test_hamiltonian_is_sparse_and_hermitian() -> None:
    *_, hamiltonian = solution.build_hamiltonian(
        solution.anisotropic_oscillator_potential,
        num_x=6,
        num_y=5,
        num_z=4,
    )
    assert issparse(hamiltonian)
    assert (hamiltonian - hamiltonian.T).nnz == 0


def test_anisotropic_oscillator_low_energies_match_reference() -> None:
    result = solution.solve_stationary_3d(
        num_x=28,
        num_y=26,
        num_z=24,
        num_states=5,
    )
    exact = solution.analytical_oscillator_energies(5)
    assert np.allclose(result.energies, exact, atol=1.6e-1)


def test_wavefunctions_are_orthonormal_and_residuals_are_small() -> None:
    result = solution.solve_stationary_3d(num_x=8, num_y=7, num_z=6, num_states=4)
    vectors = result.wavefunctions.reshape(-1, 4)
    weight = result.spacing_x * result.spacing_y * result.spacing_z
    assert np.allclose(weight * vectors.T @ vectors, np.eye(4), atol=1e-12)
    assert np.all(solution.residual_norms(result) < 1e-10)


def test_symmetric_potential_has_zero_position_expectations() -> None:
    result = solution.solve_stationary_3d(num_x=9, num_y=8, num_z=7, num_states=4)
    x_values, y_values, z_values = solution.expectation_position(result)
    assert np.allclose(x_values, 0.0, atol=1e-12)
    assert np.allclose(y_values, 0.0, atol=1e-12)
    assert np.allclose(z_values, 0.0, atol=1e-12)


def test_isotropic_first_excited_triplet_is_degenerate() -> None:
    def isotropic(
        x_mesh: solution.FloatArray,
        y_mesh: solution.FloatArray,
        z_mesh: solution.FloatArray,
    ) -> solution.FloatArray:
        return 0.5 * (x_mesh**2 + y_mesh**2 + z_mesh**2)

    result = solution.solve_stationary_3d(
        isotropic,
        num_x=10,
        num_y=10,
        num_z=10,
        num_states=4,
        x_min=-6.0,
        x_max=6.0,
        y_min=-6.0,
        y_max=6.0,
        z_min=-6.0,
        z_max=6.0,
    )
    assert np.allclose(result.energies[1:], result.energies[1], atol=1e-11)


def test_dense_memory_estimate_exposes_three_dimensional_scaling() -> None:
    assert solution.dense_hamiltonian_bytes(10, 10, 10) == 8_000_000
    assert solution.dense_hamiltonian_bytes(20, 20, 20) == 512_000_000
