import numpy as np
import pytest
import stationary_2d_numpy_solution as solution
from scipy.sparse import issparse


def test_grid_shapes_and_flattening_convention() -> None:
    result = solution.solve_stationary_2d(
        solution.coupled_quartic_potential, num_x=12, num_y=10, num_states=3
    )
    assert result.potential.shape == (10, 12)
    assert result.wavefunctions.shape == (10, 12, 3)
    assert result.hamiltonian.shape == (120, 120)


def test_hamiltonian_is_sparse_and_hermitian() -> None:
    *_, hamiltonian = solution.build_hamiltonian(
        solution.anisotropic_oscillator_potential,
        num_x=10,
        num_y=9,
    )
    assert issparse(hamiltonian)
    difference = hamiltonian - hamiltonian.T
    assert difference.nnz == 0


def test_anisotropic_oscillator_energies_match_reference() -> None:
    result = solution.solve_stationary_2d(
        solution.anisotropic_oscillator_potential,
        num_x=60,
        num_y=54,
        num_states=6,
    )
    exact = solution.analytical_oscillator_energies(6)
    assert np.allclose(result.energies, exact, atol=4.5e-2)


def test_wavefunctions_are_orthonormal_and_residuals_are_small() -> None:
    result = solution.solve_stationary_2d(
        solution.coupled_quartic_potential, num_x=18, num_y=16, num_states=5
    )
    vectors = result.wavefunctions.reshape(-1, 5)
    overlap = result.spacing_x * result.spacing_y * vectors.T @ vectors
    assert np.allclose(overlap, np.eye(5), atol=1e-12)
    assert np.all(solution.residual_norms(result) < 1e-10)


def test_symmetric_potential_has_zero_position_expectations() -> None:
    result = solution.solve_stationary_2d(
        solution.coupled_quartic_potential, num_x=20, num_y=18, num_states=5
    )
    x_values, y_values = solution.expectation_position(result)
    assert np.allclose(x_values, 0.0, atol=1e-12)
    assert np.allclose(y_values, 0.0, atol=1e-12)


def test_isotropic_first_excited_pair_is_degenerate() -> None:
    def isotropic(
        x_mesh: solution.FloatArray, y_mesh: solution.FloatArray
    ) -> solution.FloatArray:
        return 0.5 * (x_mesh**2 + y_mesh**2)

    result = solution.solve_stationary_2d(
        isotropic,
        num_x=28,
        num_y=28,
        num_states=3,
        x_min=-7.0,
        x_max=7.0,
        y_min=-7.0,
        y_max=7.0,
    )
    assert np.isclose(result.energies[1], result.energies[2], atol=1e-11)


@pytest.mark.parametrize(
    ("num_points", "minimum", "maximum", "message"),
    [(2, -1.0, 1.0, "num_points"), (10, 1.0, 1.0, "bounds")],
)
def test_invalid_axis_inputs_raise_value_error(
    num_points: int, minimum: float, maximum: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        solution.make_axis(num_points, minimum, maximum)


def test_invalid_potential_shape_is_rejected() -> None:
    def wrong_shape(
        _: solution.FloatArray, __: solution.FloatArray
    ) -> solution.FloatArray:
        return np.zeros((3, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="shape"):
        solution.build_hamiltonian(wrong_shape, num_x=8, num_y=7)


def test_coupled_quartic_is_not_separable_in_x_and_y() -> None:
    x_values = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    y_values = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float64)
    potential = solution.coupled_quartic_potential(x_values, y_values)
    separable_prediction = potential[0, 1] + potential[1, 0] - potential[0, 0]
    assert not np.isclose(potential[1, 1], separable_prediction)
