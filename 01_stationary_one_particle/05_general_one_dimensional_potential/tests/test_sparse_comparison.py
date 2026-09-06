import general_potential_numpy_solution as reference
import general_potential_torch_solution as solution
import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix


def test_sparse_solver_matches_dense_torch_for_negative_energies() -> None:
    # A negative offset catches accidental which="SM" instead of "SA".
    sparse = reference.solve_stationary_sparse(
        lambda x: 0.5 * x**2 - 10, num_points=100, num_states=4
    )
    dense = solution.solve_stationary(
        lambda x: 0.5 * x**2 - 10, num_points=100, num_states=4
    )
    assert isinstance(sparse.hamiltonian, csr_matrix)
    assert sparse.hamiltonian.nnz == 3 * 100 - 2
    np.testing.assert_allclose(
        sparse.energies, dense.energies.numpy(), atol=1e-10, rtol=0
    )
    overlap = sparse.spacing * sparse.wavefunctions.T @ sparse.wavefunctions
    np.testing.assert_allclose(overlap, np.eye(4), atol=1e-12, rtol=0)
    assert np.max(reference.residual_norms(sparse)) < 1e-8


def test_sparse_solver_rejects_request_for_full_spectrum() -> None:
    with pytest.raises(ValueError, match="num_states < num_points"):
        reference.solve_stationary_sparse(np.zeros_like, num_points=10, num_states=10)


def test_small_residual_can_coexist_with_large_domain_error() -> None:
    small = solution.solve_stationary(
        lambda x: 0.5 * x**2, num_points=39, num_states=1, x_min=-1, x_max=1
    )
    large = solution.solve_stationary(
        lambda x: 0.5 * x**2, num_points=159, num_states=1, x_min=-4, x_max=4
    )
    assert abs(small.spacing - large.spacing) < 1e-14
    assert torch.max(solution.residual_norms(small)) < 1e-10
    assert abs(small.energies[0].item() - 0.5) > 0.5
    assert abs(large.energies[0].item() - 0.5) < 1e-3
