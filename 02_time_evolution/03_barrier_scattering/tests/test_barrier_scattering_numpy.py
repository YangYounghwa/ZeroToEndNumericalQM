import barrier_scattering_numpy_solution as solution
import barrier_scattering_torch_solution as torch_reference
import numpy as np
import pytest
from scipy.sparse import issparse


def test_sparse_structure_and_barrier_edge_alignment() -> None:
    for points, denominator in ((439, 11), (839, 21), (1639, 41)):
        grid, spacing, potential, hamiltonian = solution.build_hamiltonian(points)
        assert issparse(hamiltonian)
        assert hamiltonian.nnz == 3 * points - 2
        assert np.isclose(spacing, 2 / denominator)
        assert np.isclose(spacing * np.count_nonzero(potential), 2.0)
        assert np.isclose(np.min(np.abs(grid - 1)), spacing / 2)


def test_refined_packet_transmission_matches_continuum_reference() -> None:
    exact = torch_reference.packet_transmission_reference().item()
    errors = []
    for points in (439, 839, 1639):
        result = solution.solve_scattering(
            num_points=points, time_step=0.01, num_steps=1400, store_every=100
        )
        regions = solution.region_probabilities(result)
        errors.append(abs(regions.right[-1] - exact))
        np.testing.assert_allclose(
            regions.left + regions.near + regions.right, 1, atol=2e-12, rtol=0
        )
        assert regions.near[-1] < 2e-4
        assert solution.boundary_probability(result).max() < 1e-6
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.001


def test_late_probability_plateau_and_free_packet_limit() -> None:
    result = solution.solve_scattering(num_points=439, num_steps=800, store_every=100)
    regions = solution.region_probabilities(result)
    assert regions.left[0] > 0.999
    assert regions.near[-1] < 2e-5
    assert abs(regions.right[-1] - regions.right[-2]) < 2e-6
    free = solution.solve_scattering(height=0, num_steps=700)
    free_regions = solution.region_probabilities(free)
    assert free_regions.right[-1] > 0.998
    assert free_regions.left[-1] < 0.001


def test_domain_comparison_detects_wall_distortion() -> None:
    large = solution.solve_scattering(
        num_points=1049, half_extent=50, time_step=0.01, num_steps=1400, store_every=10
    )
    errors = []
    for extent in (20, 40):
        result = solution.solve_scattering(
            num_points=21 * extent - 1,
            half_extent=extent,
            time_step=0.01,
            num_steps=1400,
            store_every=10,
        )
        start = (large.grid.size - result.grid.size) // 2
        cropped = large.wavefunctions[-1, start : start + result.grid.size]
        overlap = result.spacing * np.vdot(cropped, result.wavefunctions[-1])
        aligned = result.wavefunctions[-1] * np.exp(-1j * np.angle(overlap))
        errors.append(
            float(np.sqrt(result.spacing * np.sum(np.abs(aligned - cropped) ** 2)))
        )
    assert errors[0] > 0.1
    assert errors[1] < 1e-4


def test_sparse_exponential_and_cn_agree_at_small_step() -> None:
    grid, spacing, _, hamiltonian = solution.build_hamiltonian(79, 12)
    initial = solution.gaussian_packet(grid, center=-4, sigma=1)
    exact = solution.exponential_state(initial, hamiltonian, spacing, 0.2)
    times, states = solution.propagate(initial, hamiltonian, spacing, 0.001, 200, 77)
    assert times[-1] == 0.2
    assert states.shape == (4, 79)
    assert np.sqrt(spacing * np.sum(np.abs(states[-1] - exact) ** 2)) < 2e-6


def test_sparse_invalid_scales_and_zero_state() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        solution.build_hamiltonian(height=-1)
    with pytest.raises(ValueError, match="inside"):
        solution.build_hamiltonian(half_extent=1, width=2)
    with pytest.raises(ValueError, match="positive"):
        solution.make_grid(half_extent=float("nan"))
    _, spacing, _, hamiltonian = solution.build_hamiltonian(39, 10)
    with pytest.raises(ValueError, match="nonzero"):
        solution.propagate(np.zeros(39, dtype=np.complex128), hamiltonian, spacing)
