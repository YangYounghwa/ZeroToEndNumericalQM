import stationary_2d_numpy_solution as reference
import stationary_2d_torch_solution as solution
import torch


def test_isotropic_excited_multiplet_agrees_under_basis_rotations() -> None:
    result = solution.solve_stationary_2d(
        lambda x, y: 0.5 * (x**2 + y**2),
        num_x=12,
        num_y=12,
        num_states=4,
        x_min=-5,
        x_max=5,
        y_min=-5,
        y_max=5,
    )
    sparse = reference.solve_stationary_2d(
        lambda x, y: 0.5 * (x**2 + y**2),
        num_x=12,
        num_y=12,
        num_states=4,
        x_min=-5,
        x_max=5,
        y_min=-5,
        y_max=5,
    )
    vectors = result.wavefunctions.reshape(-1, 4)
    first = vectors[:, 1:3].to(torch.complex128)
    second = torch.from_numpy(sparse.wavefunctions.reshape(-1, 4)[:, 1:3]).to(
        torch.complex128
    )
    rotation = torch.tensor([[1, 1j], [1j, 1]], dtype=torch.complex128) / 2**0.5
    area = result.spacing_x * result.spacing_y
    assert torch.allclose(result.energies[1], result.energies[2], atol=1e-12, rtol=0)
    overlaps = solution.subspace_overlaps(first, second @ rotation, area)
    assert torch.allclose(
        overlaps, torch.ones(2, dtype=torch.float64), atol=1e-10, rtol=0
    )
    wrong = vectors[:, :2].to(torch.complex128)
    assert solution.subspace_overlaps(first, wrong, area).min() < 1e-10
