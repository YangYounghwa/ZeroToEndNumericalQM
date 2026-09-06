from dataclasses import replace

import infinite_square_well_numpy_solution as reference
import infinite_square_well_torch_solution as solution
import numpy as np
import torch


def test_residual_detects_wrong_energy_despite_normalized_state() -> None:
    result = solution.solve_infinite_well(num_points=70, num_states=3)
    assert torch.max(solution.residual_norms(result)) < 1e-10
    wrong = replace(result, energies=result.energies + 0.2)
    assert torch.allclose(
        solution.residual_norms(wrong),
        torch.full((3,), 0.2, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    numpy_result = reference.solve_infinite_well(num_points=70, num_states=3)
    assert np.max(reference.residual_norms(numpy_result)) < 1e-10
    numpy_wrong = replace(numpy_result, energies=numpy_result.energies + 0.2)
    np.testing.assert_allclose(
        reference.residual_norms(numpy_wrong), 0.2, atol=1e-10, rtol=0
    )
