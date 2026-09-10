"""NumPy comparison on identical rectangular grids and batches."""

import numpy as np
import wave_packet_2d_numpy_solution as reference
import wave_packet_2d_torch_solution as solution


def test_coupled_numpy_torch_agreement() -> None:
    numpy_result = reference.solve_coupled()
    torch_result = solution.solve_coupled()
    np.testing.assert_allclose(
        numpy_result.potential, torch_result.potential.numpy(), atol=1e-13, rtol=0
    )
    np.testing.assert_allclose(
        numpy_result.wavefunctions,
        torch_result.wavefunctions.numpy(),
        atol=2e-12,
        rtol=0,
    )


def test_free_numpy_evolution_and_batch() -> None:
    grid = reference.make_grid(100, 96, 20, 20)
    momenta = ((0.8, -0.4), (-0.6, 0.7))
    initial = np.stack(
        [reference.free_gaussian(grid, momentum=p) for p in momenta], axis=2
    )
    kinetic = reference.kinetic_energy(grid)
    _, history = reference.propagate_batch(
        initial, kinetic, np.zeros_like(kinetic), grid.area, 0.1, 20, 20
    )
    for column, p in enumerate(momenta):
        np.testing.assert_allclose(
            history[-1, :, :, column],
            reference.free_gaussian(grid, time=2, momentum=p),
            atol=1e-10,
            rtol=0,
        )
