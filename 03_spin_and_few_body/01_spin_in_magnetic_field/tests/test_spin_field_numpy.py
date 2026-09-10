"""Independent NumPy eigenbasis and CN comparisons."""

from math import pi

import numpy as np
import spin_field_numpy_solution as reference
import spin_field_torch_solution as solution
import torch


def test_eigenbasis_matches_matrix_exp_including_zero_field() -> None:
    fields = np.array([[0, 0, 0], [0, 0, 1], [0.3, -0.4, 1]], dtype=np.float64)
    initial = np.stack(
        [reference.spinor(), reference.spinor(0), reference.spinor(pi / 3, pi / 5)]
    )
    times = np.linspace(-1, 10, 71)
    numpy_result = reference.evolve_constant(
        initial, fields, times, gamma=-1.7, hbar=0.6
    )
    torch_result = solution.evolve_constant(
        torch.from_numpy(initial),
        torch.from_numpy(fields),
        torch.from_numpy(times),
        gamma=-1.7,
        hbar=0.6,
    )
    np.testing.assert_allclose(
        numpy_result.states, torch_result.states.numpy(), atol=3e-14, rtol=0
    )
    axis = np.array([1, -2, 3], dtype=np.float64)
    np.testing.assert_allclose(
        reference.measurement_probabilities(numpy_result.states, axis),
        solution.measurement_probabilities(
            torch_result.states, torch.from_numpy(axis)
        ).numpy(),
        atol=3e-14,
        rtol=0,
    )


def test_numpy_cn_agreement_norm_and_energy() -> None:
    fields = np.array([[0.3, -0.4, 1], [1, 0, 0]], dtype=np.float64)
    initial = np.stack([reference.spinor(), reference.spinor(0)])
    numpy_result = reference.propagate_cn(initial, fields, 0.07, 100, 7)
    torch_result = solution.propagate_cn(
        torch.from_numpy(initial), torch.from_numpy(fields), 0.07, 100, 7
    )
    np.testing.assert_allclose(
        numpy_result.states, torch_result.states.numpy(), atol=3e-14, rtol=0
    )
    np.testing.assert_allclose(
        np.sum(np.abs(numpy_result.states) ** 2, axis=-1), 1, atol=3e-14, rtol=0
    )
    energies = reference.energy_expectations(numpy_result)
    np.testing.assert_allclose(
        energies, np.broadcast_to(energies[0], energies.shape), atol=3e-14, rtol=0
    )
