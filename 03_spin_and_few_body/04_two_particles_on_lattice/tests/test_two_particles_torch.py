from math import pi
from typing import cast

import numpy as np
import pytest
import torch
import two_particles_numpy_solution as reference
import two_particles_torch_solution as model

STATISTICS: tuple[model.Statistics, ...] = ("distinguishable", "boson", "fermion")


@pytest.mark.parametrize("statistics", STATISTICS)
@pytest.mark.parametrize("num_sites", [2, 3, 4])
def test_basis_geometry(num_sites: int, statistics: model.Statistics) -> None:
    sizes = {
        "distinguishable": num_sites**2,
        "boson": num_sites * (num_sites + 1) // 2,
        "fermion": num_sites * (num_sites - 1) // 2,
    }
    dimension = sizes[statistics]
    q = model.basis_embedding(num_sites, statistics)
    assert q.shape == (num_sites**2, dimension)
    assert q.dtype == torch.complex128
    torch.testing.assert_close(q.mH @ q, torch.eye(dimension, dtype=q.dtype))
    if statistics != "distinguishable":
        reshaped = q.reshape(num_sites, num_sites, dimension)
        sign = 1 if statistics == "boson" else -1
        torch.testing.assert_close(reshaped.transpose(0, 1), sign * reshaped)
    h = model.single_particle_hamiltonian(num_sites)
    h2 = model.two_particle_hamiltonian(h, statistics)
    identity = torch.eye(num_sites, dtype=h.dtype)
    labeled = torch.kron(h, identity) + torch.kron(identity, h)
    torch.testing.assert_close(labeled @ q, q @ h2)


@pytest.mark.parametrize("statistics", STATISTICS)
def test_complex_hamiltonian_batches_and_conservation(
    statistics: model.Statistics,
) -> None:
    generator = torch.Generator().manual_seed(74)
    raw = torch.randn(4, 4, dtype=torch.complex128, generator=generator)
    h = (raw + raw.mH) / 2
    h2 = model.two_particle_hamiltonian(h, statistics)
    numpy_h = reference.two_particle_hamiltonian(h.numpy(), statistics)
    np.testing.assert_allclose(h2.numpy(), numpy_h, atol=1e-12)
    dimension = h2.shape[0]
    states = torch.randn(2, 3, dimension, dtype=torch.complex128, generator=generator)
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    times = torch.tensor([0, 0.17, 0.91], dtype=torch.float64)
    history = model.evolve(h2, states, times, hbar=1.3)
    assert history.shape == (3, 2, 3, dimension)
    numpy_history = reference.evolve(numpy_h, states.numpy(), times.numpy(), hbar=1.3)
    np.testing.assert_allclose(history.numpy(), numpy_history, atol=2e-12)
    torch.testing.assert_close(
        history.abs().square().sum(-1), torch.ones(3, 2, 3, dtype=torch.float64)
    )
    energy = torch.einsum("...i,ij,...j->...", history.conj(), h2, history).real
    torch.testing.assert_close(energy, energy[0].expand_as(energy))
    occupancy = model.site_occupations(history, 4, statistics)
    torch.testing.assert_close(
        occupancy.sum(-1), 2 * torch.ones(3, 2, 3, dtype=torch.float64)
    )
    np.testing.assert_allclose(
        occupancy.numpy(),
        reference.site_occupations(numpy_history, 4, statistics),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        model.coincidence_probability(history, 4, statistics).numpy(),
        reference.coincidence_probability(numpy_history, 4, statistics),
        atol=2e-12,
    )
    recovered = model.evolve(h2, history[-1], -times[-1:], hbar=1.3)[0]
    torch.testing.assert_close(recovered, states, atol=2e-12, rtol=0)
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(
                history[:, i, j], model.evolve(h2, states[i, j], times, hbar=1.3)
            )


@pytest.mark.parametrize("statistics", STATISTICS)
def test_spectrum_and_two_site_formula(statistics: model.Statistics) -> None:
    h = model.single_particle_hamiltonian(4, hopping=0.7)
    h2 = model.two_particle_hamiltonian(h, statistics)
    one = -1.4 * torch.cos(pi * torch.arange(1, 5, dtype=torch.float64) / 5)
    expected = (
        torch.stack([one[i] + one[j] for i, j in model.basis_pairs(4, statistics)])
        .sort()
        .values
    )
    torch.testing.assert_close(torch.linalg.eigvalsh(h2), expected, atol=1e-12, rtol=0)
    h2 = model.two_particle_hamiltonian(
        model.single_particle_hamiltonian(2, hopping=0.8), statistics
    )
    times = torch.linspace(-2, 4, 29, dtype=torch.float64)
    history = model.evolve(
        h2, model.localized_state(2, statistics, (0, 1)), times, hbar=1.7
    )
    factor = {"distinguishable": 0.5, "boson": 1.0, "fermion": 0.0}[statistics]
    expected_probability = factor * torch.sin(2 * 0.8 * times / 1.7).square()
    torch.testing.assert_close(
        model.coincidence_probability(history, 2, statistics),
        expected_probability,
        atol=2e-12,
        rtol=0,
    )
    torch.testing.assert_close(
        model.site_occupations(history, 2, statistics),
        torch.ones(29, 2, dtype=torch.float64),
    )


def test_boson_double_occupancy_is_normalized() -> None:
    state = model.localized_state(3, "boson", (1, 1))
    assert float(model.coincidence_probability(state, 3, "boson")) == 1
    torch.testing.assert_close(
        model.site_occupations(state, 3, "boson"),
        torch.tensor([0, 2, 0], dtype=torch.float64),
    )


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_sites"):
        model.basis_embedding(1, "fermion")
    with pytest.raises(ValueError, match="statistics"):
        model.basis_pairs(2, cast(model.Statistics, "wrong"))
    with pytest.raises(ValueError, match="canonical"):
        model.localized_state(2, "fermion", (0, 0))
    with pytest.raises(ValueError, match="real vector"):
        model.single_particle_hamiltonian(
            2, onsite=torch.zeros(2, dtype=torch.complex128)
        )
    with pytest.raises(ValueError, match="finite"):
        model.single_particle_hamiltonian(2, onsite=torch.tensor([0, float("inf")]))
    with pytest.raises(ValueError, match="square"):
        model.two_particle_hamiltonian(torch.zeros(2, 3), "boson")
    with pytest.raises(ValueError, match="Hermitian"):
        model.two_particle_hamiltonian(torch.tensor([[0.0, 1.0], [0.0, 0.0]]), "boson")
    h2 = model.two_particle_hamiltonian(model.single_particle_hamiltonian(2), "boson")
    state = model.localized_state(2, "boson", (0, 1))
    times = torch.tensor([0.0], dtype=torch.float64)
    for invalid in (state * 2, state[:2], state * float("nan")):
        with pytest.raises(ValueError, match="states"):
            model.evolve(h2, invalid, times)
    for invalid_times in (
        torch.empty(0),
        torch.zeros(1, 1),
        torch.tensor([1j]),
        torch.tensor([float("nan")]),
    ):
        with pytest.raises(ValueError, match="times"):
            model.evolve(h2, state, invalid_times)
    with pytest.raises(ValueError, match="hbar"):
        model.evolve(h2, state, times, hbar=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("statistics", STATISTICS)
def test_cuda(statistics: model.Statistics) -> None:
    cpu_h = model.two_particle_hamiltonian(
        model.single_particle_hamiltonian(3), statistics
    )
    gpu_h = model.two_particle_hamiltonian(
        model.single_particle_hamiltonian(3, device="cuda"), statistics
    )
    state = model.localized_state(3, statistics, (0, 1))
    times = torch.tensor([0.0, 0.3], dtype=torch.float64)
    expected = model.evolve(cpu_h, state, times)
    actual = model.evolve(gpu_h, state.to("cuda"), times.to("cuda"))
    assert actual.device.type == "cuda"
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-11, rtol=0)
