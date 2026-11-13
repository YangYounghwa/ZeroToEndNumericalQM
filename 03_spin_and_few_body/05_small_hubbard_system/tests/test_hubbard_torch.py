from math import sqrt

import hubbard_numpy_solution as reference
import hubbard_torch_solution as solution
import numpy as np
import pytest
import torch


def test_fermion_algebra() -> None:
    operators = [solution.annihilation(q) for q in range(4)]
    identity = torch.eye(16, dtype=torch.complex128)
    for p, a in enumerate(operators):
        for q, b in enumerate(operators):
            torch.testing.assert_close(a @ b.mH + b.mH @ a, identity * (p == q))
            torch.testing.assert_close(a @ b + b @ a, torch.zeros_like(identity))


@pytest.mark.parametrize(
    ("u", "bias", "t"),
    [
        (0.0, 0.0, 1.0),
        (4.0, 0.0, 1.0),
        (-3.0, 1.2, 0.7),
        (8.0, -2.0, -1.0),
        (3.0, 2.0, 0.0),
    ],
)
def test_independent_matrix_and_spectrum(u: float, bias: float, t: float) -> None:
    h = solution.hamiltonian(hopping=t, interaction=u, bias=bias)
    other = reference.hamiltonian(hopping=t, interaction=u, bias=bias)
    np.testing.assert_allclose(h.numpy(), other, atol=1e-14)
    torch.testing.assert_close(h, h.mH)
    energies, vectors = torch.linalg.eigh(h)
    torch.testing.assert_close(h @ vectors, vectors * energies, atol=1e-13, rtol=0)
    np.testing.assert_allclose(energies.numpy(), np.linalg.eigvalsh(other), atol=1e-13)


def test_batched_dynamics_and_invariants() -> None:
    generator = torch.Generator().manual_seed(52)
    states = torch.randn(2, 3, 4, dtype=torch.complex128, generator=generator)
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    times = torch.tensor([0.0, 0.3, 1.7], dtype=torch.float64)
    h = solution.hamiltonian(interaction=3.1, bias=0.8)
    history = solution.evolve(h, states, times)
    assert history.shape == (3, 2, 3, 4)
    np.testing.assert_allclose(
        history.numpy(),
        reference.evolve(h.numpy(), states.numpy(), times.numpy()),
        atol=2e-13,
    )
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(
                history[:, i, j], solution.evolve(h, states[i, j], times)
            )
    torch.testing.assert_close(
        history.abs().square().sum(-1), torch.ones((3, 2, 3), dtype=torch.float64)
    )
    energy = torch.einsum("...i,ij,...j->...", history.conj(), h, history)
    torch.testing.assert_close(energy, energy[0].expand_as(energy))
    torch.testing.assert_close(
        solution.site_occupations(history).sum(-1),
        torch.full((3, 2, 3), 2.0, dtype=torch.float64),
    )
    np.testing.assert_allclose(
        solution.double_occupancy(history).numpy(),
        reference.double_occupancy(history.numpy()),
        atol=1e-13,
    )
    torch.testing.assert_close(solution.evolve(h, history[-1], -times[-1:])[0], states)


def test_exact_singlet_and_triplet() -> None:
    h = solution.hamiltonian(interaction=3)
    times = torch.linspace(0, 4, 21, dtype=torch.float64)
    singlet = torch.tensor([0, 1, -1, 0], dtype=torch.complex128) / sqrt(2)
    triplet = torch.tensor([0, 1, 1, 0], dtype=torch.complex128) / sqrt(2)
    history = solution.evolve(h, singlet, times)
    torch.testing.assert_close(
        solution.double_occupancy(history),
        16 / 25 * torch.sin(5 * times / 2).square(),
        atol=1e-13,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.evolve(h, triplet, times), triplet.expand(21, 4)
    )
    energies, vectors = torch.linalg.eigh(h)
    assert float(energies[0]) == pytest.approx(-1)
    assert float(solution.double_occupancy(vectors[:, 0])) == pytest.approx(0.2)


def test_invalid_inputs() -> None:
    for q in (-1, 4, True):
        with pytest.raises(ValueError, match="orbital"):
            solution.annihilation(q)
    with pytest.raises(ValueError, match="finite"):
        solution.hamiltonian(bias=float("inf"))
    for state in (torch.zeros(4), torch.ones(3), torch.full((4,), float("nan"))):
        with pytest.raises(ValueError, match="states must"):
            solution.double_occupancy(state)
    state = torch.tensor([1, 0, 0, 0], dtype=torch.complex128)
    h = solution.hamiltonian()
    for times in (torch.tensor([]), torch.tensor([1j]), torch.tensor([float("inf")])):
        with pytest.raises(ValueError, match="times must"):
            solution.evolve(h, state, times)
    h[0, 1] = 7
    with pytest.raises(ValueError, match="Hermitian"):
        solution.evolve(h, state, torch.tensor([0.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda() -> None:
    h = solution.hamiltonian(interaction=4, device="cuda")
    state = torch.tensor([1, 0, 0, 0], dtype=torch.complex128, device="cuda")
    times = torch.tensor([0.0, 1.0], dtype=torch.float64, device="cuda")
    history = solution.evolve(h, state, times)
    assert history.device.type == "cuda"
    torch.testing.assert_close(
        history.cpu(), solution.evolve(h.cpu(), state.cpu(), times.cpu())
    )
