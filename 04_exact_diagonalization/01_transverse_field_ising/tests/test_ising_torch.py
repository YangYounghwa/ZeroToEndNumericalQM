from math import pi, sin, sqrt

import ising_numpy_solution as reference
import ising_torch_solution as solution
import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    ("sites", "coupling", "field"),
    [(1, 1.0, 0.7), (2, 1.0, 1.0), (3, -0.8, 0.4), (5, 1.2, -0.6)],
)
def test_independent_matrix(sites: int, coupling: float, field: float) -> None:
    h = solution.hamiltonian(sites, coupling=coupling, field=field)
    np.testing.assert_allclose(
        h.numpy(),
        reference.hamiltonian(sites, coupling=coupling, field=field),
        atol=1e-14,
    )
    torch.testing.assert_close(h, h.mH)
    assert h.dtype == torch.float64
    assert float(h.trace()) == pytest.approx(0)


@pytest.mark.parametrize(("coupling", "field"), [(1.0, 0.0), (0.0, 2.0), (-0.7, 1.3)])
def test_two_site_spectrum(coupling: float, field: float) -> None:
    root = sqrt(coupling**2 + 4 * field**2)
    expected = torch.tensor(
        sorted([-root, -coupling, coupling, root]), dtype=torch.float64
    )
    torch.testing.assert_close(
        torch.linalg.eigvalsh(solution.hamiltonian(2, coupling=coupling, field=field)),
        expected,
        atol=1e-13,
        rtol=0,
    )


@pytest.mark.parametrize("sites", [1, 2, 4, 6])
def test_critical_formula_and_parity(sites: int) -> None:
    h = solution.hamiltonian(sites)
    energies, vectors = torch.linalg.eigh(h)
    angle = pi / (4 * sites + 2)
    assert float(energies[0]) == pytest.approx(1 - 1 / sin(angle), abs=2e-13)
    assert float(energies[1] - energies[0]) == pytest.approx(4 * sin(angle), abs=2e-13)
    torch.testing.assert_close(h @ vectors, vectors * energies, atol=2e-13, rtol=0)
    assert float(solution.parity_expectation(vectors[:, 0], sites)) == pytest.approx(1)
    assert abs(float(solution.magnetization(vectors[:, 0], sites, "z"))) < 1e-12
    blocks = []
    for parity in (-1, 1):
        q = solution.parity_embedding(sites, parity)
        torch.testing.assert_close(
            q.T @ q, torch.eye(2 ** (sites - 1), dtype=torch.float64)
        )
        torch.testing.assert_close(q.flip(0), parity * q)
        block = q.T @ h @ q
        torch.testing.assert_close(h @ q, q @ block, atol=1e-13, rtol=0)
        blocks.append(torch.linalg.eigvalsh(block))
    torch.testing.assert_close(
        torch.cat(blocks).sort().values, energies, atol=2e-13, rtol=0
    )


def test_complex_batch_observables() -> None:
    sites = 4
    states = torch.randn(
        2, 3, 16, dtype=torch.complex128, generator=torch.Generator().manual_seed(37)
    )
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    for axis in ("x", "z"):
        local = [
            reference.expectation(
                states.numpy(), reference.local_operator(sites, i, axis)
            )
            for i in range(sites)
        ]
        np.testing.assert_allclose(
            solution.magnetization(states, sites, axis).numpy(),
            np.mean(local, axis=0),
            atol=1e-14,
        )
    for left, right in ((0, 3), (2, 2), (1, 2)):
        operator = reference.local_operator(
            sites, left, "z"
        ) @ reference.local_operator(sites, right, "z")
        np.testing.assert_allclose(
            solution.zz_correlation(states, sites, left, right).numpy(),
            reference.expectation(states.numpy(), operator),
            atol=1e-14,
        )
    for i in range(2):
        for j in range(3):
            torch.testing.assert_close(
                solution.magnetization(states, sites, "x")[i, j],
                solution.magnetization(states[i, j], sites, "x"),
            )


def test_product_and_cat_states() -> None:
    sites = 4
    product = torch.zeros(16, dtype=torch.complex128)
    product[0] = 1
    cat = (product + product.flip(0)) / sqrt(2)
    uniform = torch.ones(16, dtype=torch.complex128) / 4
    assert float(solution.magnetization(product, sites, "z")) == pytest.approx(1)
    assert float(solution.magnetization(cat, sites, "z")) == pytest.approx(0)
    assert float(solution.zz_correlation(cat, sites, 0, 3)) == pytest.approx(1)
    assert float(solution.magnetization(uniform, sites, "x")) == pytest.approx(1)
    assert float(solution.zz_correlation(uniform, sites, 0, 3)) == pytest.approx(0)


def test_invalid_inputs() -> None:
    for sites in (0, 11, True):
        with pytest.raises(ValueError, match="num_sites"):
            solution.hamiltonian(sites)
    with pytest.raises(ValueError, match="finite"):
        solution.hamiltonian(2, field=float("nan"))
    with pytest.raises(ValueError, match="parity"):
        solution.parity_embedding(2, 0)
    for states in (torch.zeros(4), torch.ones(3), torch.full((4,), float("nan"))):
        with pytest.raises(ValueError, match="states"):
            solution.magnetization(states, 2, "x")
    state = torch.ones(4) / 2
    with pytest.raises(ValueError, match="axis"):
        solution.magnetization(state, 2, "y")
    with pytest.raises(ValueError, match="indices"):
        solution.zz_correlation(state, 2, 0, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda() -> None:
    h = solution.hamiltonian(4, device="cuda")
    energies, vectors = torch.linalg.eigh(h)
    torch.testing.assert_close(energies.cpu(), torch.linalg.eigvalsh(h.cpu()))
    ground = vectors[:, 0]
    assert solution.magnetization(ground, 4, "x").device.type == "cuda"
    torch.testing.assert_close(
        solution.parity_embedding(4, 1, device="cuda").cpu(),
        solution.parity_embedding(4, 1),
    )
