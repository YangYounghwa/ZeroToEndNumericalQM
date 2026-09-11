from math import log2, pi

import bell_states_numpy_solution as reference
import bell_states_torch_solution as solution
import numpy as np
import pytest
import torch
from torch import Tensor


def test_bell_basis_projectors_and_correlations() -> None:
    states = solution.bell_states()
    rho = solution.pure_density(states)
    torch.testing.assert_close(states @ states.mH, torch.eye(4, dtype=torch.complex128))
    torch.testing.assert_close(rho @ rho, rho)
    torch.testing.assert_close(solution.purity(rho), torch.ones(4, dtype=torch.float64))
    torch.testing.assert_close(
        solution.entropy(rho), torch.zeros(4, dtype=torch.float64), atol=1e-13, rtol=0
    )
    for keep in (0, 1):
        local = solution.partial_trace(rho, keep=keep)
        torch.testing.assert_close(
            local, torch.eye(2, dtype=torch.complex128).expand(4, 2, 2) / 2
        )
        torch.testing.assert_close(
            solution.entropy(local), torch.ones(4, dtype=torch.float64)
        )
    expected = torch.diag_embed(
        torch.tensor(
            [[1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1]], dtype=torch.float64
        )
    )
    torch.testing.assert_close(
        solution.pauli_correlations(rho), expected, atol=1e-14, rtol=0
    )


def test_classical_mixture_has_same_marginals_but_different_joint_state() -> None:
    pure = solution.pure_density(solution.bell_states()[0])
    mixture = solution.mixed_density(
        torch.eye(4, dtype=torch.complex128)[[0, 3]],
        torch.tensor([0.5, 0.5], dtype=torch.float64),
    )
    for keep in (0, 1):
        torch.testing.assert_close(
            solution.partial_trace(pure, keep=keep),
            solution.partial_trace(mixture, keep=keep),
        )
    assert float(solution.purity(mixture)) == pytest.approx(0.5)
    assert float(solution.entropy(mixture)) == pytest.approx(1)
    assert float(solution.pauli_correlations(mixture)[2, 2]) == pytest.approx(1)
    assert float(solution.pauli_correlations(mixture)[0, 0]) == pytest.approx(0)
    assert float(solution.pauli_correlations(pure)[0, 0]) == pytest.approx(1)
    torch.testing.assert_close(pure.diagonal(), mixture.diagonal())


def test_complex_product_state_and_asymmetric_partial_trace() -> None:
    a = torch.tensor([1, 1j], dtype=torch.complex128) / 2**0.5
    b = torch.tensor([1, 2j, -2], dtype=torch.complex128) / 3
    state = torch.kron(a, b)
    rho = solution.pure_density(state)
    torch.testing.assert_close(
        solution.partial_trace(rho, (2, 3), keep=0), solution.pure_density(a)
    )
    torch.testing.assert_close(
        solution.partial_trace(rho, (2, 3), keep=1), solution.pure_density(b)
    )
    assert abs(float(solution.entanglement_entropy(state, (2, 3)))) < 1e-13


def test_entangled_asymmetric_trace_matches_schmidt_spectrum() -> None:
    state = torch.tensor([0, 0.5, 0, 0, 0, (0.75) ** 0.5 * 1j], dtype=torch.complex128)
    rho = solution.pure_density(state)
    expected_a = torch.diag(torch.tensor([0.25, 0.75], dtype=torch.complex128))
    expected_b = torch.diag(torch.tensor([0, 0.25, 0.75], dtype=torch.complex128))
    torch.testing.assert_close(solution.partial_trace(rho, (2, 3)), expected_a)
    torch.testing.assert_close(solution.partial_trace(rho, (2, 3), keep=1), expected_b)


def test_multiple_batch_axes_and_numpy_comparison() -> None:
    generator = torch.Generator().manual_seed(718)
    states = torch.randn(2, 3, 5, 6, generator=generator, dtype=torch.complex128)
    states /= torch.linalg.vector_norm(states, dim=-1, keepdim=True)
    weights = torch.rand(2, 3, 5, generator=generator, dtype=torch.float64)
    weights /= weights.sum(-1, keepdim=True)
    rho = solution.mixed_density(states, weights)
    other = reference.mixed_density(states.numpy(), weights.numpy())
    np.testing.assert_allclose(rho.numpy(), other, atol=1e-14)
    for keep in (0, 1):
        local = solution.partial_trace(rho, (2, 3), keep=keep)
        expected = reference.partial_trace(other, (2, 3), keep=keep)
        np.testing.assert_allclose(local.numpy(), expected, atol=1e-14)
        np.testing.assert_allclose(
            solution.entropy(local).numpy(), reference.entropy(expected), atol=1e-13
        )
        # Marginal probabilities also follow directly from the joint diagonal.
        probabilities = rho.diagonal(dim1=-2, dim2=-1).real.reshape(2, 3, 2, 3)
        marginal = probabilities.sum(-1 if keep == 0 else -2)
        torch.testing.assert_close(local.diagonal(dim1=-2, dim2=-1).real, marginal)


def test_schmidt_family_phase_and_entropy() -> None:
    angles = torch.linspace(0, pi / 2, 31, dtype=torch.float64)
    states = solution.schmidt_states(angles, phase=0.73)
    result = solution.entanglement_entropy(states)
    probability = torch.sin(angles).square()
    expected = torch.tensor(
        [
            -sum(p * log2(p) for p in (float(q), 1 - float(q)) if p > 0)
            for q in probability
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(result, expected, atol=1e-13, rtol=0)
    torch.testing.assert_close(
        result,
        solution.entanglement_entropy(solution.schmidt_states(angles)),
        atol=1e-13,
        rtol=0,
    )
    torch.testing.assert_close(
        solution.purity(solution.partial_trace(solution.pure_density(states))),
        1 - 2 * probability * (1 - probability),
    )
    np.testing.assert_allclose(
        states.numpy(), reference.schmidt_states(angles.numpy(), phase=0.73), atol=1e-14
    )


def test_exchange_entropy_against_hamiltonian_evolution() -> None:
    times = torch.linspace(0, 2 * pi, 25, dtype=torch.float64)
    # H = J/4 sigma.sigma for J=hbar=1, written independently in product basis.
    matrix = torch.tensor(
        [[0.25, 0, 0, 0], [0, -0.25, 0.5, 0], [0, 0.5, -0.25, 0], [0, 0, 0, 0.25]],
        dtype=torch.complex128,
    )
    initial = torch.tensor([0, 1, 0, 0], dtype=torch.complex128)
    evolved = torch.linalg.matrix_exp(-1j * times[:, None, None] * matrix) @ initial
    exact = solution.exchange_states(times)
    torch.testing.assert_close(
        solution.pure_density(evolved), solution.pure_density(exact), atol=1e-13, rtol=0
    )
    landmarks = solution.entanglement_entropy(
        solution.exchange_states(
            torch.tensor([0, pi / 2, pi, 3 * pi / 2, 2 * pi], dtype=torch.float64)
        )
    )
    torch.testing.assert_close(
        landmarks,
        torch.tensor([0, 1, 0, 1, 0], dtype=torch.float64),
        atol=1e-13,
        rtol=0,
    )
    np.testing.assert_allclose(
        exact.numpy(), reference.exchange_states(times.numpy()), atol=1e-14
    )


def test_local_unitaries_preserve_entanglement_and_remote_reduction() -> None:
    state = solution.schmidt_states(torch.tensor(0.31, dtype=torch.float64), phase=0.8)
    rho = solution.pure_density(state)
    x = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128)
    ua = torch.linalg.matrix_exp(-0.37j * x)
    ub = torch.linalg.matrix_exp(0.52j * y)
    joint = torch.kron(ua, ub)
    evolved = joint @ rho @ joint.mH
    torch.testing.assert_close(
        evolved, solution.pure_density(joint @ state), atol=1e-14, rtol=0
    )
    torch.testing.assert_close(
        solution.entropy(solution.partial_trace(evolved)),
        solution.entanglement_entropy(state),
    )
    remote = torch.kron(torch.eye(2, dtype=torch.complex128), ub)
    torch.testing.assert_close(
        solution.partial_trace(remote @ rho @ remote.mH), solution.partial_trace(rho)
    )
    local = solution.partial_trace(evolved)
    torch.testing.assert_close(local, ua @ solution.partial_trace(rho) @ ua.mH)
    # Local expectations computed before and after tracing must agree.
    full_mean = torch.trace(
        evolved @ torch.kron(x, torch.eye(2, dtype=torch.complex128))
    )
    torch.testing.assert_close(full_mean, torch.trace(local @ x))


def test_entropy_zero_and_small_positive_eigenvalues() -> None:
    assert (
        float(
            solution.entropy(torch.diag(torch.tensor([1, 0], dtype=torch.complex128)))
        )
        == 0
    )
    for small in (1e-4, 1e-8, 1e-14):
        rho = torch.diag(torch.tensor([1 - small, small], dtype=torch.complex128))
        expected = -(1 - small) * log2(1 - small) - small * log2(small)
        assert float(solution.entropy(rho)) == pytest.approx(
            expected, rel=1e-12, abs=1e-18
        )
    roundoff = torch.diag(torch.tensor([1 + 1e-14, -1e-14], dtype=torch.complex128))
    assert abs(float(solution.entropy(roundoff))) < 1e-12


@pytest.mark.parametrize(
    "rho",
    [
        torch.zeros(4, dtype=torch.complex128),
        torch.eye(2, dtype=torch.complex128),
        torch.tensor([[1, 1j], [0, 0]], dtype=torch.complex128),
        torch.diag(torch.tensor([1.1, -0.1], dtype=torch.complex128)),
        torch.tensor([[float("nan"), 0], [0, 1]], dtype=torch.complex128),
    ],
)
def test_invalid_density_is_rejected(rho: Tensor) -> None:
    with pytest.raises(ValueError, match="rho must"):
        solution.entropy(rho)


def test_invalid_state_ensemble_and_trace_arguments() -> None:
    states = solution.bell_states()
    with pytest.raises(ValueError, match="normalized"):
        solution.pure_density(states * 2)
    negative = torch.tensor([1.1, -0.1], dtype=torch.float64)
    with pytest.raises(ValueError, match="nonnegative"):
        solution.mixed_density(states[:2], negative)
    wrong_total = torch.tensor([0.2, 0.2], dtype=torch.float64)
    with pytest.raises(ValueError, match="sum to one"):
        solution.mixed_density(states[:2], wrong_total)
    rho = solution.pure_density(states[0])
    with pytest.raises(ValueError, match="dimensions"):
        solution.partial_trace(rho, (2, 3))
    with pytest.raises(ValueError, match="keep"):
        solution.partial_trace(rho, keep=2)
    with pytest.raises(ValueError, match="tolerance"):
        solution.validate_density(rho, tolerance=float("nan"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_matches_cpu() -> None:
    angles = torch.linspace(0, pi / 2, 17, dtype=torch.float64)
    states = solution.schmidt_states(angles.to("cuda"), phase=0.4)
    result = solution.entanglement_entropy(states)
    assert result.device.type == "cuda"
    torch.testing.assert_close(
        result.cpu(),
        solution.entanglement_entropy(solution.schmidt_states(angles, phase=0.4)),
        atol=1e-13,
        rtol=0,
    )
    correlations = solution.pauli_correlations(
        solution.pure_density(solution.bell_states(device="cuda"))
    )
    torch.testing.assert_close(
        correlations.cpu(),
        solution.pauli_correlations(solution.pure_density(solution.bell_states())),
    )
