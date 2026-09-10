"""Two distinguishable spin sites with isotropic exchange and local fields."""

from dataclasses import dataclass
from math import isfinite, pi

import torch
from torch import Tensor


@dataclass(frozen=True)
class CoupledEvolution:
    """History (times,batch,4), basis (++,+-,-+,--), first spin index first."""

    hamiltonians: Tensor
    times: Tensor
    states: Tensor


def pauli_matrices(device: str | torch.device = "cpu") -> Tensor:
    return torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
        device=device,
    )


def local_operators(device: str | torch.device = "cpu") -> tuple[Tensor, Tensor]:
    """Return sigma_a tensor I and I tensor sigma_a, each shape (3,4,4)."""
    identity = torch.eye(2, dtype=torch.complex128, device=device)
    pauli = pauli_matrices(device)
    first = torch.stack([torch.kron(component, identity) for component in pauli])
    second = torch.stack([torch.kron(identity, component) for component in pauli])
    return first, second


def normalize_states(states: Tensor) -> Tensor:
    if states.ndim < 1 or states.shape[-1] != 4 or states.numel() == 0:
        raise ValueError("joint states must have final dimension four and be nonempty")
    states = states.to(torch.complex128)
    norms: Tensor = torch.linalg.vector_norm(states, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("states must have finite nonzero norms")
    return states / norms[..., None]


def product_states(first: Tensor, second: Tensor) -> Tensor:
    """Pair spinors with identical leading shapes; flatten each 2x2 outer product."""
    if first.shape != second.shape or first.ndim < 1 or first.shape[-1] != 2:
        raise ValueError("spinors must share a shape ending in two")
    if first.device != second.device:
        raise ValueError("spinors must share a device")
    outer = (
        first.to(torch.complex128)[..., :, None]
        * second.to(torch.complex128)[..., None, :]
    )
    return normalize_states(outer.reshape(*first.shape[:-1], 4))


def hamiltonians(
    exchange: Tensor, fields: Tensor, gamma: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """J/4 sum_a sigma1_a sigma2_a - gamma*hbar/2 sum_s B_s.sigma_s.

    exchange has shape (batch,), fields (batch,2,3); J is an energy, not a frequency.
    Positive J favors the singlet at zero field.
    """
    if not isfinite(gamma) or not isfinite(hbar) or hbar <= 0:
        raise ValueError("gamma must be finite and hbar finite/positive")
    if (
        exchange.ndim != 1
        or exchange.numel() == 0
        or fields.shape != (exchange.numel(), 2, 3)
    ):
        raise ValueError("need exchange (batch,) and fields (batch,2,3)")
    if fields.device != exchange.device:
        raise ValueError("exchange and fields must share a device")
    for parameter in (exchange, fields):
        if torch.is_complex(parameter) or not bool(torch.isfinite(parameter).all()):
            raise ValueError("exchange and fields must be finite and real")
    first, second = local_operators(exchange.device)
    interaction = (first @ second).sum(0) / 4
    local = torch.einsum(
        "bsa,saij->bij", fields.to(torch.complex128), torch.stack([first, second])
    )
    return (
        exchange.to(torch.float64)[:, None, None] * interaction
        - gamma * hbar * local / 2
    )


def _check_problem(initial_states: Tensor, operators: Tensor) -> Tensor:
    if operators.ndim != 3 or operators.shape[1:] != (4, 4) or operators.shape[0] == 0:
        raise ValueError("Hamiltonians must have shape (batch,4,4)")
    if (
        initial_states.shape != (len(operators), 4)
        or initial_states.device != operators.device
    ):
        raise ValueError("initial states must have shape (batch,4) on the same device")
    if not bool(torch.isfinite(operators).all()) or not torch.allclose(
        operators, operators.mH, atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Hamiltonians must be finite and Hermitian")
    return normalize_states(initial_states)


def evolve_constant(
    initial_states: Tensor, operators: Tensor, times: Tensor, hbar: float = 1.0
) -> CoupledEvolution:
    """Native batched matrix exponential, with no integration time-step error."""
    initial = _check_problem(initial_states, operators)
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if (
        times.ndim != 1
        or times.numel() == 0
        or torch.is_complex(times)
        or not bool(torch.isfinite(times).all())
        or times.device != operators.device
    ):
        raise ValueError("times must be a finite real vector on the operators' device")
    time = times.to(torch.float64)
    matrix = operators.to(torch.complex128)
    propagators: Tensor = torch.linalg.matrix_exp(
        -1j * time[:, None, None, None] * matrix[None, :, :, :] / hbar
    )
    states = (propagators @ initial[None, :, :, None]).squeeze(-1)
    return CoupledEvolution(matrix, time, states)


def propagate_cn(
    initial_states: Tensor,
    operators: Tensor,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """Factor the constant 4x4 left matrices once and reuse batched LU solves."""
    state = _check_problem(initial_states, operators)
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    matrix = operators.to(torch.complex128)
    identity = torch.eye(4, dtype=torch.complex128, device=matrix.device)
    coefficient = 0.5j * time_step / hbar
    lu, pivots = torch.linalg.lu_factor(identity + coefficient * matrix)
    right = identity - coefficient * matrix
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        state = torch.linalg.lu_solve(lu, pivots, right @ state[:, :, None]).squeeze(-1)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    times = time_step * torch.tensor(
        saved_steps, dtype=torch.float64, device=matrix.device
    )
    return CoupledEvolution(matrix, times, torch.stack(saved))


def singlet_triplet_basis(device: str | torch.device = "cpu") -> Tensor:
    """Columns are t+, t0, t-, singlet in the ordered product basis."""
    basis = torch.eye(4, dtype=torch.complex128, device=device)
    return torch.stack(
        [
            basis[:, 0],
            (basis[:, 1] + basis[:, 2]) / 2**0.5,
            basis[:, 3],
            (basis[:, 1] - basis[:, 2]) / 2**0.5,
        ],
        dim=1,
    )


def exchange_reference(
    times: Tensor, exchange: float = 1.0, detuning: float = 0.0, hbar: float = 1.0
) -> Tensor:
    """Exact initial |+-> in longitudinal fields, shape (times,4).

    detuning=gamma*hbar*(B1z-B2z) is an energy. Common longitudinal field cancels
    in the zero-total-z subspace. Includes the global exp(+i*J*t/(4*hbar)) phase.
    """
    if not all(isfinite(v) for v in (exchange, detuning, hbar)) or hbar <= 0:
        raise ValueError("parameters must be finite and hbar positive")
    if (
        times.ndim != 1
        or torch.is_complex(times)
        or not bool(torch.isfinite(times).all())
    ):
        raise ValueError("times must be a finite real vector")
    time = times.to(torch.float64)
    splitting = (exchange**2 + detuning**2) ** 0.5
    angle = splitting * time / (2 * hbar)
    sine_over_splitting = time / (2 * hbar) * torch.sinc(angle / pi)
    stay = angle.cos() + 1j * detuning * sine_over_splitting
    swap = -1j * exchange * sine_over_splitting
    phase = torch.exp(1j * exchange * time / (4 * hbar))
    zero = torch.zeros_like(phase)
    return phase[:, None] * torch.stack([zero, stay, swap, zero], dim=1)


def joint_probabilities(states: Tensor) -> Tensor:
    """Probability weights in (++,+-,-+,--); sum equals the input norm squared."""
    return states.abs().square()


def local_bloch_vectors(states: Tensor) -> Tensor:
    """Return (...,2,3) Pauli expectations; spin means are hbar/2 times these."""
    first, second = local_operators(states.device)
    return torch.einsum(
        "...i,saij,...j->...sa", states.conj(), torch.stack([first, second]), states
    ).real


def correlations(states: Tensor, connected: bool = False) -> Tensor:
    """Return (...,3,3) <sigma1_a sigma2_b>, optionally subtracting local products."""
    first, second = local_operators(states.device)
    pair = first[:, None, :, :] @ second[None, :, :, :]
    values = torch.einsum("...i,abij,...j->...ab", states.conj(), pair, states).real
    if connected:
        local = local_bloch_vectors(states)
        values = values - local[..., 0, :, None] * local[..., 1, None, :]
    return values


def product_determinant(states: Tensor) -> Tensor:
    """|c++*c-- - c+-*c-+|; zero iff a nonzero pure two-spin state factors.

    For normalized states the range is [0,1/2]. This chapter uses the rank test;
    general mixed-state entanglement measures are deferred to later material.
    """
    return (states[..., 0] * states[..., 3] - states[..., 1] * states[..., 2]).abs()


def energy_expectations(result: CoupledEvolution) -> Tensor:
    return torch.einsum(
        "tbi,bij,tbj->tb", result.states.conj(), result.hamiltonians, result.states
    ).real


def phase_aligned_errors(states: Tensor, reference: Tensor) -> Tensor:
    overlap = (reference.conj() * states).sum(-1)
    phase = torch.exp(1j * torch.angle(overlap))
    errors: Tensor = torch.linalg.vector_norm(
        states - phase[..., None] * reference, dim=-1
    )
    return errors


def main() -> None:
    coupling = torch.tensor([1.0], dtype=torch.float64)
    fields = torch.zeros((1, 2, 3), dtype=torch.float64)
    initial = torch.tensor([[0, 1, 0, 0]], dtype=torch.complex128)
    times = torch.tensor([0, pi / 2, pi, 2 * pi], dtype=torch.float64)
    result = evolve_constant(initial, hamiltonians(coupling, fields), times)
    expected = exchange_reference(times)
    print("Two spins: J=hbar=1, zero field, initial |+z,-z>; basis (++,+-,-+,--)")
    print(
        f"Maximum state error to analytical exchange: {float(phase_aligned_errors(result.states[:, 0], expected).max()):.3e}"
    )
    for index, time in enumerate(times):
        print(
            f"t/pi={float(time) / pi:g}: P(-+)={float(joint_probabilities(result.states)[index, 0, 2]):.6f}, product determinant={float(product_determinant(result.states)[index, 0]):.6f}"
        )
    print(
        f"Maximum norm error: {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e}"
    )


if __name__ == "__main__":
    main()
