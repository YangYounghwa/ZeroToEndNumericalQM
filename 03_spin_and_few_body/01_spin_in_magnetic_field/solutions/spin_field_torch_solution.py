"""PyTorch spin-1/2 precession: H=-gamma*hbar*B.sigma/2."""

from dataclasses import dataclass
from math import isfinite, pi

import torch
from torch import Tensor


@dataclass(frozen=True)
class SpinEvolution:
    """States have shape (times,batch,2) in the (+z,-z) basis."""

    fields: Tensor
    hamiltonians: Tensor
    times: Tensor
    states: Tensor


def pauli_matrices(device: str | torch.device = "cpu") -> Tensor:
    """Return sigma_x, sigma_y, sigma_z, with shape (3,2,2)."""
    return torch.tensor(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]],
        dtype=torch.complex128,
        device=device,
    )


def spinor(
    theta: float = pi / 2, phi: float = 0.0, device: str | torch.device = "cpu"
) -> Tensor:
    """Pure state [cos(theta/2), exp(i*phi)*sin(theta/2)]; default is +x."""
    if not isfinite(theta) or not 0 <= theta <= pi or not isfinite(phi):
        raise ValueError("theta must be in [0,pi] and phi finite")
    angles = torch.tensor([theta, phi], dtype=torch.float64, device=device)
    return torch.stack(
        [
            torch.cos(angles[0] / 2).to(torch.complex128),
            torch.exp(1j * angles[1]) * torch.sin(angles[0] / 2),
        ]
    )


def normalize_states(states: Tensor) -> Tensor:
    """Normalize the final two-component axis, with no spatial integration weight."""
    if states.ndim < 1 or states.shape[-1] != 2 or states.numel() == 0:
        raise ValueError("states must have a final axis of length two and be nonempty")
    states = states.to(torch.complex128)
    norms: Tensor = torch.linalg.vector_norm(states, dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms == 0).any()):
        raise ValueError("states must have finite nonzero norms")
    return states / norms[..., None]


def hamiltonians(fields: Tensor, gamma: float = 1.0, hbar: float = 1.0) -> Tensor:
    """Construct a (batch,2,2) Hamiltonian from real fields of shape (batch,3)."""
    if not isfinite(gamma) or not isfinite(hbar) or hbar <= 0:
        raise ValueError("gamma must be finite and hbar finite/positive")
    if fields.ndim != 2 or fields.shape[1] != 3 or fields.shape[0] == 0:
        raise ValueError("fields must have shape (nonempty batch,3)")
    if torch.is_complex(fields) or not bool(torch.isfinite(fields).all()):
        raise ValueError("fields must be finite and real")
    return (
        -gamma
        * hbar
        / 2
        * torch.einsum(
            "ba,aij->bij", fields.to(torch.complex128), pauli_matrices(fields.device)
        )
    )


def _check_times(times: Tensor, fields: Tensor) -> None:
    if (
        times.ndim != 1
        or times.numel() == 0
        or torch.is_complex(times)
        or not bool(torch.isfinite(times).all())
    ):
        raise ValueError("times must be a nonempty finite real vector")
    if times.device != fields.device:
        raise ValueError("times and fields must share a device")


def matrix_propagators(
    fields: Tensor, times: Tensor, gamma: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """Evaluate exp(-iHt/hbar) independently at each time; no integration time step."""
    operators = hamiltonians(fields, gamma, hbar)
    _check_times(times, fields)
    propagators: Tensor = torch.linalg.matrix_exp(
        -1j
        * times.to(torch.float64)[:, None, None, None]
        * operators[None, :, :, :]
        / hbar
    )
    return propagators


def closed_form_propagators(
    fields: Tensor, times: Tensor, gamma: float = 1.0, hbar: float = 1.0
) -> Tensor:
    """Pauli formula with sinc, including exactly zero field without dividing by |B|."""
    operators = hamiltonians(fields, gamma, hbar)
    _check_times(times, fields)
    time = times.to(torch.float64)
    frequency: Tensor = abs(gamma) * torch.linalg.vector_norm(
        fields.to(torch.float64), dim=1
    )
    half_angle = time[:, None] * frequency[None, :] / 2
    identity = torch.eye(2, dtype=torch.complex128, device=fields.device)
    coefficient = time[:, None] * torch.sinc(half_angle / pi) / hbar
    return (
        half_angle.cos()[:, :, None, None] * identity
        - 1j * coefficient[:, :, None, None] * operators[None, :, :, :]
    )


def evolve_constant(
    initial_states: Tensor,
    fields: Tensor,
    times: Tensor,
    gamma: float = 1.0,
    hbar: float = 1.0,
) -> SpinEvolution:
    """Pair each initial state with one field; return all times using matrix_exp."""
    operators = hamiltonians(fields, gamma, hbar)
    if (
        initial_states.shape != (fields.shape[0], 2)
        or initial_states.device != fields.device
    ):
        raise ValueError("states must have shape (batch,2) on the fields' device")
    initial = normalize_states(initial_states)
    propagators = matrix_propagators(fields, times, gamma, hbar)
    states = (propagators @ initial[None, :, :, None]).squeeze(-1)
    return SpinEvolution(fields, operators, times.to(torch.float64), states)


def propagate_cn(
    initial_states: Tensor,
    fields: Tensor,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    gamma: float = 1.0,
    hbar: float = 1.0,
) -> SpinEvolution:
    """Batched CN with reused LU; normalize initially, never after time steps."""
    operators = hamiltonians(fields, gamma, hbar)
    if (
        initial_states.shape != (len(fields), 2)
        or initial_states.device != fields.device
    ):
        raise ValueError("states must have shape (batch,2) on the fields' device")
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    state = normalize_states(initial_states)
    identity = torch.eye(2, dtype=torch.complex128, device=fields.device)
    coefficient = 0.5j * time_step / hbar
    lu, pivots = torch.linalg.lu_factor(identity + coefficient * operators)
    right = identity - coefficient * operators
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        state = torch.linalg.lu_solve(lu, pivots, right @ state[:, :, None]).squeeze(-1)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    times = time_step * torch.tensor(
        saved_steps, dtype=torch.float64, device=fields.device
    )
    return SpinEvolution(fields, operators, times, torch.stack(saved))


def bloch_vectors(states: Tensor) -> Tensor:
    """Return <sigma> along the final axis; supplied states should be normalized."""
    return torch.einsum(
        "...i,aij,...j->...a", states.conj(), pauli_matrices(states.device), states
    ).real


def measurement_probabilities(states: Tensor, axis: Tensor) -> Tensor:
    """Return (+,-) measurement weights along a normalized real direction.

    Their sum equals the input norm squared, exposing drift instead of hiding it.
    For normalized states these are probabilities of outcomes +/-hbar/2.
    """
    if (
        axis.shape != (3,)
        or torch.is_complex(axis)
        or not bool(torch.isfinite(axis).all())
    ):
        raise ValueError(
            "measurement axis must be a finite real vector of length three"
        )
    if axis.device != states.device:
        raise ValueError("axis and states must share a device")
    norm = torch.linalg.vector_norm(axis.to(torch.float64))
    if float(norm) == 0:
        raise ValueError("measurement axis must be nonzero")
    projection = (bloch_vectors(states) * (axis / norm)).sum(-1)
    total = states.abs().square().sum(-1)
    return torch.stack([(total + projection) / 2, (total - projection) / 2], dim=-1)


def energy_expectations(result: SpinEvolution) -> Tensor:
    return torch.einsum(
        "tbi,bij,tbj->tb", result.states.conj(), result.hamiltonians, result.states
    ).real


def rodrigues_bloch(
    initial_bloch: Tensor, fields: Tensor, times: Tensor, gamma: float = 1.0
) -> Tensor:
    """Analytical r(t)=R(Omega*t)r(0), with Omega=-gamma*B, shape (times,batch,3)."""
    hamiltonians(fields, gamma)
    _check_times(times, fields)
    if initial_bloch.shape != fields.shape or initial_bloch.device != fields.device:
        raise ValueError("initial Bloch vectors must match the fields and device")
    omega = -gamma * fields.to(torch.float64)
    magnitude: Tensor = torch.linalg.vector_norm(omega, dim=1)
    denominator = torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))
    axis = omega / denominator[:, None]
    angle = times.to(torch.float64)[:, None, None] * magnitude[None, :, None]
    parallel = (axis * initial_bloch).sum(1)[:, None] * axis
    cross: Tensor = torch.linalg.cross(axis, initial_bloch, dim=1)
    return (
        angle.cos() * initial_bloch[None, :, :]
        + angle.sin() * cross[None, :, :]
        + (1 - angle.cos()) * parallel[None, :, :]
    )


def phase_aligned_errors(states: Tensor, reference: Tensor) -> Tensor:
    """State-vector errors after removing each state's unobservable global phase."""
    overlap = (reference.conj() * states).sum(-1)
    phase = torch.exp(1j * torch.angle(overlap))
    errors: Tensor = torch.linalg.vector_norm(
        states - phase[..., None] * reference, dim=-1
    )
    return errors


def main() -> None:
    fields = torch.tensor([[0, 0, 1], [1, 0, 0], [0.3, -0.4, 1]], dtype=torch.float64)
    initial = torch.stack([spinor(), spinor(0), spinor(pi / 3, pi / 5)])
    times = torch.linspace(0, 2 * pi, 101, dtype=torch.float64)
    result = evolve_constant(initial, fields, times)
    expected = rodrigues_bloch(bloch_vectors(initial), fields, times)
    energy = energy_expectations(result)
    print("H=-gamma*hbar*B.sigma/2; gamma=hbar=1; three independent field/state pairs")
    print(
        f"Maximum Bloch error to Rodrigues: {float((bloch_vectors(result.states) - expected).abs().max()):.3e}"
    )
    print(
        f"Maximum norm error: {float((result.states.abs().square().sum(-1) - 1).abs().max()):.3e}"
    )
    print(f"Maximum energy drift: {float((energy - energy[0]).abs().max()):.3e}")
    print(
        f"Final +z probabilities: {measurement_probabilities(result.states[-1], fields[0])[:, 0].tolist()}"
    )


if __name__ == "__main__":
    main()
