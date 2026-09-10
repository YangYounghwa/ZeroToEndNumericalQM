"""NumPy spin-precession comparison using Hermitian eigendecomposition."""

from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SpinEvolution:
    fields: FloatArray
    hamiltonians: ComplexArray
    times: FloatArray
    states: ComplexArray


def pauli_matrices() -> ComplexArray:
    return np.array(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128
    )


def spinor(theta: float = pi / 2, phi: float = 0.0) -> ComplexArray:
    if not isfinite(theta) or not 0 <= theta <= pi or not isfinite(phi):
        raise ValueError("theta must be in [0,pi] and phi finite")
    return np.array(
        [np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=np.complex128
    )


def normalize_states(states: ComplexArray) -> ComplexArray:
    if states.ndim < 1 or states.shape[-1] != 2 or states.size == 0:
        raise ValueError("states must have a final axis of length two and be nonempty")
    norms = np.linalg.norm(states, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("states must have finite nonzero norms")
    return np.asarray(states / norms[..., None], dtype=np.complex128)


def hamiltonians(
    fields: FloatArray, gamma: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    if not isfinite(gamma) or not isfinite(hbar) or hbar <= 0:
        raise ValueError("gamma must be finite and hbar finite/positive")
    if fields.ndim != 2 or fields.shape[1] != 3 or fields.shape[0] == 0:
        raise ValueError("fields must have shape (nonempty batch,3)")
    if np.iscomplexobj(fields) or not np.isfinite(fields).all():
        raise ValueError("fields must be finite and real")
    return np.asarray(
        -gamma * hbar / 2 * np.einsum("ba,aij->bij", fields, pauli_matrices()),
        dtype=np.complex128,
    )


def eigen_propagators(
    fields: FloatArray, times: FloatArray, gamma: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    """Build U(t)=V diag(exp(-iEt/hbar)) V.H independently for each requested time."""
    operators = hamiltonians(fields, gamma, hbar)
    if (
        times.ndim != 1
        or times.size == 0
        or np.iscomplexobj(times)
        or not np.isfinite(times).all()
    ):
        raise ValueError("times must be a nonempty finite real vector")
    energies, vectors = np.linalg.eigh(operators)
    phases = np.exp(-1j * times[:, None, None] * energies[None, :, :] / hbar)
    return np.asarray(
        (vectors[None, :, :, :] * phases[:, :, None, :])
        @ vectors.conj().swapaxes(-1, -2)[None, :, :, :],
        dtype=np.complex128,
    )


def evolve_constant(
    initial_states: ComplexArray,
    fields: FloatArray,
    times: FloatArray,
    gamma: float = 1.0,
    hbar: float = 1.0,
) -> SpinEvolution:
    operators = hamiltonians(fields, gamma, hbar)
    if initial_states.shape != (len(fields), 2):
        raise ValueError("states must have shape (batch,2)")
    initial = normalize_states(initial_states)
    propagators = eigen_propagators(fields, times, gamma, hbar)
    states = (propagators @ initial[None, :, :, None]).squeeze(-1)
    return SpinEvolution(fields, operators, times, states)


def propagate_cn(
    initial_states: ComplexArray,
    fields: FloatArray,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    gamma: float = 1.0,
    hbar: float = 1.0,
) -> SpinEvolution:
    """For these tiny matrices, solve once for each 2x2 CN step operator."""
    operators = hamiltonians(fields, gamma, hbar)
    if initial_states.shape != (len(fields), 2):
        raise ValueError("states must have shape (batch,2)")
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    state = normalize_states(initial_states)
    identity = np.eye(2, dtype=np.complex128)
    coefficient = 0.5j * time_step / hbar
    step_operator = np.linalg.solve(
        identity + coefficient * operators, identity - coefficient * operators
    )
    saved, saved_steps = [state], [0]
    for step in range(1, num_steps + 1):
        state = (step_operator @ state[:, :, None]).squeeze(-1)
        if step % store_every == 0 or step == num_steps:
            saved.append(state)
            saved_steps.append(step)
    return SpinEvolution(
        fields,
        operators,
        time_step * np.asarray(saved_steps, dtype=np.float64),
        np.stack(saved),
    )


def bloch_vectors(states: ComplexArray) -> FloatArray:
    return np.asarray(
        np.einsum("...i,aij,...j->...a", states.conj(), pauli_matrices(), states).real,
        dtype=np.float64,
    )


def measurement_probabilities(states: ComplexArray, axis: FloatArray) -> FloatArray:
    if axis.shape != (3,) or np.iscomplexobj(axis) or not np.isfinite(axis).all():
        raise ValueError(
            "measurement axis must be a finite real vector of length three"
        )
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("measurement axis must be nonzero")
    projection = (bloch_vectors(states) * (axis / norm)).sum(-1)
    total = (np.abs(states) ** 2).sum(-1)
    return np.stack([(total + projection) / 2, (total - projection) / 2], axis=-1)


def energy_expectations(result: SpinEvolution) -> FloatArray:
    return np.asarray(
        np.einsum(
            "tbi,bij,tbj->tb", result.states.conj(), result.hamiltonians, result.states
        ).real,
        dtype=np.float64,
    )


def main() -> None:
    fields = np.array([[0, 0, 1], [1, 0, 0], [0.3, -0.4, 1]], dtype=np.float64)
    initial = np.stack([spinor(), spinor(0), spinor(pi / 3, pi / 5)])
    times = np.linspace(0, 2 * pi, 101)
    result = evolve_constant(initial, fields, times)
    energy = energy_expectations(result)
    print("NumPy eigenbasis evolution, gamma=hbar=1")
    print(
        f"Maximum norm error: {np.max(np.abs(np.sum(np.abs(result.states) ** 2, axis=-1) - 1)):.3e}"
    )
    print(f"Maximum energy drift: {np.max(np.abs(energy - energy[0])):.3e}")
    print(
        f"Final +z probabilities: {measurement_probabilities(result.states[-1], fields[0])[:, 0].tolist()}"
    )


if __name__ == "__main__":
    main()
