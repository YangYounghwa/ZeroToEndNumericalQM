"""Independent NumPy eigenbasis comparison for two coupled spin sites."""

from dataclasses import dataclass
from math import isfinite, pi

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class CoupledEvolution:
    hamiltonians: ComplexArray
    times: FloatArray
    states: ComplexArray


def pauli_matrices() -> ComplexArray:
    return np.array(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128
    )


def local_operators() -> tuple[ComplexArray, ComplexArray]:
    identity = np.eye(2, dtype=np.complex128)
    pauli = pauli_matrices()
    return np.stack([np.kron(component, identity) for component in pauli]), np.stack(
        [np.kron(identity, component) for component in pauli]
    )


def normalize_states(states: ComplexArray) -> ComplexArray:
    if states.ndim < 1 or states.shape[-1] != 4 or states.size == 0:
        raise ValueError("joint states must have final dimension four and be nonempty")
    norms = np.linalg.norm(states, axis=-1)
    if not np.isfinite(norms).all() or np.any(norms == 0):
        raise ValueError("states must have finite nonzero norms")
    return np.asarray(states / norms[..., None], dtype=np.complex128)


def product_states(first: ComplexArray, second: ComplexArray) -> ComplexArray:
    if first.shape != second.shape or first.ndim < 1 or first.shape[-1] != 2:
        raise ValueError("spinors must share a shape ending in two")
    outer = first[..., :, None] * second[..., None, :]
    return normalize_states(outer.reshape(*first.shape[:-1], 4))


def hamiltonians(
    exchange: FloatArray, fields: FloatArray, gamma: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    if not isfinite(gamma) or not isfinite(hbar) or hbar <= 0:
        raise ValueError("gamma must be finite and hbar finite/positive")
    if (
        exchange.ndim != 1
        or exchange.size == 0
        or fields.shape != (exchange.size, 2, 3)
    ):
        raise ValueError("need exchange (batch,) and fields (batch,2,3)")
    if any(np.iscomplexobj(p) or not np.isfinite(p).all() for p in (exchange, fields)):
        raise ValueError("exchange and fields must be finite and real")
    first, second = local_operators()
    interaction = (first @ second).sum(0) / 4
    local = np.einsum("bsa,saij->bij", fields, np.stack([first, second]))
    return np.asarray(
        exchange[:, None, None] * interaction - gamma * hbar * local / 2,
        dtype=np.complex128,
    )


def _check_problem(
    initial_states: ComplexArray, operators: ComplexArray
) -> ComplexArray:
    if operators.ndim != 3 or operators.shape[1:] != (4, 4) or operators.shape[0] == 0:
        raise ValueError("Hamiltonians must have shape (batch,4,4)")
    if initial_states.shape != (len(operators), 4):
        raise ValueError("initial states must have shape (batch,4)")
    if not np.isfinite(operators).all() or not np.allclose(
        operators, operators.conj().swapaxes(-1, -2), atol=1e-12, rtol=1e-12
    ):
        raise ValueError("Hamiltonians must be finite and Hermitian")
    return normalize_states(initial_states)


def evolve_constant(
    initial_states: ComplexArray,
    operators: ComplexArray,
    times: FloatArray,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """Reconstruct all propagators from each Hermitian eigenbasis."""
    initial = _check_problem(initial_states, operators)
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if (
        times.ndim != 1
        or times.size == 0
        or np.iscomplexobj(times)
        or not np.isfinite(times).all()
    ):
        raise ValueError("times must be a nonempty finite real vector")
    energies, vectors = np.linalg.eigh(operators)
    phases = np.exp(-1j * times[:, None, None] * energies[None, :, :] / hbar)
    propagators = (
        vectors[None, :, :, :] * phases[:, :, None, :]
    ) @ vectors.conj().swapaxes(-1, -2)[None, :, :, :]
    states = (propagators @ initial[None, :, :, None]).squeeze(-1)
    return CoupledEvolution(operators, times, states)


def propagate_cn(
    initial_states: ComplexArray,
    operators: ComplexArray,
    time_step: float = 0.1,
    num_steps: int = 100,
    store_every: int = 1,
    hbar: float = 1.0,
) -> CoupledEvolution:
    """Solve for the tiny 4x4 CN step once; never form an explicit inverse."""
    state = _check_problem(initial_states, operators)
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if not isfinite(time_step) or time_step == 0 or num_steps < 1 or store_every < 1:
        raise ValueError("time step must be finite/nonzero and step counts positive")
    identity = np.eye(4, dtype=np.complex128)
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
    return CoupledEvolution(
        operators,
        time_step * np.asarray(saved_steps, dtype=np.float64),
        np.stack(saved),
    )


def local_bloch_vectors(states: ComplexArray) -> FloatArray:
    first, second = local_operators()
    return np.asarray(
        np.einsum(
            "...i,saij,...j->...sa", states.conj(), np.stack([first, second]), states
        ).real,
        dtype=np.float64,
    )


def correlations(states: ComplexArray, connected: bool = False) -> FloatArray:
    first, second = local_operators()
    pair = first[:, None, :, :] @ second[None, :, :, :]
    values: FloatArray = np.einsum(
        "...i,abij,...j->...ab", states.conj(), pair, states
    ).real
    if connected:
        local = local_bloch_vectors(states)
        values = values - local[..., 0, :, None] * local[..., 1, None, :]
    return values


def product_determinant(states: ComplexArray) -> FloatArray:
    return np.asarray(
        np.abs(states[..., 0] * states[..., 3] - states[..., 1] * states[..., 2]),
        dtype=np.float64,
    )


def energy_expectations(result: CoupledEvolution) -> FloatArray:
    return np.asarray(
        np.einsum(
            "tbi,bij,tbj->tb", result.states.conj(), result.hamiltonians, result.states
        ).real,
        dtype=np.float64,
    )


def main() -> None:
    operators = hamiltonians(np.array([1.0]), np.zeros((1, 2, 3)))
    initial = np.array([[0, 1, 0, 0]], dtype=np.complex128)
    result = evolve_constant(initial, operators, np.array([0, pi / 2, pi, 2 * pi]))
    print("NumPy exchange dynamics, J=hbar=1, initial |+z,-z>")
    print(f"Swap probabilities: {(np.abs(result.states[:, 0, 2]) ** 2).tolist()}")
    print(f"Product determinants: {product_determinant(result.states[:, 0]).tolist()}")


if __name__ == "__main__":
    main()
