"""NumPy comparison; partial traces use explicit subsystem index loops."""

from math import isfinite, sqrt

import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def pure_density(states: ComplexArray) -> ComplexArray:
    """Project normalized (..., D) state vectors into density matrices."""
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim < 1 or states.size == 0 or not np.isfinite(states).all():
        raise ValueError("states must contain finite, nonempty state vectors")
    if not np.allclose(np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("states must be normalized")
    return states[..., :, None] * states.conj()[..., None, :]


def validate_density(rho: ComplexArray, *, tolerance: float = 1e-12) -> None:
    """Validate shape, finiteness, Hermiticity, trace, and positivity."""
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if rho.ndim < 2 or rho.shape[-1] != rho.shape[-2] or rho.size == 0:
        raise ValueError("rho must have nonempty shape (..., D, D)")
    if not np.isfinite(rho).all():
        raise ValueError("rho must be finite")
    if not np.allclose(rho, rho.conj().swapaxes(-2, -1), atol=tolerance, rtol=0):
        raise ValueError("rho must be Hermitian")
    if not np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1, atol=tolerance, rtol=0):
        raise ValueError("rho must have unit trace")
    if np.any(np.linalg.eigvalsh(rho) < -tolerance):
        raise ValueError("rho must be positive semidefinite")


def mixed_density(states: ComplexArray, probabilities: RealArray) -> ComplexArray:
    """Average (..., K, D) state projectors with (..., K) probabilities."""
    if states.ndim < 2 or probabilities.shape != states.shape[:-1]:
        raise ValueError("probabilities must match the ensemble axes (..., K)")
    if (
        np.iscomplexobj(probabilities)
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
    ):
        raise ValueError("probabilities must be real, finite, and nonnegative")
    if not np.allclose(probabilities.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("probabilities must sum to one")
    projectors = pure_density(states)
    return np.sum(probabilities[..., None, None] * projectors, axis=-3)


def partial_trace(
    rho: ComplexArray, dims: tuple[int, int] = (2, 2), *, keep: int = 0
) -> ComplexArray:
    """Reference implementation in the a*dB+b product-basis convention."""
    validate_density(rho)
    if (
        len(dims) != 2
        or any(d <= 0 for d in dims)
        or dims[0] * dims[1] != rho.shape[-1]
    ):
        raise ValueError("positive subsystem dimensions must multiply to D")
    if keep not in (0, 1):
        raise ValueError("keep must be 0 or 1")
    da, db = dims
    size = dims[keep]
    result = np.zeros((*rho.shape[:-2], size, size), dtype=np.complex128)
    for row in range(size):
        for column in range(size):
            if keep == 0:
                for b in range(db):
                    result[..., row, column] += rho[..., row * db + b, column * db + b]
            else:
                for a in range(da):
                    result[..., row, column] += rho[..., a * db + row, a * db + column]
    return result


def purity(rho: ComplexArray) -> RealArray:
    validate_density(rho)
    return np.asarray(np.trace(rho @ rho, axis1=-2, axis2=-1).real, dtype=np.float64)


def entropy(rho: ComplexArray) -> RealArray:
    """Entropy in bits, with zero log zero defined by its continuous limit."""
    validate_density(rho)
    values = np.maximum(np.linalg.eigvalsh(rho), 0)
    safe = np.where(values > 0, values, 1)
    return np.asarray(-np.sum(values * np.log2(safe), axis=-1), dtype=np.float64)


def entanglement_entropy(
    states: ComplexArray, dims: tuple[int, int] = (2, 2)
) -> RealArray:
    return entropy(partial_trace(pure_density(states), dims, keep=0))


def bell_states() -> ComplexArray:
    """Rows: Phi+, Phi-, Psi+, Psi-."""
    return np.array(
        [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]], dtype=np.complex128
    ) / sqrt(2)


def schmidt_states(angles: RealArray, *, phase: float = 0.0) -> ComplexArray:
    if np.iscomplexobj(angles) or not np.isfinite(angles).all() or not isfinite(phase):
        raise ValueError("angles and phase must be finite and real")
    theta = np.asarray(angles, dtype=np.float64)
    zero = np.zeros_like(theta)
    return np.asarray(
        np.stack(
            (np.cos(theta), zero, zero, np.exp(1j * phase) * np.sin(theta)), axis=-1
        ),
        dtype=np.complex128,
    )


def exchange_states(
    times: RealArray, *, exchange: float = 1.0, hbar: float = 1.0
) -> ComplexArray:
    if (
        np.iscomplexobj(times)
        or not np.isfinite(times).all()
        or not isfinite(exchange)
        or not isfinite(hbar)
        or hbar <= 0
    ):
        raise ValueError("times and exchange must be finite; hbar must be positive")
    angle = exchange * np.asarray(times, dtype=np.float64) / (2 * hbar)
    zero = np.zeros_like(angle)
    return np.asarray(
        np.stack((zero, np.cos(angle), -1j * np.sin(angle), zero), axis=-1),
        dtype=np.complex128,
    )


def pauli_correlations(rho: ComplexArray) -> RealArray:
    validate_density(rho)
    if rho.shape[-1] != 4:
        raise ValueError("Pauli correlations require two qubits")
    sigma = np.array(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=np.complex128
    )
    result = np.empty((*rho.shape[:-2], 3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            result[..., a, b] = np.trace(
                rho @ np.kron(sigma[a], sigma[b]), axis1=-2, axis2=-1
            ).real
    return result


def main() -> None:
    rho = pure_density(bell_states())
    print("Bell order: Phi+, Phi-, Psi+, Psi-")
    print("Joint purity:", purity(rho).tolist())
    print("Local entropy (bits):", entropy(partial_trace(rho)).tolist())


if __name__ == "__main__":
    main()
