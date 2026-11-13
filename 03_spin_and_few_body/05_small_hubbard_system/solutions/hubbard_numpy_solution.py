"""Independent explicit matrix and spectral evolution for the Hubbard dimer."""

from math import isfinite

import numpy as np
from numpy.typing import NDArray

type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def hamiltonian(
    *,
    hopping: float = 1.0,
    interaction: float = 0.0,
    bias: float = 0.0,
) -> ComplexArray:
    """Basis D0, up/down, down/up, D1; ascending creation-operator order."""
    if not all(isfinite(x) for x in (hopping, interaction, bias)):
        raise ValueError("parameters must be finite")
    t, u, d = hopping, interaction, bias
    return np.array(
        [[u + d, -t, t, 0], [-t, 0, 0, -t], [t, 0, 0, t], [0, -t, t, u - d]],
        dtype=np.complex128,
    )


def _states(states: ComplexArray) -> None:
    if states.ndim < 1 or states.shape[-1] != 4 or states.size == 0:
        raise ValueError("states must have nonempty shape (..., 4)")
    if not np.isfinite(states).all():
        raise ValueError("states must be finite")
    if not np.allclose(np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError("states must be normalized")


def evolve(h: ComplexArray, states: ComplexArray, times: RealArray) -> ComplexArray:
    """Return (T, ..., 4) states using spectral phases; hbar=1."""
    if h.shape != (4, 4) or not np.isfinite(h).all():
        raise ValueError("h must be a finite 4 by 4 matrix")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0):
        raise ValueError("h must be Hermitian")
    _states(states)
    if times.ndim != 1 or times.size == 0 or np.iscomplexobj(times):
        raise ValueError("times must be a nonempty real vector")
    if not np.isfinite(times).all():
        raise ValueError("times must be finite")
    energies, vectors = np.linalg.eigh(h)
    coefficients = np.einsum("ij,...i->...j", vectors.conj(), states)
    phases = np.exp(-1j * times[:, None] * energies)
    return np.asarray(
        np.einsum("ij,tj,...j->t...i", vectors, phases, coefficients),
        dtype=np.complex128,
    )


def double_occupancy(states: ComplexArray) -> RealArray:
    _states(states)
    probabilities = np.abs(states) ** 2
    return np.asarray(probabilities[..., 0] + probabilities[..., 3])


def site_occupations(states: ComplexArray) -> RealArray:
    _states(states)
    weights = np.array([[2.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
    return np.asarray(np.abs(states) ** 2 @ weights)


def main() -> None:
    energies, vectors = np.linalg.eigh(hamiltonian(interaction=4))
    print("Energies:", energies.tolist())
    print("Ground-state double occupancy:", float(double_occupancy(vectors[:, 0])))
    print("Site occupations:", site_occupations(vectors[:, 0]).tolist())


if __name__ == "__main__":
    main()
