"""Independent occupation-number hopping construction for two particles."""

from itertools import combinations, combinations_with_replacement
from math import isfinite, pi, sqrt
from typing import Literal

import numpy as np
from numpy.typing import NDArray

type Statistics = Literal["distinguishable", "boson", "fermion"]
type ComplexArray = NDArray[np.complex128]
type RealArray = NDArray[np.float64]


def basis_pairs(num_sites: int, statistics: Statistics) -> list[tuple[int, int]]:
    if isinstance(num_sites, bool) or not isinstance(num_sites, int) or num_sites < 2:
        raise ValueError("num_sites must be an integer of at least two")
    if statistics == "distinguishable":
        return [(i, j) for i in range(num_sites) for j in range(num_sites)]
    if statistics == "boson":
        return list(combinations_with_replacement(range(num_sites), 2))
    if statistics == "fermion":
        return list(combinations(range(num_sites), 2))
    raise ValueError("unknown particle statistics")


def single_particle_hamiltonian(
    num_sites: int, *, hopping: float = 1.0, onsite: RealArray | None = None
) -> ComplexArray:
    """Nearest-neighbor open chain with a real site potential."""
    basis_pairs(num_sites, "distinguishable")
    if not isfinite(hopping):
        raise ValueError("hopping must be finite")
    if onsite is None:
        values = np.zeros(num_sites, dtype=np.float64)
    else:
        if onsite.shape != (num_sites,) or np.iscomplexobj(onsite):
            raise ValueError("onsite must be a real vector of length num_sites")
        values = np.asarray(onsite, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("onsite must be finite")
    h = np.zeros((num_sites, num_sites), dtype=np.complex128)
    for i in range(num_sites):
        h[i, i] = values[i]
        if i + 1 < num_sites:
            h[i, i + 1] = h[i + 1, i] = -hopping
    return h


def _validate_hamiltonian(h: ComplexArray) -> None:
    if (
        h.ndim != 2
        or h.shape[0] == 0
        or h.shape[0] != h.shape[1]
        or not np.isfinite(h).all()
    ):
        raise ValueError("Hamiltonian must be finite, nonempty, and square")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian")


def _validate_states(states: ComplexArray, dimension: int) -> None:
    if (
        states.ndim < 1
        or states.size == 0
        or states.shape[-1] != dimension
        or not np.isfinite(states).all()
    ):
        raise ValueError("states must have finite, nonempty shape (..., dimension)")
    if not np.allclose(np.sum(np.abs(states) ** 2, axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError("states must be normalized; no automatic renormalization")


def occupation_basis(num_sites: int, statistics: Statistics) -> list[tuple[int, ...]]:
    """Site counts for each basis pair; labeled pairs can have equal counts."""
    return [
        tuple(int(k == i) + int(k == j) for k in range(num_sites))
        for i, j in basis_pairs(num_sites, statistics)
    ]


def two_particle_hamiltonian(h: ComplexArray, statistics: Statistics) -> ComplexArray:
    """Build matrix elements by labeled moves or creation/annihilation rules."""
    _validate_hamiltonian(h)
    num_sites = h.shape[0]
    pairs = basis_pairs(num_sites, statistics)
    result = np.zeros((len(pairs), len(pairs)), dtype=np.complex128)
    if statistics == "distinguishable":
        lookup = {pair: index for index, pair in enumerate(pairs)}
        for column, (a, b) in enumerate(pairs):
            for target in range(num_sites):
                result[lookup[(target, b)], column] += h[target, a]
                result[lookup[(a, target)], column] += h[target, b]
        return result

    configurations = occupation_basis(num_sites, statistics)
    indices = {occupation: index for index, occupation in enumerate(configurations)}
    for column, occupation in enumerate(configurations):
        for source in range(num_sites):
            if occupation[source] == 0:
                continue
            after_removal = list(occupation)
            after_removal[source] -= 1
            for target in range(num_sites):
                if statistics == "boson":
                    factor = sqrt(occupation[source] * (after_removal[target] + 1))
                else:
                    if after_removal[target] != 0:
                        continue
                    parity = sum(occupation[:source]) + sum(after_removal[:target])
                    factor = float((-1) ** parity)
                final = after_removal.copy()
                final[target] += 1
                result[indices[tuple(final)], column] += h[target, source] * factor
    return result


def localized_state(
    num_sites: int, statistics: Statistics, pair: tuple[int, int]
) -> ComplexArray:
    pairs = basis_pairs(num_sites, statistics)
    if len(pair) != 2 or any(
        isinstance(i, bool) or not isinstance(i, int) for i in pair
    ):
        raise ValueError("pair must contain two integer sites")
    if pair not in pairs:
        raise ValueError("pair is absent from the canonical basis")
    state = np.zeros(len(pairs), dtype=np.complex128)
    state[pairs.index(pair)] = 1
    return state


def evolve(
    hamiltonian: ComplexArray,
    states: ComplexArray,
    times: RealArray,
    *,
    hbar: float = 1.0,
) -> ComplexArray:
    """Independent spectral evolution; output has shape (T, ..., D)."""
    _validate_hamiltonian(hamiltonian)
    _validate_states(states, hamiltonian.shape[0])
    if not isfinite(hbar) or hbar <= 0:
        raise ValueError("hbar must be finite and positive")
    if (
        times.ndim != 1
        or times.size == 0
        or np.iscomplexobj(times)
        or not np.isfinite(times).all()
    ):
        raise ValueError("times must be a finite, nonempty real vector")
    energies, vectors = np.linalg.eigh(hamiltonian)
    coefficients = np.einsum("ij,...j->...i", vectors.conj().T, states)
    phases = np.exp(-1j * times[:, None] * energies / hbar)
    return np.asarray(
        np.einsum("ij,tj,...j->t...i", vectors, phases, coefficients),
        dtype=np.complex128,
    )


def site_occupations(
    states: ComplexArray, num_sites: int, statistics: Statistics
) -> RealArray:
    counts = np.asarray(occupation_basis(num_sites, statistics), dtype=np.float64)
    _validate_states(states, len(counts))
    return np.asarray(np.abs(states) ** 2 @ counts, dtype=np.float64)


def coincidence_probability(
    states: ComplexArray, num_sites: int, statistics: Statistics
) -> RealArray:
    pairs = basis_pairs(num_sites, statistics)
    _validate_states(states, len(pairs))
    mask = np.asarray([i == j for i, j in pairs], dtype=np.float64)
    return np.asarray(np.abs(states) ** 2 @ mask, dtype=np.float64)


def main() -> None:
    h = single_particle_hamiltonian(2)
    times = np.array([0, pi / 8, pi / 4, pi / 2], dtype=np.float64)
    for statistics in ("distinguishable", "boson", "fermion"):
        h2 = two_particle_hamiltonian(h, statistics)
        history = evolve(h2, localized_state(2, statistics, (0, 1)), times)
        print(f"{statistics}: energies={np.linalg.eigvalsh(h2).tolist()}")
        print(
            f"  same-site probability={coincidence_probability(history, 2, statistics).tolist()}"
        )


if __name__ == "__main__":
    main()
