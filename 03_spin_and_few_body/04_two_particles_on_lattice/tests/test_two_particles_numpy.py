from math import pi, sqrt
from typing import cast

import numpy as np
import pytest
import two_particles_numpy_solution as model

STATISTICS: tuple[model.Statistics, ...] = ("distinguishable", "boson", "fermion")


@pytest.mark.parametrize("statistics", STATISTICS)
@pytest.mark.parametrize("num_sites", [2, 3, 5])
def test_exact_open_chain_spectrum(
    num_sites: int, statistics: model.Statistics
) -> None:
    h = model.single_particle_hamiltonian(num_sites, hopping=0.7)
    h2 = model.two_particle_hamiltonian(h, statistics)
    one = -1.4 * np.cos(pi * np.arange(1, num_sites + 1) / (num_sites + 1))
    expected = sorted(
        one[i] + one[j] for i, j in model.basis_pairs(num_sites, statistics)
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(h2), expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(h2, h2.conj().T, atol=1e-12, rtol=0)


def test_boson_enhancement_and_fermion_long_hop() -> None:
    h = model.single_particle_hamiltonian(2)
    expected = -sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    np.testing.assert_allclose(model.two_particle_hamiltonian(h, "boson"), expected)
    # Hop 0 -> 2 across occupied site 1: |0,1> -> |1,2> has a minus sign.
    long_hop = np.zeros((3, 3), dtype=np.complex128)
    long_hop[2, 0] = 0.3 + 0.4j
    long_hop[0, 2] = 0.3 - 0.4j
    fermion = model.two_particle_hamiltonian(long_hop, "fermion")
    assert fermion[2, 0] == pytest.approx(-0.3 - 0.4j)
    assert fermion[0, 2] == pytest.approx(-0.3 + 0.4j)


@pytest.mark.parametrize("statistics", STATISTICS)
def test_two_site_probabilities(statistics: model.Statistics) -> None:
    hopping, hbar = 0.8, 1.7
    times = np.linspace(-2, 4, 29)
    h2 = model.two_particle_hamiltonian(
        model.single_particle_hamiltonian(2, hopping=hopping), statistics
    )
    history = model.evolve(
        h2, model.localized_state(2, statistics, (0, 1)), times, hbar=hbar
    )
    factor = {"distinguishable": 0.5, "boson": 1.0, "fermion": 0.0}[statistics]
    expected = factor * np.sin(2 * hopping * times / hbar) ** 2
    np.testing.assert_allclose(
        model.coincidence_probability(history, 2, statistics), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        model.site_occupations(history, 2, statistics), 1, atol=1e-12
    )


@pytest.mark.parametrize("statistics", STATISTICS)
def test_uniform_shift_and_zero_hopping(statistics: model.Statistics) -> None:
    h = model.single_particle_hamiltonian(3)
    original = model.two_particle_hamiltonian(h, statistics)
    shifted = model.two_particle_hamiltonian(h + 0.37 * np.eye(3), statistics)
    np.testing.assert_allclose(
        shifted, original + 0.74 * np.eye(len(original)), atol=1e-12
    )
    onsite = np.array([-0.2, 0.5, 1.1])
    diagonal = model.two_particle_hamiltonian(
        model.single_particle_hamiltonian(3, hopping=0, onsite=onsite), statistics
    )
    expected = np.diag(
        [onsite[i] + onsite[j] for i, j in model.basis_pairs(3, statistics)]
    )
    np.testing.assert_allclose(diagonal, expected, atol=1e-12)


def test_pauli_exclusion_and_input_validation() -> None:
    with pytest.raises(ValueError, match="canonical"):
        model.localized_state(3, "fermion", (1, 1))
    with pytest.raises(ValueError, match="canonical"):
        model.localized_state(3, "boson", (2, 0))
    with pytest.raises(ValueError, match="statistics"):
        model.basis_pairs(3, cast(model.Statistics, "unknown"))
    for count in (0, 1, True):
        with pytest.raises(ValueError, match="num_sites"):
            model.basis_pairs(count, "boson")
    with pytest.raises(ValueError, match="finite"):
        model.single_particle_hamiltonian(3, hopping=float("nan"))
    with pytest.raises(ValueError, match="real vector"):
        model.single_particle_hamiltonian(3, onsite=np.zeros(2))
    with pytest.raises(ValueError, match="Hermitian"):
        model.two_particle_hamiltonian(
            np.array([[0, 1j], [1j, 0]], dtype=np.complex128), "boson"
        )
    h2 = model.two_particle_hamiltonian(model.single_particle_hamiltonian(2), "boson")
    initial = model.localized_state(2, "boson", (0, 1))
    with pytest.raises(ValueError, match="normalized"):
        model.evolve(h2, 2 * initial, np.array([0.0]))
    with pytest.raises(ValueError, match="hbar"):
        model.evolve(h2, initial, np.array([0.0]), hbar=0)
    with pytest.raises(ValueError, match="times"):
        model.evolve(h2, initial, np.array([np.nan]))
